"""Single-process recipe execution for model adapters."""

from itertools import cycle, islice
import json
from pathlib import Path
import random

import torch
from transformers import set_seed
from transformers.optimization import get_constant_schedule_with_warmup
import yaml

from omnitaskonomy.data.common import read_jsonl, sha256
from omnitaskonomy.gradients.artifacts import stable_seed
from omnitaskonomy.train import ROOT, _write_json, checkpoint_record, dataset_configs
from omnitaskonomy.umm import LossContext, ParameterInventory, adapter_provenance, load_adapter


def _pool(group):
    rows = list(read_jsonl(group["manifest"]))
    limit = group["num_used_data"]
    if not rows or (limit is not None and limit > len(rows)):
        raise ValueError(f"{group['manifest']}: requested pool {limit}, found {len(rows)} rows")
    if not group["shuffle_before_slice"] and limit is not None:
        rows = rows[:limit]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate uid in {group['manifest']}")
    random.Random(group["data_seed"]).shuffle(rows)
    if group["shuffle_before_slice"] and limit is not None:
        rows = rows[:limit]
    return rows


def _checkpoint_files(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


@torch.enable_grad()
def _train_stage(adapter, inventory, plan, stage, seed, model_seed, directory):
    receipt = inventory.apply("generation" if stage["freeze"] else "all")
    parameters = inventory.parameters()
    # Promote optimizer weights in place so tied parameters keep their identity.
    for parameter in parameters:
        if parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.float()
    unsupported = [f"{name} ({parameter.dtype})" for name, (parameter, _, _) in inventory.entries.items()
                   if parameter.requires_grad and parameter.dtype not in (torch.float32, torch.float64)]
    if unsupported:
        details = ", ".join(unsupported[:8])
        if len(unsupported) > 8:
            details += f", ... ({len(unsupported) - 8} more)"
        raise ValueError(f"Unsupported custom UMM trainable parameter dtypes: {details}")
    _write_json(directory / "logs/trainable_parameters.json", receipt)
    optimizer = torch.optim.AdamW(parameters, lr=plan["learning_rate"], betas=(0.9, 0.95),
                                  eps=1e-15, weight_decay=0)
    scheduler = get_constant_schedule_with_warmup(optimizer, stage["warmup_steps"])
    adapter.model.train()
    configs = dataset_configs(plan, stage, seed)
    for name, config in configs.items():
        (directory / name).write_text(yaml.safe_dump(config, sort_keys=False))
    phases = configs["dataset.yaml"].get("curriculum", [{
        "dataset_config_file": "dataset.yaml", "num_samples": stage["budget"],
        "target_dataset_name": stage["target_dataset"],
        "conditioning_dropout_prob": stage["condition_dropout"],
    }])
    microbatches = updates = visits = 0
    objective_visits = {"i2i": 0, "i2t": 0}
    phase_receipts = []
    optimizer.zero_grad(set_to_none=True)
    with (directory / "logs/progress.jsonl").open("w") as log:
        for phase_index, phase in enumerate(phases):
            groups = configs[phase["dataset_config_file"]]
            pools = {name: _pool(group) for name, group in groups.items()}
            streams = {name: cycle(rows) for name, rows in pools.items()}
            phase_visits = 0
            start_updates = updates
            while phase_visits < phase["num_samples"] and visits < stage["budget"]:
                counted = 0
                losses = {}
                for name, group in groups.items():
                    batch = list(islice(streams[name], group["fixed_batch_size"]))
                    context = LossContext(
                        manifest=Path(group["manifest"]),
                        seed=stable_seed(model_seed, f"{stage['name']}:{phase_index}:{microbatches}:{name}"),
                        condition_dropout=phase["conditioning_dropout_prob"] if group["kind"] == "i2i" else 0.0,
                        target_interpolation=group.get("target_interpolation", "bicubic"),
                        vit_transform=group["vit_image_transform_args"],
                        max_tokens=plan["max_tokens_per_sample"],
                    )
                    loss = adapter.loss(batch, group["kind"], context).mean
                    losses[name] = float(loss.detach())
                    (loss / plan["gradient_accumulation"]).backward()
                    objective_visits[group["kind"]] += len(batch)
                    if not phase["target_dataset_name"] or name == phase["target_dataset_name"]:
                        counted += len(batch)
                phase_visits += counted
                visits += counted
                microbatches += 1
                if microbatches % plan["gradient_accumulation"] == 0:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    log.write(json.dumps({"optimizer_updates": updates, "phase": phase_index,
                                          "sample_visits": visits, "last_microbatch_loss": losses,
                                          "next_learning_rate": scheduler.get_last_lr()[0]}) + "\n")
            phase_receipts.append({"phase": phase_index, "sample_visits": phase_visits,
                                   "optimizer_updates_before": start_updates,
                                   "optimizer_updates_after": updates,
                                   "condition_dropout": phase["conditioning_dropout_prob"],
                                   "pool_uids": {name: [row["uid"] for row in rows]
                                                 for name, rows in pools.items()}})
    checkpoint = directory / "checkpoints/final"
    checkpoint.mkdir(parents=True)
    adapter.save_checkpoint(checkpoint)
    files = _checkpoint_files(checkpoint)
    if not files:
        raise RuntimeError(f"Adapter did not save checkpoint files in {checkpoint}")
    return {"checkpoint": str(checkpoint), "checkpoint_files_sha256": files,
            "sample_visits": visits, "objective_visits": objective_visits,
            "optimizer_updates": updates,
            "pending_microbatches": microbatches % plan["gradient_accumulation"],
            "phases": phase_receipts}


def run(plan):
    output = Path(plan["output_dir"])
    provenance = {
        "plan": plan,
        "manifest_sha256": {kind: sha256(plan[kind]) for kind in ("i2i_manifest", "i2t_manifest") if plan[kind]},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "omnitaskonomy/train.py", ROOT / "omnitaskonomy/umm.py",
            ROOT / "configs/train" / f"{plan['suite']}.yaml",
        )},
    }
    saved_plan = output / "plan.json"
    if saved_plan.exists() and json.loads(saved_plan.read_text()) != provenance:
        raise ValueError(f"{output} already belongs to another configuration or manifest; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(saved_plan, provenance)
    records = []
    for seed, model_seed in zip(plan["seeds"], plan["model_seeds"]):
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_records = []
        initialization = plan["stage1_checkpoint"]
        for index, stage in enumerate(plan["stages"], 2 if initialization else 1):
            directory = seed_dir / f"stage{index}_{stage['name']}"
            receipt_path = directory / "logs/completion.json"
            if not receipt_path.is_file() and directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Incomplete stage exists: {directory}; use a new output directory")
            set_seed(model_seed)
            adapter = load_adapter(plan["adapter"], plan["model_path"], checkpoint=initialization,
                                   device=plan["device"], options=plan["adapter_options"])
            initial = adapter_provenance(adapter, plan["adapter"], plan["adapter_options"])
            if receipt_path.is_file():
                result = json.loads(receipt_path.read_text())
                if result["initialization_provenance"] != initial:
                    raise ValueError(f"Checkpoint initialization changed: {receipt_path}")
            else:
                (directory / "logs").mkdir(parents=True)
                inventory = ParameterInventory(adapter.model, adapter.parameter_specs())
                print(f"Training seed {seed}, {stage['name']}; log: {directory / 'logs/progress.jsonl'}", flush=True)
                result = _train_stage(adapter, inventory, plan, stage, seed, model_seed, directory)
                result["initialization_provenance"] = initial
                _write_json(receipt_path, result)
                del inventory
            checkpoint = Path(result["checkpoint"])
            actual_files = _checkpoint_files(checkpoint)
            if not actual_files or actual_files != result["checkpoint_files_sha256"]:
                raise ValueError(f"Saved checkpoint files changed: {checkpoint}")
            if result["sample_visits"] < stage["budget"] or result["optimizer_updates"] < 1:
                raise RuntimeError(f"Incomplete training accounting: {receipt_path}")
            trainable = directory / "logs/trainable_parameters.json"
            record = checkpoint_record(plan, stage, seed, initialization or plan["model_path"], result)
            record.update(adapter=plan["adapter"], adapter_options=plan["adapter_options"], device=plan["device"],
                          objective_visits=result["objective_visits"],
                          checkpoint_files_sha256=result["checkpoint_files_sha256"],
                          trainable_parameters=str(trainable), trainable_parameters_sha256=sha256(trainable))
            records.append(record)
            seed_records.append(record)
            initialization = str(checkpoint)
            del adapter
        _write_json(seed_dir / "run.json", {"schema_version": 1, "task": plan["task"],
                    "recipe": plan["recipe"], "seed": seed, "status": "complete",
                    "base_model_path": plan["model_path"], "final_checkpoint": initialization,
                    "stages": seed_records})
    destination = output / "checkpoints.json"
    _write_json(destination, {"schema_version": 1, "records": records})
    return destination
