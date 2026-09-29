"""Train OmniTaskonomy recipes from portable manifests."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from omnitaskonomy.data.common import sha256
from omnitaskonomy.umm import read_options

ROOT = Path(__file__).resolve().parents[1]
BAGEL = ROOT / "Bagel"
RECIPES = {
    "r1": "i2t-only", "r2": "i2i-to-i2t", "r3": "mixed-to-i2t",
    "r4": "frozen-i2i-to-mixed", "r5": "mixed", "r6": "i2i-to-mixed",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="Task label stored with the checkpoints")
    p.add_argument("--recipe", type=str.lower, default="i2i-to-i2t", choices=[*RECIPES, *RECIPES.values()])
    p.add_argument("--suite", choices=["transfer", "controlled"], default="transfer")
    p.add_argument("--i2i-manifest", type=Path)
    p.add_argument("--i2i-target-interpolation", choices=["bicubic", "nearest"], default="bicubic",
                   help="Target resize interpolation; use nearest for semantic segmentation")
    p.add_argument("--i2t-manifest", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--adapter", help="Custom UMM factory as package.module:function")
    p.add_argument("--adapter-options", type=Path, help="JSON object passed to the UMM factory")
    p.add_argument("--device", default="cuda:0", help="Device for a custom UMM adapter")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    model_rng = p.add_mutually_exclusive_group()
    model_rng.add_argument("--global-seed", type=int, default=4396,
                           help="Fixed model RNG for controlled; transfer adds (seed - 42)")
    model_rng.add_argument("--model-seeds", type=int, nargs="+",
                           help="Explicit model RNG seeds, one per --seeds entry")
    p.add_argument("--i2i-seed", type=int, default=42, help="I2I pool/order seed, fixed across runs")
    p.add_argument("--stage1-checkpoint", type=Path,
                   help="Reuse a saved I2I checkpoint directory and train only stage 2")
    p.add_argument("--nproc-per-node", type=int, default=4)
    p.add_argument("--batch-size", type=int, help="Per-GPU I2T/pure-I2I microbatch size")
    p.add_argument("--gradient-accumulation", type=int)
    p.add_argument("--cpu-offload", action="store_true", help="Enable FSDP CPU offload")
    p.add_argument("--i2i-budget", type=int)
    p.add_argument("--i2t-budget", type=int)
    p.add_argument("--i2t-pool", type=int, help="Number of distinct I2T rows used, before repetition")
    p.add_argument("--i2i-pool", type=int, help="Cap I2I rows; default uses the complete manifest")
    p.add_argument("--condition-dropout", type=float)
    p.add_argument("--base-weights", choices=["ema", "model"], default="ema")
    p.add_argument("--freeze-input-ln", action="store_true", help="Freeze understanding input RMSNorm during I2I stage only")
    p.add_argument("--stage1-freeze-last-half-llm", action="store_true",
                   help="Freeze the last half of LLM layers during the I2I stage (instance-alignment protocol)")
    p.add_argument("--stop-after-stage1", action="store_true",
                   help="Train only the initial I2I stage of r2/r4/r6, for shared-checkpoint experiments")
    p.add_argument("--max-tokens-per-sample", type=int, default=15000)
    p.add_argument("--dry-run", action="store_true", help="Print commands/configs without loading CUDA or creating outputs")
    return p


def build_plan(args):
    if args.adapter_options and not args.adapter:
        raise ValueError("--adapter-options requires --adapter")
    if args.adapter:
        if args.nproc_per_node != 1:
            raise ValueError("Custom UMM training requires --nproc-per-node 1")
        if args.cpu_offload or args.freeze_input_ln or args.stage1_freeze_last_half_llm:
            raise ValueError("Custom UMM training does not support BAGEL CPU offload or layer-specific freezing")
        if args.base_weights != "ema":
            raise ValueError("--base-weights is a BAGEL launcher option; select custom weights through adapter options")
    recipe = RECIPES.get(args.recipe, args.recipe)
    defaults = yaml.safe_load((ROOT / "configs/train" / f"{args.suite}.yaml").read_text())
    if args.max_tokens_per_sample < 1:
        raise ValueError("--max-tokens-per-sample must be positive")
    if args.nproc_per_node < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("GPU count must be positive and seeds must be distinct")
    model_seeds = args.model_seeds
    if model_seeds is None:
        model_seeds = [args.global_seed + (seed - 42 if args.suite == "transfer" else 0)
                       for seed in args.seeds]
    if len(model_seeds) != len(args.seeds):
        raise ValueError("--model-seeds must contain one value per --seeds entry")
    if min([*args.seeds, *model_seeds, args.i2i_seed]) < 0:
        raise ValueError("Data and model RNG seeds must be nonnegative")
    if args.stage1_checkpoint and recipe not in {"i2i-to-i2t", "frozen-i2i-to-mixed", "i2i-to-mixed"}:
        raise ValueError("--stage1-checkpoint requires an I2I first-stage recipe (r2/r4/r6)")
    if args.stop_after_stage1 and (args.stage1_checkpoint or recipe not in {"i2i-to-i2t", "frozen-i2i-to-mixed", "i2i-to-mixed"}):
        raise ValueError("--stop-after-stage1 requires a new I2I first stage (r2/r4/r6)")
    if args.stage1_freeze_last_half_llm and (args.stage1_checkpoint or recipe not in {"i2i-to-i2t", "i2i-to-mixed"}):
        raise ValueError("--stage1-freeze-last-half-llm requires a new, unfrozen I2I first stage")
    needs_i2i = recipe != "i2t-only" and not (args.stage1_checkpoint and recipe == "i2i-to-i2t")
    if needs_i2i and args.i2i_manifest is None:
        raise ValueError(f"{recipe} requires --i2i-manifest")
    if args.stage1_checkpoint and args.freeze_input_ln:
        raise ValueError("--freeze-input-ln cannot change a reused stage-1 checkpoint")
    if args.freeze_input_ln and recipe not in {"i2i-to-i2t", "i2i-to-mixed"}:
        raise ValueError("--freeze-input-ln requires an unfrozen I2I first stage")
    batch = args.batch_size if args.batch_size is not None else min(defaults["batch_size_per_gpu"], 64 // args.nproc_per_node)
    if batch <= 0:
        raise ValueError("--batch-size must be positive")
    microbatch = batch * args.nproc_per_node
    accumulation = args.gradient_accumulation
    if accumulation is None:
        if 64 % microbatch:
            raise ValueError("GPU count × batch size must divide 64; otherwise set --gradient-accumulation explicitly")
        accumulation = 64 // microbatch
    if accumulation < 1:
        raise ValueError("--gradient-accumulation must be positive")
    if args.cpu_offload and accumulation > 1:
        raise ValueError("--cpu-offload requires gradient accumulation 1; use an effective batch of 64 in one microbatch")
    i2i_budget = args.i2i_budget if args.i2i_budget is not None else defaults["i2i_budget"]
    i2t_budget = args.i2t_budget if args.i2t_budget is not None else defaults["i2t_budget"]
    pool = args.i2t_pool if args.i2t_pool is not None else defaults["i2t_pool"]
    if min(i2i_budget, i2t_budget, pool) < 1 or (args.i2i_pool is not None and args.i2i_pool < 1):
        raise ValueError("Sample budgets and pools must be positive")
    if i2t_budget < microbatch * accumulation or (recipe != "i2t-only" and i2i_budget < microbatch * accumulation):
        raise ValueError("Each stage budget must cover at least one optimizer update")
    requested_i2t_budget = i2t_budget
    effective_batch = microbatch * accumulation
    i2i_stage_budget = i2i_budget
    if args.suite == "controlled":
        if effective_batch != defaults["effective_i2t_batch"]:
            raise ValueError("Controlled recipes require an effective I2T batch of 64")
        # Complete final updates without changing the selected I2I pool size.
        i2t_budget = ((i2t_budget + effective_batch - 1) // effective_batch) * effective_batch
        i2i_stage_budget = ((i2i_budget + effective_batch - 1) // effective_batch) * effective_batch
    dropout = defaults["condition_dropout"]
    if args.condition_dropout is not None:
        dropout = args.condition_dropout
    if not 0 <= dropout < 1:
        raise ValueError("Condition dropout must be in [0, 1)")

    image_transform = {"image_stride": 16, "max_image_size": 512, "min_image_size": 256}
    vit_transform = {"image_stride": 14, "max_image_size": 518, "min_image_size": 224}

    def group(kind, size):
        manifest = args.i2i_manifest if kind == "i2i" else args.i2t_manifest
        config = {"manifest": str(manifest.resolve()), "kind": kind,
                "num_used_data": (args.i2i_pool if args.i2i_pool is not None else
                                  (i2i_budget if args.suite == "controlled" else None)) if kind == "i2i" else pool,
                "shuffle_before_slice": args.suite == "controlled",
                "image_transform_args": image_transform if kind == "i2i" else vit_transform,
                "vit_image_transform_args": vit_transform,
                "is_mandatory": True, "fixed_batch_size": size, "weight": 1.0}
        if kind == "i2i":
            config["target_interpolation"] = args.i2i_target_interpolation
        return config

    pure_i2i = {"i2i": group("i2i", batch)} if args.i2i_manifest else None
    pure_i2t = {"i2t": group("i2t", batch)}
    stages = []

    def add(name, config, budget, target, frozen=False, phases=None):
        generation = name != "i2t"
        stages.append({"name": name, "dataset_config": config, "phase_configs": phases or {},
                       "budget": budget, "target_dataset": target,
                       "visual_gen": generation, "freeze": frozen,
                       "freeze_last_half_llm": bool(args.stage1_freeze_last_half_llm and name == "i2i"),
                       "freeze_input_ln": bool(args.freeze_input_ln and name == "i2i"),
                       "condition_dropout": dropout if generation else 0.0,
                       "warmup_steps": defaults["warmup_i2i" if name == "i2i" else "warmup_i2t"]})

    if recipe in {"i2i-to-i2t", "frozen-i2i-to-mixed", "i2i-to-mixed"} and not args.stage1_checkpoint:
        add("i2i", pure_i2i, i2i_stage_budget, "", recipe == "frozen-i2i-to-mixed")
    if args.stop_after_stage1:
        pass
    elif recipe in {"i2t-only", "i2i-to-i2t"}:
        add("i2t", pure_i2t, i2t_budget, "")
    else:
        if args.suite == "controlled":
            if i2i_budget not in defaults["mixed_generation_batch"]:
                raise ValueError("Controlled mixed recipes require I2I budget 1876, 9380, 30016, or 100000")
            batch_map = "curriculum_generation_batch" if recipe == "mixed-to-i2t" else "mixed_generation_batch"
            generation_batch = defaults[batch_map][i2i_budget]
        else:
            generation_batch = microbatch * accumulation
            if recipe == "mixed-to-i2t":
                generation_batch *= 2
        divisor = args.nproc_per_node * accumulation
        if generation_batch % divisor:
            raise ValueError("Mixed generation batch cannot be divided across GPUs/accumulation; use fewer GPUs or a larger microbatch")
        mixed = {"i2i": group("i2i", generation_batch // divisor), "i2t": group("i2t", batch)}
        if recipe == "mixed-to-i2t":
            first_half = i2t_budget // 2
            if args.suite == "controlled":
                first_half = ((first_half + effective_batch - 1) // effective_batch) * effective_batch
            if min(first_half, i2t_budget - first_half) < effective_batch:
                raise ValueError("Mixed→I2T needs at least one optimizer update in each phase")
            phases = {"mixed.yaml": mixed, "i2t.yaml": pure_i2t}
            curriculum = {"curriculum": [
                {"dataset_config_file": "mixed.yaml", "num_samples": first_half, "target_dataset_name": "i2t",
                 "conditioning_dropout_prob": dropout},
                {"dataset_config_file": "i2t.yaml", "num_samples": i2t_budget - first_half, "target_dataset_name": "i2t",
                 "conditioning_dropout_prob": 0.0},
            ]}
            add("mixed_to_i2t", curriculum, i2t_budget, "i2t", phases=phases)
        else:
            add("mixed", mixed, i2t_budget, "i2t")
    plan = {"task": args.task, "recipe": recipe, "suite": args.suite,
            "model_path": str(args.model_path.resolve()), "output_dir": str(args.output_dir.resolve()),
            "i2i_manifest": str(args.i2i_manifest.resolve()) if args.i2i_manifest else None,
            "i2t_manifest": str(args.i2t_manifest.resolve()), "seeds": args.seeds,
            "model_seeds": model_seeds, "i2i_seed": args.i2i_seed,
            "stage1_checkpoint": str(args.stage1_checkpoint.resolve()) if args.stage1_checkpoint else None,
            "stop_after_stage1": args.stop_after_stage1,
            "requested_i2i_budget": i2i_budget,
            "requested_i2t_budget": requested_i2t_budget,
            "nproc_per_node": args.nproc_per_node, "cpu_offload": args.cpu_offload,
            "batch_size": batch, "gradient_accumulation": accumulation,
            "learning_rate": defaults["learning_rate"], "base_weights": args.base_weights,
            "max_tokens_per_sample": args.max_tokens_per_sample, "stages": stages}
    if args.adapter:
        options = read_options(args.adapter_options)
        plan.update(adapter=args.adapter, adapter_options=options, device=args.device)
        settings = {key: value.resolve() if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "dry_run" and value is not None}
        if args.model_seeds is not None:
            del settings["global_seed"]
        plan["command"] = [sys.executable, str(ROOT / "scripts/train.py"), *arguments_from_settings(settings)]
    return plan


def arguments_from_settings(settings):
    argv = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *map(str, value)])
        else:
            argv.extend([flag, str(value)])
    return argv


def dataset_configs(plan, stage, seed):
    configs = deepcopy({"dataset.yaml": stage["dataset_config"], **stage["phase_configs"]})
    for groups in configs.values():
        if "curriculum" in groups:
            continue
        for group in groups.values():
            group["data_seed"] = plan["i2i_seed"] if group["kind"] == "i2i" else seed
    return configs


def command(plan, stage, seed, stage_dir, initialization):
    flags = {
        "dataset_config_file": stage_dir / "dataset.yaml",
        "model_path": plan["model_path"], "resume_from": initialization,
        "finetune_from_hf": True,
        "finetune_from_ema": (not plan["stage1_checkpoint"] and initialization == plan["model_path"]
                              and plan["base_weights"] == "ema"),
        "auto_resume": False, "resume_model_only": True,
        "train_with_no_ema": True, "save_ema_only": False,
        "visual_und": True, "visual_gen": stage["visual_gen"],
        "freeze_shared_for_i2i": stage["freeze"],
        "freeze_vit": stage["freeze"],
        "freeze_und": stage["freeze"], "freeze_vae": True,
        "freeze_llm_input_ln": stage["freeze_input_ln"], "train_only_input_ln": False,
        "gradient_accumulation_steps": plan["gradient_accumulation"],
        "lr": plan["learning_rate"], "warmup_steps": stage["warmup_steps"], "lr_scheduler": "constant",
        "num_shard": plan["nproc_per_node"], "num_replicate": 1,
        "sharding_strategy": "FULL_SHARD", "num_workers": 1, "cpu_offload": plan["cpu_offload"],
        "data_seed": seed, "global_seed": plan["model_seeds"][plan["seeds"].index(seed)],
        "total_data_num": stage["budget"], "target_dataset_name": stage["target_dataset"],
        "total_steps": 2147483647, "save_every": 2147483647, "log_every": 10,
        "text_cond_dropout_prob": stage["condition_dropout"],
        "vae_cond_dropout_prob": stage["condition_dropout"],
        "vit_cond_dropout_prob": stage["condition_dropout"],
        "checkpoint_dir": stage_dir / "checkpoints", "results_dir": stage_dir / "logs",
        "max_latent_size": 64, "expected_num_tokens": plan["max_tokens_per_sample"],
        "max_num_tokens": plan["max_tokens_per_sample"],
        "max_num_tokens_per_sample": plan["max_tokens_per_sample"],
        "wandb_project": "omnitaskonomy", "wandb_offline": True,
        "wandb_name": f"{plan['task']}_{plan['recipe']}_{seed}_{stage['name']}",
    }
    if stage["freeze_last_half_llm"]:
        flags["freeze_llm_layers_ratio_from_end"] = 0.5
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
           f"--nproc_per_node={plan['nproc_per_node']}", str(BAGEL / "train/pretrain_unified_navit.py")]
    for name, value in flags.items():
        cmd.extend(["--" + name, str(value)])
    return cmd


def _write_json(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_record(plan, stage, seed, initialization, result):
    return {"task": plan["task"], "recipe": plan["recipe"], "seed": seed,
            "model_seed": plan["model_seeds"][plan["seeds"].index(seed)],
            "i2i_seed": plan["i2i_seed"], "i2t_seed": seed,
            "initialization": initialization, "stage": stage["name"],
            "checkpoint": str(Path(result["checkpoint"])), "model_path": plan["model_path"],
            "i2i_manifest": plan["i2i_manifest"], "i2t_manifest": plan["i2t_manifest"],
            "sample_visits": result["sample_visits"], "optimizer_updates": result["optimizer_updates"],
            "pending_microbatches": result["pending_microbatches"]}


def run(plan):
    if "adapter" not in plan:
        base = Path(plan["model_path"])
        required = ["llm_config.json", "vit_config.json"]
        if not plan["stage1_checkpoint"]:
            required.append(plan["base_weights"] + ".safetensors")
        if any(stage["visual_gen"] for stage in plan["stages"]):
            required.append("ae.safetensors")
        for name in required:
            if not (base / name).is_file():
                raise FileNotFoundError(f"Missing base model file: {base / name}")
    if plan["suite"] == "controlled":
        from omnitaskonomy.data.recipe import TASKS, prepare_recipe

        manifests = {kind: Path(plan[f"{kind}_manifest"]) for kind in ("i2i", "i2t")
                     if plan[f"{kind}_manifest"]}
        if plan["task"] in TASKS and any(not path.is_file() for path in manifests.values()):
            prepare_recipe(plan["task"], manifests)
    if (plan["suite"] == "transfer" and plan["i2i_manifest"]
            and any(stage["visual_gen"] for stage in plan["stages"])):
        from omnitaskonomy.data.taskonomy import TASKS, prepare_taskonomy

        manifest = Path(plan["i2i_manifest"])
        if plan["task"] in TASKS and not manifest.is_file():
            prepare_taskonomy(plan["task"], manifest)
    if "adapter" in plan:
        from omnitaskonomy.umm_training import run as run_umm
        return run_umm(plan)
    hashes = {kind: sha256(plan[kind]) for kind in ["i2i_manifest", "i2t_manifest"] if plan[kind]}
    stage1_hash = sha256(Path(plan["stage1_checkpoint"]) / "model.safetensors") if plan["stage1_checkpoint"] else None
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    saved_plan = output / "plan.json"
    source_paths = [Path(__file__), ROOT / "configs/train" / f"{plan['suite']}.yaml",
                    BAGEL / "train/pretrain_unified_navit.py", BAGEL / "train/freeze_policy.py",
                    ROOT / "omnitaskonomy/datasets.py"]
    provenance = {"plan": plan, "manifest_sha256": hashes, "stage1_checkpoint_sha256": stage1_hash,
                  "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}}
    if saved_plan.exists() and json.loads(saved_plan.read_text()) != provenance:
        raise ValueError(f"{output} already belongs to another configuration or manifest; choose a new output directory")
    _write_json(saved_plan, provenance)
    records = []
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(BAGEL), str(ROOT)])
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    for seed in plan["seeds"]:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_records = []
        initialization = plan["stage1_checkpoint"] or plan["model_path"]
        for index, stage in enumerate(plan["stages"], 2 if plan["stage1_checkpoint"] else 1):
            stage_dir = seed_dir / f"stage{index}_{stage['name']}"
            receipt = stage_dir / "logs/completion.json"
            cmd = command(plan, stage, seed, stage_dir, initialization)
            if not receipt.is_file():
                if stage_dir.exists() and any(stage_dir.iterdir()):
                    raise RuntimeError(f"Incomplete stage exists: {stage_dir}; use a new output directory")
                stage_dir.mkdir(exist_ok=True)
                for name, config in dataset_configs(plan, stage, seed).items():
                    (stage_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))
                _write_json(stage_dir / "command.json", {"argv": cmd, "cwd": str(BAGEL)})
                with (stage_dir / "launcher.log").open("w") as log:
                    print(f"Training seed {seed}, {stage['name']}; log: {stage_dir / 'launcher.log'}", flush=True)
                    subprocess.run(cmd, cwd=BAGEL, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(receipt.read_text())
            checkpoint = Path(result["checkpoint"])
            if not (checkpoint / "model.safetensors").is_file():
                raise RuntimeError(f"Completion receipt has no saved model: {checkpoint}")
            if result["sample_visits"] < stage["budget"] or result["optimizer_updates"] < 1:
                raise RuntimeError(f"Incomplete training accounting: {receipt}")
            record = checkpoint_record(plan, stage, seed, initialization, result)
            record["stage1_checkpoint_sha256"] = stage1_hash
            trainable_receipt = stage_dir / "logs/trainable_parameters.json"
            if trainable_receipt.is_file():
                record["trainable_parameters"] = str(trainable_receipt)
                record["trainable_parameters_sha256"] = sha256(trainable_receipt)
            records.append(record)
            seed_records.append(record)
            initialization = str(checkpoint)
        _write_json(seed_dir / "run.json", {"schema_version": 1, "task": plan["task"],
                    "recipe": plan["recipe"], "seed": seed, "status": "complete",
                    "base_model_path": plan["model_path"], "final_checkpoint": initialization,
                    "stages": seed_records})
    _write_json(output / "checkpoints.json", {"schema_version": 1, "records": records})
    return output / "checkpoints.json"


def preview_commands(plan):
    if "adapter" in plan:
        return [plan["command"]]
    commands = []
    for seed in plan["seeds"]:
        initialization = plan["stage1_checkpoint"] or plan["model_path"]
        for index, stage in enumerate(plan["stages"], 2 if plan["stage1_checkpoint"] else 1):
            directory = Path(plan["output_dir"]) / f"seed_{seed}" / f"stage{index}_{stage['name']}"
            commands.append(command(plan, stage, seed, directory, initialization))
            initialization = str(directory / "checkpoints/<final-checkpoint>")
    return commands


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps({"plan": plan, "commands": preview_commands(plan)}, indent=2))
        return
    print(run(plan))


if __name__ == "__main__":
    main()
