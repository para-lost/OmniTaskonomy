"""Run the checked-in paper experiment lists with the ordinary training launcher."""

import argparse
import json
from pathlib import Path
import re

from omnitaskonomy.data.common import sha256
from omnitaskonomy.umm import read_options
from omnitaskonomy.train import (
    _write_json, arguments_from_settings, build_plan, parser as training_parser, preview_commands, run,
)


def load_experiment(path):
    spec = json.loads(Path(path).read_text())
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema_version")
    if not re.fullmatch(r"[a-z0-9_-]+", spec["id"]):
        raise ValueError("Experiment ID must be a path-safe name")
    seen = {}
    for job in spec["jobs"]:
        name = job["id"]
        if not re.fullmatch(r"[a-z0-9_-]+", name) or name in seen:
            raise ValueError(f"Invalid or duplicate experiment job: {name}")
        dependency = job.get("stage1_from")
        if dependency is not None and dependency not in seen:
            raise ValueError(f"{name}: stage1_from must identify an earlier job")
        if dependency is not None:
            source = {**spec["training"], **seen[dependency]["training"]}
            if not source.get("stop_after_stage1") or len(source["seeds"]) != 1:
                raise ValueError(f"{name}: shared stage-1 job must train one I2I checkpoint")
        seen[name] = job
    return spec


def select_jobs(spec, names=None):
    by_name = {job["id"]: job for job in spec["jobs"]}
    selected = set(names) if names else set(by_name)
    unknown = selected - by_name.keys()
    if unknown:
        raise ValueError(f"Unknown experiment jobs: {sorted(unknown)}")
    for name in list(selected):
        dependency = by_name[name].get("stage1_from")
        while dependency is not None:
            selected.add(dependency)
            dependency = by_name[dependency].get("stage1_from")
    return [job for job in spec["jobs"] if job["id"] in selected]


def job_plan(spec, job, data_root, model_path, output_dir, stage1_checkpoint=None, nproc=None,
             adapter=None, adapter_options=None, device="cuda:0"):
    settings = {**spec["training"], **job["training"]}
    for field in ("i2i_manifest", "i2t_manifest"):
        if field in settings:
            settings[field] = Path(data_root) / settings[field]
    settings.update(model_path=model_path, output_dir=Path(output_dir) / spec["id"] / job["id"])
    if job.get("stage1_from"):
        if stage1_checkpoint is None:
            raise ValueError(f"{job['id']}: missing resolved stage-1 checkpoint")
        settings["stage1_checkpoint"] = stage1_checkpoint
    if nproc is not None:
        settings["nproc_per_node"] = nproc
    if adapter is not None:
        settings.update(adapter=adapter, device=device)
    if adapter_options is not None:
        settings["adapter_options"] = adapter_options
    plan = build_plan(training_parser().parse_args(arguments_from_settings(settings)))
    plan["experiment"] = {"id": spec["id"], "job": job["id"], "paper_commit": spec["paper_commit"]}
    return plan


def execute(spec, config_path, data_root, model_path, output_dir, *, jobs=None, dry_run=False, nproc=None,
            adapter=None, adapter_options=None, device="cuda:0"):
    selected = select_jobs(spec, jobs)
    completed, preview = {}, []
    if not dry_run:
        directory = Path(output_dir) / spec["id"]
        directory.mkdir(parents=True, exist_ok=True)
        identity = {"experiment": spec, "config_sha256": sha256(config_path),
                    "data_root": str(Path(data_root).resolve()), "model_path": str(Path(model_path).resolve()),
                    "nproc_override": nproc}
        if adapter:
            identity.update(adapter=adapter, device=device, adapter_options=read_options(adapter_options))
        receipt = directory / "experiment.json"
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise ValueError(f"{directory} already belongs to another experiment configuration")
        _write_json(receipt, identity)
    for job in selected:
        checkpoint = completed[job["stage1_from"]] if job.get("stage1_from") else None
        plan = job_plan(spec, job, data_root, model_path, output_dir, checkpoint, nproc,
                        adapter, adapter_options, device)
        if dry_run:
            preview.append({"id": job["id"], "plan": plan, "commands": preview_commands(plan)})
            if plan["stop_after_stage1"]:
                if len(plan["seeds"]) != 1:
                    raise ValueError("A shared stage-1 job must have exactly one seed")
                completed[job["id"]] = str(Path(plan["output_dir"]) / "<stage1-final-checkpoint>")
        else:
            result = run(plan)
            if plan["stop_after_stage1"]:
                records = json.loads(result.read_text())["records"]
                if len(records) != 1 or records[0]["stage"] != "i2i":
                    raise ValueError("A shared stage-1 job must produce one I2I checkpoint")
                completed[job["id"]] = records[0]["checkpoint"]
    return {"experiment": spec["id"], "paper_commit": spec["paper_commit"],
            "selected_jobs": [job["id"] for job in selected], "jobs": preview}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", help="Custom UMM factory as package.module:function")
    parser.add_argument("--adapter-options", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--jobs", nargs="+", help="Job IDs; required stage-1 jobs are included automatically")
    parser.add_argument("--nproc-per-node", type=int, help="Override GPU topology (can change numerical results)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = execute(load_experiment(args.config), args.config, args.data_root, args.model_path,
                     args.output_dir, jobs=args.jobs, dry_run=args.dry_run, nproc=args.nproc_per_node,
                     adapter=args.adapter, adapter_options=args.adapter_options, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
