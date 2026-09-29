"""Connect completed paper training/evaluation runs to question-level analysis."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from omnitaskonomy.data.common import sha256
from omnitaskonomy.evaluate import (
    BENCHMARKS, MODEL_NAME, UMM_MODEL_NAME, judge_diagnostics, scored_workbook, select_checkpoints,
)
from omnitaskonomy.experiments import job_plan, load_experiment
from omnitaskonomy.taxonomy import DEFAULT_DIRECTORY, load_taxonomy


def collect(experiment_path, training_root, evaluation_root, taxonomy_dir, output, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    output, training_root, evaluation_root = Path(output).resolve(), Path(training_root), Path(evaluation_root)
    experiment_hash = sha256(experiment_path)
    spec = load_experiment(experiment_path)
    taxonomy = load_taxonomy(taxonomy_dir)
    allowed = {task["id"] for task in taxonomy["tasks"] if task["modality"] == "i2i"}
    jobs = [job for job in spec["jobs"] if not {**spec["training"], **job["training"]}.get("stop_after_stage1", False)]
    models = {job["id"]: {"task_id": job["task_id"]} for job in jobs}
    baselines = [name for name, value in models.items() if value["task_id"] is None]
    if len(baselines) != 1 or len(jobs) < 2 or any(v["task_id"] not in allowed for v in models.values() if v["task_id"]):
        raise ValueError("Experiment must contain one I2T baseline and released I2I source tasks")
    audit = []

    def read(path):
        digest = sha256(path)
        result = json.loads(Path(path).read_text())
        if sha256(path) != digest:
            raise ValueError(f"Input changed while reading: {path}")
        audit.append({"path": os.path.relpath(Path(path).resolve(), output.parent), "sha256": digest})
        return result

    experiment = read(training_root / "experiment.json")
    if experiment["experiment"] != spec or experiment["config_sha256"] != experiment_hash:
        raise ValueError("Training receipt belongs to a different experiment configuration")
    runs = []
    for job in jobs:
        name = job["id"]
        checkpoint_file = training_root / name / "checkpoints.json"
        trained = read(checkpoint_file)
        final = {row["seed"]: row for row in trained["records"]}
        selected = select_checkpoints(SimpleNamespace(run_file=checkpoint_file, model_path=None, checkpoint=None,
                                                      seeds=seeds, adapter=None, adapter_options=None, device=None))
        expected = job_plan(spec, job, Path(experiment["data_root"]), Path(experiment["model_path"]),
                            training_root.parent,
                            Path(final[seeds[0]]["initialization"]) if job.get("stage1_from") else None,
                            experiment["nproc_override"])
        if not set(seeds) <= set(expected["seeds"]):
            raise ValueError(f"Requested seeds are absent from the experiment job: {name}")
        for seed in seeds:
            record = final[seed]
            identity = (record["task"], record["recipe"], record["stage"])
            wanted = (expected["task"], expected["recipe"], expected["stages"][-1]["name"])
            if identity != wanted:
                raise ValueError(f"Training job/final-stage identity differs: {name}/{seed}: {identity} != {wanted}")
            if record["model_seed"] != expected["model_seeds"][expected["seeds"].index(seed)]:
                raise ValueError(f"Training model RNG differs from the experiment: {name}/{seed}")
        evaluated = read(evaluation_root / name / "summary.json")
        results = {row["seed"]: row for row in evaluated["runs"]}
        if len(results) != len(evaluated["runs"]):
            raise ValueError(f"Duplicate evaluation seeds: {name}")
        if set(seeds) - results.keys():
            raise ValueError(f"Missing evaluation seeds: {name}: {sorted(set(seeds) - results.keys())}")
        for chosen in selected:
            seed = chosen["seed"]
            result = results[seed]
            if result["checkpoint"] != chosen["checkpoint"]:
                raise ValueError(f"Evaluation did not use the recorded final checkpoint: {name}/{seed}")
            if "adapter" in chosen and any(result.get(field) != chosen.get(field)
                                            for field in ("adapter", "adapter_options", "checkpoint_files_sha256")):
                raise ValueError(f"Evaluation adapter identity differs from training: {name}/{seed}")
            status_path = Path(result["status_file"])
            if not status_path.is_absolute():
                status_path = evaluation_root / name / status_path
            status = read(status_path)
            if set(result["metrics"]) != set(BENCHMARKS):
                raise ValueError(f"Full transfer needs all eight benchmarks: {name}/{seed}")
            judge = result["judge"]
            judge_args = result.get("judge_args", {})
            scores = {}
            model_name = UMM_MODEL_NAME if "adapter" in chosen else MODEL_NAME
            for benchmark in BENCHMARKS:
                done = status["datasets"][benchmark]
                if (done["status"] != "done" or done.get("error_message") or done.get("skip_reason")
                        or done["judge_model"] != judge or done["metrics"] != result["metrics"][benchmark]):
                    raise ValueError(f"Evaluation status differs from summary: {name}/{seed}/{benchmark}")
                workbook = scored_workbook(status_path.parent, model_name, benchmark, judge, judge_args)
                scores[benchmark] = {"path": os.path.relpath(workbook.resolve(), output.parent), "sha256": sha256(workbook)}
                if judge_args.get("llm_first", False):
                    if judge_diagnostics(workbook) != result["judging"][benchmark]:
                        raise ValueError(f"Judge diagnostics differ from summary: {name}/{seed}/{benchmark}")
            runs.append({"model": name, "seed": seed, "model_seed": final[seed]["model_seed"],
                         "checkpoint": chosen["checkpoint"], "judge": judge, "scores": scores})
            if judge_args:
                runs[-1].update(judge_args=judge_args, judging=result["judging"])
    for row in audit:
        if sha256((output.parent / row["path"]).resolve()) != row["sha256"]:
            raise ValueError("Training/evaluation receipts changed during collection")
    baseline_rng = {row["seed"]: row["model_seed"] for row in runs if row["model"] == baselines[0]}
    if any(row["model_seed"] != baseline_rng[row["seed"]] for row in runs):
        raise ValueError("Source and baseline model RNGs differ")
    judge_settings = {(row["judge"], json.dumps(row.get("judge_args", {}), sort_keys=True)) for row in runs}
    if len(judge_settings) != 1:
        raise ValueError("Transfer runs must use the same judge and judging policy")
    if sha256(experiment_path) != experiment_hash:
        raise ValueError("Experiment configuration changed during collection")
    return {"schema_version": 1, "taxonomy": os.path.relpath(Path(taxonomy_dir).resolve(), output.parent),
            "models": models, "baseline": baselines[0], "benchmarks": list(BENCHMARKS), "runs": runs,
            "provenance": {"experiment_sha256": experiment_hash, "receipts": audit,
                           "note": "Requested judge labels do not prove API availability; retain scorer logs."}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parents[2] / "configs/experiments/transfer.json")
    parser.add_argument("--training-root", type=Path, required=True, help="Directory containing one subdirectory per experiment job")
    parser.add_argument("--evaluation-root", type=Path, required=True, help="Directory containing <job>/summary.json")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect(args.experiment, args.training_root, args.evaluation_root, args.taxonomy, args.output, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(result["runs"]),
                      "scored_files": sum(len(run["scores"]) for run in result["runs"])}))


if __name__ == "__main__":
    main()
