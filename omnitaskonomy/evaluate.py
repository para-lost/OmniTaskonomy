"""Plan VLMEvalKit runs and aggregate their benchmark metrics across seeds."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from omnitaskonomy.umm import read_options, verify_adapter_provenance
from omnitaskonomy.data.common import sha256


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "OmniTaskonomy-BAGEL"
UMM_MODEL_NAME = "OmniTaskonomy-UMM"
BENCHMARK_CONFIGS = json.loads((ROOT / "configs/eval/benchmarks.json").read_text())
BENCHMARKS = tuple(BENCHMARK_CONFIGS)


def scored_workbook(directory, model_name, benchmark, judge, judge_args=None):
    suffix = judge if BENCHMARK_CONFIGS[benchmark]["class"] == "CVBench" else {
        "chatgpt-0125": "openai", "gpt-4-0125": "gpt4",
    }.get(judge, judge)
    if (judge_args or {}).get("llm_first", False):
        suffix += "_llm_first"
    return Path(directory) / f"{model_name}_{benchmark}_{suffix}_result.xlsx"


def judge_diagnostics(workbook):
    from omnitaskonomy.analysis.vlmeval_scores import read_scores

    scores = read_scores(workbook)
    logs = [row.get("log", "") for row in scores["rows"].values()]
    return {"scored_file": scores["path"], "sha256": scores["sha256"], "samples": len(logs),
            "em_fallback_count": sum("Exact matching fallback:" in log for log in logs),
            "unresolved_count": sum("Unresolved LLM judge response." in log for log in logs)}


def select_checkpoints(args):
    if args.run_file is None:
        if args.model_path is None:
            raise ValueError("--model-path is required without --run-file")
        records = [{"seed": seed, "model_path": args.model_path, "checkpoint": args.checkpoint}
                   for seed in args.seeds]
        base = Path.cwd()
    else:
        if args.model_path is not None or args.checkpoint is not None:
            raise ValueError("--run-file already supplies model and checkpoint paths")
        source = json.loads(args.run_file.read_text())
        if source["schema_version"] != 1:
            raise ValueError("Unsupported checkpoints.json schema_version")
        records = [row for row in source["records"] if row["seed"] in args.seeds]
        if len({(row["task"], row["recipe"]) for row in records}) > 1:
            raise ValueError("Use a run manifest for one task/recipe at a time")
        # Training writes stages in execution order, so the last record is the final stage.
        final = {row["seed"]: row for row in records}
        missing = set(args.seeds) - final.keys()
        if missing:
            raise ValueError(f"Run manifest has no checkpoint for seeds {sorted(missing)}")
        records = [final[seed] for seed in args.seeds]
        base = args.run_file.resolve().parent
    options = read_options(args.adapter_options) if args.adapter_options is not None else None
    selected = []
    for row in records:
        adapter = args.adapter or row.get("adapter")
        model_path = (base / row["model_path"]).resolve()
        checkpoint = (base / row["checkpoint"]).resolve() if row["checkpoint"] else None
        if not adapter and checkpoint is None:
            checkpoint = model_path / "ema.safetensors"
        if not adapter and checkpoint.suffix != ".safetensors":
            checkpoint /= "model.safetensors"
        record = {"seed": row["seed"], "model_path": str(model_path),
                  "checkpoint": str(checkpoint) if checkpoint is not None else None}
        if adapter:
            adapter_options = options if options is not None else row.get("adapter_options", {})
            if not isinstance(adapter_options, dict):
                raise ValueError("Adapter options must be a JSON object")
            record.update(adapter=adapter, adapter_options=adapter_options,
                          device=args.device or row.get("device", "cuda:0"))
            if "checkpoint_files_sha256" in row:
                record["checkpoint_files_sha256"] = row["checkpoint_files_sha256"]
        elif options is not None or args.device is not None:
            raise ValueError("--adapter-options and --device require an adapter")
        selected.append(record)
    return selected


def build_plan(args):
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        raise ValueError("Benchmarks must be distinct")
    vendor = ROOT / "VLMEvalKit"
    provenance = json.loads((vendor / "VENDORED.json").read_text())
    judge_args = {"llm_first": True} if args.judge != "exact_matching" else {}
    plans = []
    for record in select_checkpoints(args):
        directory = args.output_dir.resolve() / f"seed_{record['seed']}"
        if "adapter" in record:
            if args.nproc_per_node > 1 and record["device"] not in {"cuda", "cuda:0"}:
                raise ValueError("Distributed adapter evaluation requires --device cuda:0; each worker sees its assigned GPUs")
            if args.parameter_dtype != "bfloat16":
                raise ValueError("--parameter-dtype is BAGEL-specific; set custom precision through --adapter-options")
            model_name = UMM_MODEL_NAME
            record["adapter_provenance_file"] = str(directory / "adapter_provenance.json")
            model = {"class": "OmniTaskonomyUMM", **record,
                     "provenance_path": record["adapter_provenance_file"]}
            model.pop("checkpoint_files_sha256", None)
            model.pop("adapter_provenance_file")
        else:
            model_name = MODEL_NAME
            model = {"class": "OmniTaskonomyBAGEL", **record,
                     "parameter_dtype": args.parameter_dtype}
        config = {
            # Explicit class takes the upstream path that restores WORLD_SIZE after loading.
            "model": {model_name: model},
            "data": {name: BENCHMARK_CONFIGS[name] for name in args.benchmarks},
        }
        command = [sys.executable]
        if args.nproc_per_node > 1:
            command += ["-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node", str(args.nproc_per_node)]
        command += [str(vendor / "run.py"), "--config", str(directory / "vlmeval.json"),
                    "--work-dir", str(directory / "results"), "--mode", args.mode,
                    "--judge", args.judge, "--num-seeds", "1", "--seed-start", str(record["seed"])]
        if judge_args:
            command += ["--judge-args", json.dumps(judge_args)]
        if args.reuse or args.mode == "eval":
            command.append("--reuse")
        plans.append({**record, "directory": str(directory), "config": config, "command": command,
                      "judge": args.judge, "judge_args": judge_args,
                      "vlmeval_revision": provenance["revision"],
                      "vlmeval_source_sha256": provenance["source_manifest_sha256"]})
    return plans


def verify_checkpoint_files(plan):
    for name, expected in plan.get("checkpoint_files_sha256", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Checkpoint receipt requires relative file names: {name}")
        path = Path(plan["checkpoint"]) / relative
        if sha256(path) != expected:
            raise ValueError(f"Checkpoint files differ from the training receipt: {path}")


def read_status(plan, benchmarks, mode="all", changed_after_ns=0):
    configuration = Path(plan["directory"]) / "vlmeval.json"
    if json.loads(configuration.read_text()) != plan["config"]:
        raise ValueError(f"Saved evaluation config differs from the requested run: {configuration}")
    saved = json.loads((Path(plan["directory"]) / "evaluation.json").read_text())
    if saved.get("vlmeval_source_sha256") != plan["vlmeval_source_sha256"]:
        raise ValueError(f"Saved VLMEvalKit source differs from the requested run: {configuration.parent}")
    if saved.get("judge_args", {}) != plan.get("judge_args", {}):
        raise ValueError(f"Saved judge policy differs from the requested run: {configuration.parent}")
    if "adapter" in plan:
        if saved.get("checkpoint_files_sha256") != plan.get("checkpoint_files_sha256"):
            raise ValueError(f"Saved evaluation checkpoint hashes differ: {configuration.parent}")
        verify_checkpoint_files(plan)
        identity = verify_adapter_provenance(
            plan["adapter_provenance_file"], plan["adapter"], plan["adapter_options"],
        )
    model_name, = plan["config"]["model"]
    result_root = Path(plan["directory"]) / "results" / model_name
    candidates = list(result_root.glob("*/status.json"))
    if not candidates:
        raise RuntimeError(f"No VLMEvalKit status.json under {result_root}")
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    if path.stat().st_mtime_ns < changed_after_ns:
        raise RuntimeError(f"Evaluation produced no new status file: {path}")
    status = json.loads(path.read_text())
    metrics, judging = {}, {}
    for benchmark in benchmarks:
        row = status["datasets"].get(benchmark)
        if row is None or row.get("error_message"):
            raise RuntimeError(f"Missing or failed {benchmark} evaluation: {row}; see {path}")
        if mode == "infer":
            if row.get("skip_reason") != "mode_infer":
                raise RuntimeError(f"Inference is incomplete for {benchmark}; see {path}")
            continue
        if row.get("status") != "done" or row.get("skip_reason") or not row.get("metrics"):
            raise RuntimeError(f"Evaluation is incomplete for {benchmark}; see {path}")
        if row.get("judge_model") != plan["judge"]:
            raise ValueError(f"Judge differs for {benchmark}: requested {plan['judge']}, "
                             f"found {row.get('judge_model')}; see {path}")
        metrics[benchmark] = row["metrics"]
        if plan.get("judge_args", {}).get("llm_first", False):
            workbook = scored_workbook(path.parent, model_name, benchmark, plan["judge"], plan["judge_args"])
            judging[benchmark] = judge_diagnostics(workbook)
    result = {"seed": plan["seed"], "checkpoint": plan["checkpoint"],
              "judge": plan["judge"], "status_file": str(path), "metrics": metrics}
    if plan.get("judge_args"):
        result.update(judge_args=plan["judge_args"], judging=judging)
    if "adapter" in plan:
        result.update(adapter=plan["adapter"], adapter_options=plan["adapter_options"],
                      adapter_provenance=identity)
        if "checkpoint_files_sha256" in plan:
            result["checkpoint_files_sha256"] = plan["checkpoint_files_sha256"]
    return result


def summarize_values(values):
    """Summarize a nonempty sequence without changing its metric units."""
    sample_std = statistics.stdev(values) if len(values) > 1 else None
    return {"n": len(values), "mean": statistics.mean(values), "sample_std": sample_std,
            "sem": sample_std / math.sqrt(len(values)) if sample_std is not None else None}


def aggregate_results(results):
    """Keep upstream metric units and benchmark definitions; never pool benchmarks."""
    if not results or len({row["seed"] for row in results}) != len(results):
        raise ValueError("Expected one result for each distinct seed")
    reference = results[0]["metrics"]
    if any(set(row["metrics"]) != set(reference) for row in results):
        raise ValueError("Benchmark coverage differs between seeds")
    aggregate = {}
    for benchmark, metrics in reference.items():
        if any(set(row["metrics"][benchmark]) != set(metrics) for row in results):
            raise ValueError(f"Metric coverage differs between seeds for {benchmark}")
        aggregate[benchmark] = {}
        for metric in metrics:
            values = [row["metrics"][benchmark][metric] for row in results]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values):
                raise ValueError(f"Non-numeric or non-finite metric {benchmark}/{metric}: {values}")
            aggregate[benchmark][metric] = {
                "seeds": [row["seed"] for row in results], "values": values,
                **summarize_values(values),
            }
    return {"schema_version": 1, "runs": results, "benchmarks": aggregate}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", type=Path, help="Base model/config directory")
    result.add_argument("--checkpoint", type=Path, help="Weights file or checkpoint directory; default: base model")
    result.add_argument("--adapter", help="Custom UMM factory as package.module:create_adapter")
    result.add_argument("--adapter-options", type=Path, help="JSON file with adapter-specific options")
    result.add_argument("--device", help="Custom adapter device; default: saved training device or cuda:0")
    result.add_argument("--parameter-dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Parameter precision; use float32 for historical non-offload BAGEL evaluations")
    result.add_argument("--run-file", type=Path, help="Training checkpoints.json (schema_version=1)")
    result.add_argument("--seeds", type=int, nargs="+", default=[42])
    result.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--nproc-per-node", type=int, default=1,
                        help="One model replica per GPU; does not shard the model")
    result.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    result.add_argument("--judge", default="chatgpt-0125",
                        choices=("exact_matching", "chatgpt-0125", "gpt-4-0125"),
                        help="LLM answer judge (default: chatgpt-0125); API failures fall back to exact matching")
    result.add_argument("--reuse", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="Print configuration and commands; write nothing")
    result.add_argument("--aggregate-only", action="store_true", help="Read completed runs without inference")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        plans = build_plan(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return
    if args.aggregate_only and args.mode == "infer":
        arguments.error("--aggregate-only requires scored results (--mode all or eval)")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "VLMEvalKit"), environment.get("PYTHONPATH", "")]
    )
    # The upstream runner accepts this global override; it would merge different seeds.
    environment.pop("MMEVAL_ROOT", None)
    results = []
    for plan in plans:
        started = 0
        if not args.aggregate_only:
            if "adapter" in plan:
                for value in (plan["model_path"], plan["checkpoint"]):
                    if value is not None and not Path(value).exists():
                        raise FileNotFoundError(value)
                verify_checkpoint_files(plan)
            else:
                for path in [Path(plan["checkpoint"]), Path(plan["model_path"]) / "llm_config.json",
                             Path(plan["model_path"]) / "vit_config.json"]:
                    if not path.is_file():
                        raise FileNotFoundError(path)
            directory = Path(plan["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "vlmeval.json"
            if config_path.exists():
                if json.loads(config_path.read_text()) != plan["config"]:
                    raise ValueError(f"Changed checkpoint/config requires a new output directory: {config_path}")
            receipt = directory / "evaluation.json"
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous.get("vlmeval_source_sha256") != plan["vlmeval_source_sha256"]:
                    raise ValueError(f"Changed VLMEvalKit source requires a new output directory: {directory}")
                if (previous["judge"] != plan["judge"]
                        or previous.get("judge_args", {}) != plan["judge_args"]):
                    raise ValueError(f"Changed judge/policy requires a new output directory: {directory}")
                if "adapter" in plan and previous.get("checkpoint_files_sha256") != plan.get("checkpoint_files_sha256"):
                    raise ValueError(f"Changed checkpoint files require a new output directory: {directory}")
            if "adapter" in plan:
                identity_path = Path(plan["adapter_provenance_file"])
                cached_results = any(path.is_file() for path in (directory / "results").rglob("*"))
                if identity_path.is_file() or cached_results:
                    verify_adapter_provenance(identity_path, plan["adapter"], plan["adapter_options"])
            config_path.write_text(json.dumps(plan["config"], indent=2) + "\n")
            receipt.write_text(json.dumps(plan, indent=2) + "\n")
            print(shlex.join(plan["command"]), flush=True)
            started = time.time_ns()
            # Every rank inherits the same unique run directory, including same-day reruns.
            environment["OMNITASKONOMY_EVAL_ID"] = f"T{started}"
            subprocess.run(plan["command"], cwd=ROOT / "VLMEvalKit", env=environment, check=True)
        results.append(read_status(plan, args.benchmarks, args.mode, started))
    output = args.output_dir.resolve() / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = aggregate_results(results) if args.mode != "infer" else {"schema_version": 1, "runs": results}
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
