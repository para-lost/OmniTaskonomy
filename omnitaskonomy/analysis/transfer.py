"""Aggregate frozen question scores and compare seed-matched checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from omnitaskonomy.analysis.paired_permutation import paired_test
from omnitaskonomy.analysis.vlmeval_scores import compare_questions, read_scores
from omnitaskonomy.data.common import sha256
from omnitaskonomy.evaluate import summarize_values
from omnitaskonomy.taxonomy import load_taxonomy


def analyze(config_path, seeds=(42,), *, paired=True):
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if paired and len(seeds) > 3:
        raise ValueError("Exact paired tests support 1–3 seeds; use --summary-only for more seeds")
    config_path = Path(config_path).resolve()
    config_hash = sha256(config_path)
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported transfer configuration schema_version")
    taxonomy_dir = (config_path.parent / config["taxonomy"]).resolve()
    provenance_hash = sha256(taxonomy_dir / "provenance.json")
    taxonomy = load_taxonomy(taxonomy_dir)
    tasks = {task["id"]: task for task in taxonomy["tasks"]}
    models = config["models"]
    baseline = config["baseline"]
    if baseline not in models or len(models) < 2 or models[baseline]["task_id"] is not None:
        raise ValueError("Configure a baseline with task_id null and at least one source model")
    for model, spec in models.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model IDs must be nonempty strings")
        if model != baseline and (spec["task_id"] not in tasks or tasks[spec["task_id"]]["modality"] != "i2i"):
            raise ValueError(f"Source {model} must reference a released I2I task")
    benchmarks = config["benchmarks"]
    available = {row["benchmark"] for row in taxonomy["retained"]}
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or not set(benchmarks) <= available:
        raise ValueError("Benchmarks must be distinct names present in the retained taxonomy")

    runs = {}
    for run in config["runs"]:
        if type(run["seed"]) is not int or run["seed"] < 0 or run["model"] not in models:
            raise ValueError("Each run needs a configured model and a nonnegative integer seed")
        key = (run["model"], run["seed"])
        if key in runs:
            raise ValueError(f"Duplicate run: {key}")
        if set(run["scores"]) != set(benchmarks):
            raise ValueError(f"Benchmark coverage differs for {key}")
        if "model_seed" in run and (type(run["model_seed"]) is not int or run["model_seed"] < 0):
            raise ValueError(f"Invalid model RNG seed for {key}")
        runs[key] = run
    expected_runs = {(model, seed) for model in models for seed in seeds}
    missing = expected_runs - runs.keys()
    if missing:
        raise ValueError(f"Missing source or same-seed baseline runs: {sorted(missing)}")
    for model, seed in expected_runs:
        if runs[model, seed].get("model_seed") != runs[baseline, seed].get("model_seed"):
            raise ValueError(f"Model RNG differs from the same-seed baseline: {model}/{seed}")
    reference = runs[baseline, seeds[0]]
    judge_policy = (reference.get("judge"), reference.get("judge_args", {}))
    if any((runs[key].get("judge"), runs[key].get("judge_args", {})) != judge_policy for key in expected_runs):
        raise ValueError("Judge model/policy differs between comparison runs")

    questions = sorted((row for row in taxonomy["retained"] if row["benchmark"] in benchmarks),
                       key=lambda row: row["uid"])
    positions = {bench: [(i, row["index"]) for i, row in enumerate(questions) if row["benchmark"] == bench]
                 for bench in benchmarks}
    expected_indexes = {bench: {row["index"] for row in taxonomy["retained"] + taxonomy["excluded"]
                                if row["benchmark"] == bench} for bench in benchmarks}
    hits = {key: np.empty(len(questions), dtype=np.int8) for key in expected_runs}
    canonical = {}
    inventory = []
    model_order = [baseline, *[model for model in models if model != baseline]]
    for model in model_order:
        for seed in seeds:
            run = runs[model, seed]
            for benchmark in benchmarks:
                spec = run["scores"][benchmark]
                path = (config_path.parent / spec["path"]).resolve()
                scored = read_scores(path, expected_sha256=spec.get("sha256"))
                rows = scored["rows"]
                if set(rows) != expected_indexes[benchmark]:
                    missing_ids = sorted(expected_indexes[benchmark] - rows.keys())[:5]
                    extra_ids = sorted(rows.keys() - expected_indexes[benchmark])[:5]
                    raise ValueError(f"Question coverage differs for {model}/{seed}/{benchmark}: "
                                     f"missing={missing_ids}, extra={extra_ids}")
                if benchmark in canonical:
                    compare_questions(canonical[benchmark], rows)
                else:
                    canonical[benchmark] = rows
                for position, index in positions[benchmark]:
                    hits[model, seed][position] = rows[index]["hit"]
                inventory.append({"model": model, "seed": seed, "model_seed": run.get("model_seed"),
                                  "benchmark": benchmark, "path": scored["path"], "sha256": scored["sha256"],
                                  "n_rows": len(rows), "n_retained": len(positions[benchmark]),
                                  "checkpoint": run.get("checkpoint"), "judge": run.get("judge")})

    nodes = [{"id": "__OVERALL__", "name": "Overall (micro)", "type": "overall"}]
    nodes += [{"id": family["id"], "name": family["name"], "type": "family"}
              for family in taxonomy["families"]]
    nodes += [{"id": task["id"], "name": task["name"], "type": "capability"}
              for task in taxonomy["tasks"] if task["modality"] == "i2t"]
    per_seed, summaries, tests, matrix = [], [], [], []
    for scope in ["ALL", *benchmarks]:
        for node in nodes:
            selected = [i for i, row in enumerate(questions)
                        if (scope == "ALL" or row["benchmark"] == scope)
                        and (node["type"] == "overall"
                             or node["type"] == "family" and row["family"] == node["id"]
                             or node["type"] == "capability" and row["task_id"] == node["id"])]
            count = len(selected)
            identity = {"scope": scope, "node_id": node["id"], "node_type": node["type"],
                        "node_name": node["name"], "n_questions": count, "n_seeds": len(seeds)}
            baseline_hits = np.asarray([hits[baseline, seed][selected] for seed in seeds])
            for model in model_order:
                model_hits = np.asarray([hits[model, seed][selected] for seed in seeds])
                accuracies, gains = [], []
                for index, seed in enumerate(seeds):
                    correct = int(model_hits[index].sum())
                    base_correct = int(baseline_hits[index].sum())
                    accuracy = 100 * correct / count if count else None
                    gain = 100 * (correct - base_correct) / count if count else None
                    per_seed.append({**identity, "model": model, "seed": seed,
                                     "correct": correct, "baseline_correct": base_correct,
                                     "accuracy_pct": accuracy, "gain_pp": gain})
                    if count:
                        accuracies.append(accuracy)
                        gains.append(gain)
                empty = {"mean": None, "sample_std": None, "sem": None}
                acc = summarize_values(accuracies) if count else empty
                gain = summarize_values(gains) if count else empty
                summary = {**identity, "model": model, "source_task_id": models[model]["task_id"],
                           "accuracy_mean_pct": acc["mean"], "accuracy_sample_std_pp": acc["sample_std"],
                           "accuracy_sem_pp": acc["sem"], "gain_mean_pp": gain["mean"],
                           "gain_sample_std_pp": gain["sample_std"], "gain_sem_pp": gain["sem"]}
                summaries.append(summary)
                test = None
                if paired and count and model != baseline:
                    test = {**identity, "model": model, **paired_test(model_hits, baseline_hits)}
                    tests.append(test)
                if scope == "ALL" and node["type"] == "capability" and model != baseline:
                    matrix.append({**summary, "p_value": test["p_value"] if test else None,
                                   "significant": test["significant"] if test else None,
                                   "direction": test["direction"] if test else None})

    # Confirm the files still identify the scores used throughout this calculation.
    for item in inventory:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Scored input changed during analysis: {item['path']}")
    if sha256(config_path) != config_hash:
        raise ValueError("Transfer configuration changed during analysis")
    if sha256(taxonomy_dir / "provenance.json") != provenance_hash:
        raise ValueError("Taxonomy provenance changed during analysis")
    for filename, expected in taxonomy["provenance"]["files"].items():
        if sha256(taxonomy_dir / filename) != expected:
            raise ValueError(f"Taxonomy input changed during analysis: {filename}")
    audit = {"passed": True, "config_path": str(config_path), "config_sha256": config_hash,
             "taxonomy_path": str(taxonomy_dir), "taxonomy_files": taxonomy["provenance"]["files"],
             "taxonomy_provenance_sha256": provenance_hash,
             "n_questions": len(questions), "n_score_files": len(inventory), "inputs": inventory}
    return {"schema_version": 1, "baseline": baseline, "models": models, "seeds": seeds,
            "benchmarks": benchmarks, "n_questions": len(questions), "exact_tests": paired,
            "significance": {"alpha": 0.05, "two_sided": True, "multiple_comparison_correction": None,
                             "pairing_unit": "question, with all selected seeds swapped together"},
            "per_seed": per_seed, "statistics": summaries, "paired_tests": tests,
            "transfer_matrix": matrix, "input_audit": audit}


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables = {"per_seed.csv": report["per_seed"], "summary.csv": report["statistics"],
              "transfer_matrix.csv": report["transfer_matrix"]}
    if report["paired_tests"]:
        tables["paired_tests.csv"] = report["paired_tests"]
    for filename, rows in tables.items():
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "input_audit.json").write_text(json.dumps(report["input_audit"], indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment and scored-file manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for CSV/JSON reports")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--summary-only", action="store_true", help="Aggregate any number of seeds without paired tests")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    report = analyze(args.config, args.seeds, paired=not args.summary_only)
    write_report(report, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "n_questions": report["n_questions"],
                      "n_seeds": len(args.seeds), "transfer_cells": len(report["transfer_matrix"])}))


if __name__ == "__main__":
    main()
