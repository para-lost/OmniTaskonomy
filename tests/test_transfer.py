import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omnitaskonomy.analysis.transfer import analyze, main, write_report
from omnitaskonomy.data.common import sha256
from omnitaskonomy.taxonomy import load_taxonomy


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        taxonomy = self.root / "taxonomy"
        taxonomy.mkdir()
        catalogue = {"schema_version": 1,
            "families": [{"id": "REC", "name": "Recognition", "definition": "Identify."},
                         {"id": "RCN", "name": "Reconstruction", "definition": "Geometry."}],
            "tasks": [
                {"id": "i2i:source", "name": "Source", "modality": "i2i", "family": "REC",
                 "definition": "Render labels.", "source_id": "source"},
                {"id": "i2t:A", "name": "Identity", "modality": "i2t", "family": "REC", "definition": "Identify."},
                {"id": "i2t:B", "name": "Depth", "modality": "i2t", "family": "RCN", "definition": "Recover depth."}]}
        (taxonomy / "tasks.json").write_text(json.dumps(catalogue))
        retained = [{"uid": f"Bench::test::{index}", "benchmark": "Bench", "index": str(index),
                     "task_id": "i2t:A" if index < 2 else "i2t:B", "family": "REC" if index < 2 else "RCN"}
                    for index in range(3)]
        excluded = [{"uid": "Bench::test::3", "benchmark": "Bench", "index": "3", "reason": "Excluded by refinement"}]
        for name, rows in [("retained_questions.jsonl", retained), ("excluded_questions.jsonl", excluded)]:
            (taxonomy / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (taxonomy / "provenance.json").write_text(json.dumps({"schema_version": 1,
            "files": {name: sha256(taxonomy / name) for name in
                      ["tasks.json", "retained_questions.jsonl", "excluded_questions.jsonl"]},
            "counts": {"i2i": 1, "i2t": 2, "retained": 3, "excluded": 1}}))
        self.config = {"schema_version": 1, "taxonomy": "taxonomy", "baseline": "baseline",
                       "models": {"baseline": {"task_id": None}, "source": {"task_id": "i2i:source"}},
                       "benchmarks": ["Bench"], "runs": []}
        # Different baselines across seeds expose accidental cross-seed pairing.
        scores = {42: {"baseline": [0, 0, 0, 1], "source": [1, 0, 1, 0]},
                  43: {"baseline": [1, 1, 1, 1], "source": [1, 0, 1, 0]},
                  44: {"baseline": [0, 1, 0, 1], "source": [1, 1, 1, 0]},
                  45: {"baseline": [0, 0, 0, 1], "source": [1, 0, 1, 0]}}
        for seed, models in scores.items():
            for model, hits in models.items():
                path = self.root / f"{model}_{seed}.csv"
                self.write_scores(path, hits)
                self.config["runs"].append({"model": model, "seed": seed, "model_seed": seed + 4354,
                    "scores": {"Bench": {"path": path.name, "sha256": sha256(path)}}})
        self.path = self.root / "transfer.json"
        self.save_config()

    def write_scores(self, path, hits):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "hit", "question", "answer", "A", "B"])
            writer.writeheader()
            for index in reversed(range(len(hits))):
                writer.writerow({"index": index, "hit": hits[index], "question": f"Question {index}",
                                 "answer": "A", "A": "yes", "B": "no"})

    def save_config(self):
        self.path.write_text(json.dumps(self.config))

    @staticmethod
    def overall(report, model="source"):
        return next(row for row in report["statistics"]
                    if row["scope"] == "ALL" and row["node_id"] == "__OVERALL__" and row["model"] == model)

    def test_single_seed_uses_retained_questions_and_micro_accuracy(self):
        report = analyze(self.path)
        row = self.overall(report)
        self.assertEqual(report["n_questions"], 3)
        self.assertAlmostEqual(row["accuracy_mean_pct"], 200 / 3)
        self.assertAlmostEqual(row["gain_mean_pp"], 200 / 3)
        self.assertIsNone(row["accuracy_sample_std_pp"])
        self.assertIsNone(row["gain_sem_pp"])
        self.assertEqual(len(report["transfer_matrix"]), 2)
        self.assertEqual(report["input_audit"]["n_score_files"], 2)
        self.assertEqual(report["input_audit"]["taxonomy_provenance_sha256"],
                         sha256(self.root / "taxonomy" / "provenance.json"))
        self.assertEqual({r["n_rows"] for r in report["input_audit"]["inputs"]}, {4})
        self.assertNotEqual(row["accuracy_mean_pct"], 75)  # Unweighted leaf mean is incorrect.

    def test_three_seed_gain_and_accuracy_dispersion_are_distinct(self):
        report = analyze(self.path, [42, 43, 44])
        row = self.overall(report)
        self.assertAlmostEqual(row["accuracy_mean_pct"], 700 / 9)
        self.assertAlmostEqual(row["gain_mean_pp"], 100 / 3)
        self.assertNotEqual(row["accuracy_sample_std_pp"], row["gain_sample_std_pp"])
        self.assertAlmostEqual(row["gain_sem_pp"], row["gain_sample_std_pp"] / 3**0.5)
        per_seed = {r["seed"]: r["gain_pp"] for r in report["per_seed"]
                    if r["scope"] == "ALL" and r["node_id"] == "__OVERALL__" and r["model"] == "source"}
        self.assertEqual(per_seed, {42: 200 / 3, 43: -100 / 3, 44: 200 / 3})
        test = next(r for r in report["paired_tests"] if r["scope"] == "ALL" and r["node_id"] == "__OVERALL__")
        self.assertEqual((test["n_questions"], test["n_seeds"]), (3, 3))
        self.assertEqual(test["p_value"], 0.5)  # Weights [2, -1, 2], four of eight sign assignments.

    def test_four_seeds_need_explicit_summary_only(self):
        with self.assertRaisesRegex(ValueError, "1–3 seeds"):
            analyze(self.path, [42, 43, 44, 45])
        report = analyze(self.path, [42, 43, 44, 45], paired=False)
        self.assertEqual(report["paired_tests"], [])
        self.assertTrue(all(row["p_value"] is None for row in report["transfer_matrix"]))
        self.assertEqual(self.overall(report)["n_seeds"], 4)

    def test_missing_or_duplicate_runs_and_model_seed_mismatch_fail(self):
        original = copy.deepcopy(self.config)
        self.config["runs"] = [r for r in self.config["runs"] if (r["model"], r["seed"]) != ("baseline", 43)]
        self.save_config()
        with self.assertRaisesRegex(ValueError, "same-seed baseline"):
            analyze(self.path, [42, 43])
        self.config = copy.deepcopy(original)
        self.config["runs"].append(copy.deepcopy(self.config["runs"][0]))
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Duplicate run"):
            analyze(self.path)
        self.config = copy.deepcopy(original)
        self.config["runs"][1]["model_seed"] += 1
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Model RNG differs"):
            analyze(self.path)

    def test_score_hash_and_full_question_coverage_are_checked(self):
        source = self.root / "source_42.csv"
        self.write_scores(source, [1, 0, 1])
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            analyze(self.path)
        self.config["runs"][1]["scores"]["Bench"]["sha256"] = sha256(source)
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Question coverage differs"):
            analyze(self.path)

    def test_comparisons_require_the_same_judge_and_policy(self):
        expected = self.overall(analyze(self.path, [42, 43]))
        for run in self.config["runs"]:
            run.update(judge="chatgpt-0125", judge_args={"llm_first": True})
        self.save_config()
        self.assertEqual(self.overall(analyze(self.path, [42, 43])), expected)
        original = copy.deepcopy(self.config)
        for field, replacement in [("judge", "exact_matching"), ("judge_args", {}), ("judge", None)]:
            with self.subTest(field=field, replacement=replacement):
                self.config = copy.deepcopy(original)
                self.config["runs"][1][field] = replacement
                self.save_config()
                with self.assertRaisesRegex(ValueError, "Judge model/policy differs"):
                    analyze(self.path, [42, 43])
        self.config = copy.deepcopy(original)
        for run in self.config["runs"]:
            if run["seed"] == 43:
                run["judge_args"] = {}
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Judge model/policy differs"):
            analyze(self.path, [42, 43])
        analyze(self.path, [42])

    def test_changed_question_is_not_silently_joined_by_index(self):
        source = self.root / "source_42.csv"
        source.write_text(source.read_text().replace("Question 0", "Different question"))
        self.config["runs"][1]["scores"]["Bench"]["sha256"] = sha256(source)
        self.save_config()
        with self.assertRaisesRegex(ValueError, "question drift"):
            analyze(self.path)

    def test_provenance_drift_during_analysis_fails_the_audit(self):
        def changed_taxonomy(directory):
            taxonomy = load_taxonomy(directory)
            receipt = directory / "provenance.json"
            receipt.write_text(receipt.read_text() + "\n")
            return taxonomy

        with mock.patch("omnitaskonomy.analysis.transfer.load_taxonomy", side_effect=changed_taxonomy):
            with self.assertRaisesRegex(ValueError, "Taxonomy provenance changed"):
                analyze(self.path)

    def test_output_files_are_consistent_and_cannot_overwrite_an_old_report(self):
        output = self.root / "report"
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["--config", str(self.path), "--output-dir", str(output), "--seeds", "42", "43", "44"])
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(json.loads(capture.getvalue())["transfer_cells"], 2)
        self.assertTrue((output / "paired_tests.csv").is_file())
        self.assertEqual(json.loads((output / "input_audit.json").read_text()), report["input_audit"])
        with (output / "transfer_matrix.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        before = (output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            write_report(report, output)
        self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_invalid_seed_request_creates_no_output(self):
        output = self.root / "invalid"
        for seeds in [[], [42, 42], [-1], [True]]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                analyze(self.path, seeds)
        with self.assertRaisesRegex(ValueError, "Missing source"):
            main(["--config", str(self.path), "--output-dir", str(output), "--seeds", "123"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
