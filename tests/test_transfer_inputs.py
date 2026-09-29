import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from omnitaskonomy.analysis.inputs import collect
from omnitaskonomy.data.common import sha256
from omnitaskonomy.evaluate import BENCHMARKS, MODEL_NAME, UMM_MODEL_NAME, judge_diagnostics, scored_workbook
from omnitaskonomy.taxonomy import DEFAULT_DIRECTORY


class TransferInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training, self.evaluation = self.root / "training", self.root / "evaluation"
        self.output = self.root / "collected/inputs.json"
        self.spec = dict(schema_version=1, id="transfer", paper_commit="paper-test", training={
            "suite": "transfer", "seeds": [42, 43], "model_seeds": [4396, 4397],
            "i2t_manifest": "llava/train.jsonl"}, jobs=[
                {"id": "baseline", "task_id": None, "training": {"task": "baseline", "recipe": "r1"}},
                {"id": "normal", "task_id": "i2i:surface_normals", "training": {
                    "task": "normal", "recipe": "r2", "i2i_manifest": "normal/train.jsonl"}}])
        self.config = self.json("experiment-config.json", self.spec)
        self.receipt = self.json("training/experiment.json", dict(experiment=self.spec,
            config_sha256=sha256(self.config), data_root=str(self.root / "data"),
            model_path=str(self.root / "base"), nproc_override=None))
        for job in ("baseline", "normal"):
            records, evaluated = [], []
            for seed in (42, 43):
                checkpoint = self.training / job / f"seed_{seed}/checkpoint"
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.safetensors").write_bytes(b"checkpoint path fixture")
                records.append(dict(task=job, recipe="i2t-only" if job == "baseline" else "i2i-to-i2t",
                    seed=seed, stage="i2t", model_seed=4396 + seed - 42, checkpoint=str(checkpoint),
                    model_path=str(self.root / "base"), initialization=str(self.root / "base")))
                datasets = {}
                directory = self.evaluation / job / f"seed_{seed}/results/{MODEL_NAME}/T1"
                directory.mkdir(parents=True)
                for benchmark in BENCHMARKS:
                    datasets[benchmark] = dict(status="done", error_message=None, skip_reason=None,
                        judge_model="exact_matching", metrics={"Overall": 1.0})
                    pd.DataFrame([dict(index=0, question="Fixture question?", answer="A", A="yes", B="no", hit=1)]).to_excel(
                        directory / f"{MODEL_NAME}_{benchmark}_exact_matching_result.xlsx", index=False)
                status = directory / "status.json"
                self.json(status, {"datasets": datasets})
                evaluated.append(dict(seed=seed, checkpoint=str(checkpoint / "model.safetensors"),
                    judge="exact_matching", status_file=str(status),
                    metrics={benchmark: {"Overall": 1.0} for benchmark in BENCHMARKS}))
            self.json(self.training / job / "checkpoints.json", {"schema_version": 1, "records": records})
            self.json(self.evaluation / job / "summary.json", {"schema_version": 1, "runs": evaluated})

    def json(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def collect(self):
        return collect(self.config, self.training, self.evaluation, DEFAULT_DIRECTORY, self.output, [42, 43])

    def test_full_eight_benchmark_chain_preserves_job_rng_paths_and_hashes(self):
        result = self.collect()
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(result["models"]["normal"]["task_id"], "i2i:surface_normals")
        for run in result["runs"]:
            self.assertEqual(run["model_seed"], 4396 + run["seed"] - 42)
            self.assertEqual(set(run["scores"]), set(BENCHMARKS))
            for file in run["scores"].values():
                self.assertEqual(file["sha256"], sha256((self.output.parent / file["path"]).resolve()))
        self.assertEqual(result["provenance"]["experiment_sha256"], sha256(self.config))
        self.assertTrue(any(Path(row["path"]).name == "experiment.json" for row in result["provenance"]["receipts"]))

    def test_custom_model_receipts_keep_checkpoint_directories_and_benchmark_paths(self):
        for job in ("baseline", "normal"):
            manifest = self.training / job / "checkpoints.json"
            trained = json.loads(manifest.read_text())
            summary = self.evaluation / job / "summary.json"
            evaluated = json.loads(summary.read_text())
            for record, run in zip(trained["records"], evaluated["runs"]):
                record.update(adapter="example:create_adapter", adapter_options={}, device="cpu")
                checkpoint = Path(record["checkpoint"])
                (checkpoint / "model.safetensors").rename(checkpoint / "weights.pt")
                run["checkpoint"] = str(checkpoint)
                run.update(adapter=record["adapter"], adapter_options=record["adapter_options"])
                previous = Path(run["status_file"]).parent
                current = previous.parent.with_name(UMM_MODEL_NAME) / previous.name
                previous.parent.rename(current.parent)
                for path in current.glob("*.xlsx"):
                    path.rename(path.with_name(path.name.replace(MODEL_NAME, UMM_MODEL_NAME)))
                run["status_file"] = str(current / "status.json")
            self.json(manifest, trained)
            self.json(summary, evaluated)
        result = self.collect()
        self.assertEqual(len(result["runs"]), 4)
        for run in result["runs"]:
            self.assertTrue(Path(run["checkpoint"]).is_dir())
            for score in run["scores"].values():
                self.assertIn(UMM_MODEL_NAME, Path(score["path"]).name)
                self.assertEqual(score["sha256"], sha256((self.output.parent / score["path"]).resolve()))
        summary = self.evaluation / "normal/summary.json"
        evaluated = json.loads(summary.read_text())
        evaluated["runs"][0]["adapter"] = "another:create_adapter"
        self.json(summary, evaluated)
        with self.assertRaisesRegex(ValueError, "adapter identity differs"):
            self.collect()

    def test_llm_workbooks_use_cvbench_suffix_and_keep_fallback_counts(self):
        for judge_args in ({}, {"llm_first": True}):
            for job in ("baseline", "normal"):
                summary = self.evaluation / job / "summary.json"
                evaluated = json.loads(summary.read_text())
                for run in evaluated["runs"]:
                    status_file = Path(run["status_file"])
                    status = json.loads(status_file.read_text())
                    judging = {}
                    for benchmark in BENCHMARKS:
                        old = scored_workbook(status_file.parent, MODEL_NAME, benchmark,
                                              run["judge"], run.get("judge_args"))
                        workbook = scored_workbook(status_file.parent, MODEL_NAME, benchmark,
                                                   "chatgpt-0125", judge_args)
                        frame = pd.read_excel(old)
                        old.unlink()
                        frame["log"] = ("Exact matching fallback: API unavailable."
                                        if benchmark == "MMVP" else "Match Log: A.")
                        frame.to_excel(workbook, index=False)
                        judging[benchmark] = judge_diagnostics(workbook)
                        status["datasets"][benchmark]["judge_model"] = "chatgpt-0125"
                    run.update(judge="chatgpt-0125", judge_args=judge_args, judging=judging)
                    self.json(status_file, status)
                self.json(summary, evaluated)
            result = self.collect()
            for run in result["runs"]:
                suffix = "_llm_first" if judge_args else ""
                self.assertEqual(Path(run["scores"]["CV-Bench-3D"]["path"]).name,
                                 f"{MODEL_NAME}_CV-Bench-3D_chatgpt-0125{suffix}_result.xlsx")
                self.assertEqual(Path(run["scores"]["BLINK"]["path"]).name,
                                 f"{MODEL_NAME}_BLINK_openai{suffix}_result.xlsx")
                if judge_args:
                    self.assertEqual(run["judging"]["MMVP"]["em_fallback_count"], 1)
                    self.assertEqual(run["judging"]["BLINK"]["em_fallback_count"], 0)
            if judge_args:
                workbook = (self.output.parent / result["runs"][0]["scores"]["MMVP"]["path"]).resolve()
                frame = pd.read_excel(workbook)
                frame["log"] = "Match Log: A."
                frame.to_excel(workbook, index=False)
                with self.assertRaisesRegex(ValueError, "Judge diagnostics differ"):
                    self.collect()

    def test_different_source_and_baseline_judges_cannot_be_compared(self):
        summary = self.evaluation / "normal/summary.json"
        evaluated = json.loads(summary.read_text())
        run = evaluated["runs"][0]
        status_file = Path(run["status_file"])
        status = json.loads(status_file.read_text())
        for benchmark in BENCHMARKS:
            old = scored_workbook(status_file.parent, MODEL_NAME, benchmark, "exact_matching")
            old.rename(scored_workbook(status_file.parent, MODEL_NAME, benchmark, "chatgpt-0125"))
            status["datasets"][benchmark]["judge_model"] = "chatgpt-0125"
        run["judge"] = "chatgpt-0125"
        self.json(status_file, status)
        self.json(summary, evaluated)
        with self.assertRaisesRegex(ValueError, "same judge and judging policy"):
            self.collect()

    def test_misplaced_job_and_stage1_receipts_cannot_be_labeled_as_final_source(self):
        path = self.training / "normal/checkpoints.json"
        original = json.loads(path.read_text())
        for key, value in (("task", "baseline"), ("recipe", "i2t-only"), ("stage", "i2i")):
            data = json.loads(json.dumps(original))
            data["records"][0][key] = value
            self.json(path, data)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.collect()
        self.json(path, original)

    def test_changed_model_rng_and_experiment_receipts_are_rejected(self):
        path = self.training / "normal/checkpoints.json"
        original = json.loads(path.read_text())
        data = json.loads(json.dumps(original))
        data["records"][0]["model_seed"] = 17
        self.json(path, data)
        with self.assertRaisesRegex(ValueError, "model RNG"):
            self.collect()
        self.json(path, original)
        receipt = json.loads(self.receipt.read_text())
        receipt["config_sha256"] = "0" * 64
        self.json(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "different experiment"):
            self.collect()

    def test_shared_stage1_job_is_skipped_and_final_stage_is_planned_normally(self):
        self.spec["jobs"].insert(1, {"id": "normal_stage1", "task_id": "i2i:surface_normals",
            "training": {"task": "normal", "recipe": "r2", "i2i_manifest": "normal/train.jsonl",
                         "stop_after_stage1": True, "seeds": [42], "model_seeds": [4396]}})
        self.spec["jobs"][-1]["stage1_from"] = "normal_stage1"
        self.json(self.config, self.spec)
        receipt = json.loads(self.receipt.read_text())
        receipt.update(experiment=self.spec, config_sha256=sha256(self.config))
        self.json(self.receipt, receipt)
        final = self.training / "normal/checkpoints.json"
        records = json.loads(final.read_text())
        for row in records["records"]:
            row["initialization"] = str(self.training / "normal_stage1/shared/model.safetensors")
        self.json(final, records)
        result = self.collect()
        self.assertEqual(set(result["models"]), {"baseline", "normal"})
        self.assertEqual(len(result["runs"]), 4)

    def test_missing_benchmark_and_unfinished_status_stop_collection(self):
        summary = self.evaluation / "normal/summary.json"
        original = json.loads(summary.read_text())
        altered = json.loads(json.dumps(original))
        altered["runs"][0]["metrics"].pop("BLINK")
        self.json(summary, altered)
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.collect()
        self.json(summary, original)
        status = Path(original["runs"][0]["status_file"])
        data = json.loads(status.read_text())
        data["datasets"]["BLINK"]["status"] = "running"
        self.json(status, data)
        with self.assertRaisesRegex(ValueError, "status differs"):
            self.collect()

    def test_evaluated_checkpoint_and_seed_coverage_must_match_training(self):
        path = self.evaluation / "normal/summary.json"
        original = json.loads(path.read_text())
        changed = json.loads(json.dumps(original))
        changed["runs"][0]["checkpoint"] = "/different/weights.safetensors"
        self.json(path, changed)
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            self.collect()
        changed = json.loads(json.dumps(original))
        changed["runs"].append(changed["runs"][0])
        self.json(path, changed)
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation seeds"):
            self.collect()
        changed["runs"] = original["runs"][:1]
        self.json(path, changed)
        with self.assertRaisesRegex(ValueError, "Missing evaluation seeds"):
            self.collect()


if __name__ == "__main__":
    unittest.main()
