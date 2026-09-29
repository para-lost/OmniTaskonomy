import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omnitaskonomy.evaluate import (
    MODEL_NAME, ROOT, aggregate_results, build_plan, main, parser, read_status,
)


class EvaluationTests(unittest.TestCase):
    def arguments(self, directory, *extra):
        return parser().parse_args(["--output-dir", str(directory / "eval"), *extra])

    def test_default_base_plan_and_distributed_seed_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            default = build_plan(self.arguments(directory, "--model-path", str(directory / "base")))
            self.assertEqual(default[0]["seed"], 42)
            self.assertEqual(default[0]["config"]["model"][MODEL_NAME]["parameter_dtype"], "bfloat16")
            self.assertEqual(Path(default[0]["checkpoint"]).name, "ema.safetensors")
            self.assertEqual(default[0]["judge"], "chatgpt-0125")
            self.assertEqual(default[0]["judge_args"], {"llm_first": True})
            command = default[0]["command"]
            self.assertEqual(json.loads(command[command.index("--judge-args") + 1]), {"llm_first": True})
            explicit_em = build_plan(self.arguments(
                directory, "--model-path", str(directory / "base"), "--judge", "exact_matching"))[0]
            self.assertEqual(explicit_em["judge_args"], {})
            self.assertNotIn("--judge-args", explicit_em["command"])
            plans = build_plan(self.arguments(
                directory, "--model-path", str(directory / "base"), "--checkpoint", str(directory / "finetuned"),
                "--seeds", "42", "123", "--nproc-per-node", "2", "--benchmarks", "MMVP", "CV-Bench-3D",
            ))
            self.assertNotEqual(plans[0]["directory"], plans[1]["directory"])
            self.assertEqual(Path(plans[0]["checkpoint"]).name, "model.safetensors")
            self.assertEqual(plans[1]["config"]["model"][MODEL_NAME]["seed"], 123)
            self.assertEqual(plans[0]["config"]["model"][MODEL_NAME]["class"], "OmniTaskonomyBAGEL")
            self.assertIn("torch.distributed.run", plans[0]["command"])
            self.assertEqual(plans[0]["command"][plans[0]["command"].index("--num-seeds") + 1], "1")
            self.assertEqual(plans[1]["command"][plans[1]["command"].index("--seed-start") + 1], "123")
            self.assertEqual(plans[0]["config"]["data"]["CV-Bench-3D"]["class"], "CVBench")

    def test_parameter_precision_is_saved_and_cannot_reuse_another_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            arguments = ["--model-path", str(directory / "base"), "--output-dir", str(directory / "eval")]
            plan = build_plan(parser().parse_args(arguments + ["--parameter-dtype", "float32"]))[0]
            self.assertEqual(plan["config"]["model"][MODEL_NAME]["parameter_dtype"], "float32")
            config_path = Path(plan["directory"]) / "vlmeval.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(plan["config"]))
            other = build_plan(parser().parse_args(arguments))[0]
            with self.assertRaisesRegex(ValueError, "config differs"):
                read_status(other, ["MMVP"])

    def test_training_manifest_uses_final_stage_for_each_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = []
            for seed in [42, 123]:
                for stage in ["stage1", "stage2"]:
                    records.append({"task": "jigsaw", "recipe": "i2i_then_i2t", "seed": seed,
                                    "stage": stage, "model_path": "base",
                                    "checkpoint": f"seed_{seed}/{stage}/checkpoints/0000001"})
            manifest = directory / "checkpoints.json"
            manifest.write_text(json.dumps({"schema_version": 1, "records": records}))
            plans = build_plan(self.arguments(directory, "--run-file", str(manifest), "--seeds", "42", "123"))
            self.assertTrue(all("stage2" in row["checkpoint"] for row in plans))
            self.assertEqual(plans[0]["model_path"], str(directory / "base"))
            with self.assertRaisesRegex(ValueError, "no checkpoint"):
                build_plan(self.arguments(directory, "--run-file", str(manifest), "--seeds", "999"))
            with self.assertRaisesRegex(ValueError, "already supplies"):
                build_plan(self.arguments(directory, "--run-file", str(manifest), "--model-path", "base"))

    def test_ambiguous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = directory / "checkpoints.json"
            manifest.write_text(json.dumps({"schema_version": 1, "records": [
                {"task": "jigsaw", "recipe": "a", "seed": 42},
                {"task": "zoomin", "recipe": "a", "seed": 42},
            ]}))
            with self.assertRaisesRegex(ValueError, "one task/recipe"):
                build_plan(self.arguments(directory, "--run-file", str(manifest)))

    def test_dry_run_is_cpu_only_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dry-run-output"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/evaluate.py"), "--model-path", "/missing/base",
                 "--seeds", "42", "123", "--output-dir", str(output), "--dry-run"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(len(json.loads(completed.stdout)), 2)
            self.assertFalse(output.exists())

    def test_aggregate_distinguishes_sample_std_and_sem_and_preserves_metric_units(self):
        rows = [
            {"seed": 42, "metrics": {"BLINK": {"Overall": 0.25}, "MMVP": {"Overall": 40.0}}},
            {"seed": 123, "metrics": {"BLINK": {"Overall": 0.50}, "MMVP": {"Overall": 50.0}}},
            {"seed": 456, "metrics": {"BLINK": {"Overall": 0.75}, "MMVP": {"Overall": 60.0}}},
        ]
        result = aggregate_results(rows)["benchmarks"]
        self.assertEqual(result["BLINK"]["Overall"]["mean"], 0.5)
        self.assertEqual(result["MMVP"]["Overall"]["mean"], 50.0)
        self.assertEqual(result["MMVP"]["Overall"]["n"], 3)
        self.assertEqual(result["MMVP"]["Overall"]["sample_std"], 10.0)
        self.assertAlmostEqual(result["MMVP"]["Overall"]["sem"], 10 / 3 ** 0.5)
        self.assertEqual(result["BLINK"]["Overall"]["sample_std"], 0.25)
        self.assertAlmostEqual(result["BLINK"]["Overall"]["sem"], 0.25 / 3 ** 0.5)
        rows[1]["metrics"].pop("MMVP")
        with self.assertRaisesRegex(ValueError, "coverage"):
            aggregate_results(rows)

    def test_single_seed_has_undefined_uncertainty_and_identical_runs_have_zero_spread(self):
        rows = [{"seed": seed, "metrics": {"BLINK": {"Overall": 0.5}}} for seed in [42, 123]]
        single = aggregate_results(rows[:1])["benchmarks"]["BLINK"]["Overall"]
        self.assertEqual(single["n"], 1)
        self.assertEqual(single["mean"], 0.5)
        self.assertIsNone(single["sample_std"])
        self.assertIsNone(single["sem"])
        repeated = aggregate_results(rows)["benchmarks"]["BLINK"]["Overall"]
        self.assertEqual(repeated["sample_std"], 0.0)
        self.assertEqual(repeated["sem"], 0.0)

    def test_status_failures_and_stale_results_do_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan = {"seed": 42, "directory": str(directory), "checkpoint": "weights.safetensors",
                    "judge": "exact_matching", "vlmeval_source_sha256": "fixture-source",
                    "config": {"model": {MODEL_NAME: {"seed": 42}}}}
            (directory / "vlmeval.json").write_text(json.dumps(plan["config"]))
            (directory / "evaluation.json").write_text(json.dumps(plan))
            status = directory / "results" / MODEL_NAME / "run-1" / "status.json"
            status.parent.mkdir(parents=True)
            payload = {"datasets": {"MMVP": {"status": "done", "judge_model": "exact_matching",
                                            "metrics": {"Overall": 50.0}}}}
            status.write_text(json.dumps(payload))
            self.assertEqual(read_status(plan, ["MMVP"])["metrics"]["MMVP"]["Overall"], 50.0)
            (directory / "evaluation.json").write_text(json.dumps({"vlmeval_source_sha256": "another-source"}))
            with self.assertRaisesRegex(ValueError, "source differs"):
                read_status(plan, ["MMVP"])
            (directory / "evaluation.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "no new status"):
                read_status(plan, ["MMVP"], changed_after_ns=status.stat().st_mtime_ns + 1)
            payload["datasets"]["MMVP"]["judge_model"] = "another-judge"
            status.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Judge differs"):
                read_status(plan, ["MMVP"])
            payload["datasets"]["MMVP"]["error_message"] = "failed prediction load"
            status.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "failed"):
                read_status(plan, ["MMVP"])
            payload = {"datasets": {"MMVP": {"status": "done", "skip_reason": "mode_infer"}}}
            status.write_text(json.dumps(payload))
            self.assertEqual(read_status(plan, ["MMVP"], mode="infer")["metrics"], {})
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                read_status(plan, ["MMVP"])
            (directory / "vlmeval.json").write_text(json.dumps({"model": "different checkpoint"}))
            with self.assertRaisesRegex(ValueError, "config differs"):
                read_status(plan, ["MMVP"])

    def test_changed_checkpoint_never_overwrites_existing_output_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base = directory / "base"
            base.mkdir()
            for name in ["llm_config.json", "vit_config.json", "ema.safetensors", "model.safetensors"]:
                (base / name).touch()
            arguments = ["--model-path", str(base), "--output-dir", str(directory / "eval")]
            plan = build_plan(parser().parse_args(arguments))[0]
            config_path = Path(plan["directory"]) / "vlmeval.json"
            config_path.parent.mkdir(parents=True)
            original = json.dumps(plan["config"])
            config_path.write_text(original)
            for extra in [[], ["--reuse"], ["--mode", "eval"]]:
                with self.subTest(extra=extra), patch("omnitaskonomy.evaluate.subprocess.run") as launch:
                    with self.assertRaisesRegex(ValueError, "new output directory"):
                        main(arguments + ["--checkpoint", str(base / "model.safetensors")] + extra)
                    self.assertEqual(config_path.read_text(), original)
                    self.assertFalse((config_path.parent / "evaluation.json").exists())
                    launch.assert_not_called()

    def test_changed_judge_policy_cannot_reuse_old_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base = directory / "base"
            base.mkdir()
            for name in ["llm_config.json", "vit_config.json", "ema.safetensors"]:
                (base / name).touch()
            arguments = ["--model-path", str(base), "--output-dir", str(directory / "eval")]
            plan = build_plan(parser().parse_args(arguments))[0]
            output = Path(plan["directory"])
            output.mkdir(parents=True)
            (output / "vlmeval.json").write_text(json.dumps(plan["config"]))
            previous = {**plan, "judge_args": {}}
            receipt = output / "evaluation.json"
            receipt.write_text(json.dumps(previous))
            with self.assertRaisesRegex(ValueError, "judge policy differs"):
                read_status(plan, ["MMVP"])
            with patch("omnitaskonomy.evaluate.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "Changed judge/policy"):
                    main(arguments + ["--reuse"])
                launch.assert_not_called()
            self.assertEqual(json.loads(receipt.read_text()), previous)


if __name__ == "__main__":
    unittest.main()
