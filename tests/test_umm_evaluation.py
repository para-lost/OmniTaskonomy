import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from omnitaskonomy.evaluate import ROOT, UMM_MODEL_NAME, build_plan, main, parser, read_status
from omnitaskonomy.eval_status import record_evaluation
from omnitaskonomy.data.common import sha256

FACTORY = "omnitaskonomy.examples.tiny_umm:create_adapter"


class UMMEvaluationTests(unittest.TestCase):
    def test_custom_dry_run_preserves_model_files_options_and_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = root / "options.json"
            options.write_text('{"example_option": 7}')
            output = root / "evaluation"
            for checkpoint in (root / "checkpoint-directory", root / "weights.pt"):
                with self.subTest(checkpoint=checkpoint):
                    capture = io.StringIO()
                    with contextlib.redirect_stdout(capture):
                        main(["--adapter", FACTORY, "--adapter-options", str(options),
                              "--model-path", str(root / "base"), "--checkpoint", str(checkpoint),
                              "--device", "cpu", "--benchmarks", "MMVP", "--output-dir", str(output),
                              "--seeds", "42", "123", "--dry-run"])
                    plans = json.loads(capture.getvalue())
                    self.assertEqual([row["seed"] for row in plans], [42, 123])
                    for plan in plans:
                        self.assertEqual(plan["checkpoint"], str(checkpoint))
                        config = plan["config"]["model"][UMM_MODEL_NAME]
                        self.assertEqual(config["class"], "OmniTaskonomyUMM")
                        self.assertEqual(config["adapter"], FACTORY)
                        self.assertEqual(config["adapter_options"], {"example_option": 7})
                        self.assertEqual(config["device"], "cpu")
                        self.assertNotIn("parameter_dtype", config)
                    self.assertFalse(output.exists())

    def test_training_receipts_supply_adapter_and_final_checkpoint_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [dict(task="normal", recipe="i2i-to-i2t", seed=seed, stage=stage,
                            model_path="base", checkpoint=f"seed_{seed}/{stage}", adapter=FACTORY,
                            adapter_options={"example_option": 7}, device="cpu")
                       for seed in (42, 123) for stage in ("i2i", "i2t")]
            run_file = root / "checkpoints.json"
            run_file.write_text(json.dumps({"schema_version": 1, "records": records}))
            argv = ["--run-file", str(run_file), "--output-dir", str(root / "evaluation"),
                    "--seeds", "42", "123"]
            plans = build_plan(parser().parse_args(argv))
            for plan in plans:
                self.assertEqual(plan["checkpoint"], str(root / f"seed_{plan['seed']}/i2t"))
                self.assertEqual(plan["model_path"], str(root / "base"))
                self.assertEqual(plan["adapter"], FACTORY)
                self.assertEqual(plan["adapter_options"], {"example_option": 7})
                self.assertEqual(plan["device"], "cpu")
            options = root / "override.json"
            options.write_text('{"example_option": 9}')
            override = build_plan(parser().parse_args(argv + ["--adapter-options", str(options)]))
            self.assertEqual(override[0]["adapter_options"], {"example_option": 9})

    def test_base_adapter_loads_without_bagel_checkpoint_defaults(self):
        plan = build_plan(parser().parse_args([
            "--model-path", "/model", "--adapter", FACTORY, "--device", "cpu",
            "--output-dir", "/output",
        ]))[0]
        self.assertIsNone(plan["checkpoint"])

    def test_distributed_device_and_model_specific_precision_are_explicit(self):
        argv = ["--model-path", "/model", "--adapter", FACTORY, "--output-dir", "/output"]
        with self.assertRaisesRegex(ValueError, "each worker"):
            build_plan(parser().parse_args(argv + ["--nproc-per-node", "2", "--device", "cuda:3"]))
        with self.assertRaisesRegex(ValueError, "custom precision"):
            build_plan(parser().parse_args(argv + ["--parameter-dtype", "float32"]))
        plan = build_plan(parser().parse_args(argv + ["--nproc-per-node", "2", "--device", "cuda:0"]))[0]
        self.assertEqual(plan["config"]["model"][UMM_MODEL_NAME]["device"], "cuda:0")

    def test_changed_custom_checkpoint_cannot_reuse_old_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "weights.pt"
            weights.write_bytes(b"initial weights")
            record = dict(task="normal", recipe="i2i-to-i2t", seed=42, stage="i2t", model_path=str(root),
                          checkpoint=str(checkpoint), adapter=FACTORY, adapter_options={}, device="cpu",
                          checkpoint_files_sha256={weights.name: sha256(weights)})
            run_file = root / "checkpoints.json"
            run_file.write_text(json.dumps({"schema_version": 1, "records": [record]}))
            argv = ["--run-file", str(run_file), "--output-dir", str(root / "evaluation"), "--reuse"]
            plan = build_plan(parser().parse_args(argv))[0]
            evaluation = Path(plan["directory"])
            evaluation.mkdir(parents=True)
            configuration = evaluation / "vlmeval.json"
            receipt = evaluation / "evaluation.json"
            configuration.write_text(json.dumps(plan["config"]))
            receipt.write_text(json.dumps(plan))
            original = receipt.read_bytes()
            weights.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "files differ from the training receipt"):
                main(argv)
            self.assertEqual(receipt.read_bytes(), original)
            record["checkpoint_files_sha256"][weights.name] = sha256(weights)
            run_file.write_text(json.dumps({"schema_version": 1, "records": [record]}))
            with self.assertRaisesRegex(ValueError, "new output directory"):
                main(argv)
            self.assertEqual(receipt.read_bytes(), original)

    def test_custom_model_runs_real_benchmark_inference_and_scoring_on_cpu(self):
        import torch
        import pandas as pd
        from PIL import Image
        from omnitaskonomy.examples.tiny_umm import TinyUMM

        sys.path.insert(0, str(ROOT / "VLMEvalKit"))
        import vlmeval.config
        import vlmeval.vlm
        from vlmeval.dataset.image_mcq import ImageMCQDataset
        from vlmeval.inference import infer_data_job
        from vlmeval.smp import get_pred_file_path, load

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyUMM()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.scale.fill_(1)
                model.shared.weight[0, 0] = 1
                model.understanding.weight[0, 0] = -1
                model.understanding.bias[0] = 0.2
                model.understanding.weight[1, 0] = 1
            checkpoint = root / "custom.pt"
            torch.save(model.state_dict(), checkpoint)
            plan = build_plan(parser().parse_args([
                "--adapter", FACTORY, "--model-path", str(root), "--checkpoint", str(checkpoint),
                "--device", "cpu", "--benchmarks", "MMVP", "--output-dir", str(root / "eval"),
                "--judge", "exact_matching",
            ]))[0]
            config = dict(plan["config"]["model"][UMM_MODEL_NAME])
            wrapper = getattr(vlmeval.vlm, config.pop("class"))(**config)
            self.assertFalse(wrapper.adapter.model.training)

            rows = []
            for index, (red, answer) in enumerate(((0, "A"), (255, "B"))):
                image = root / f"{index}.png"
                Image.new("RGB", (16, 16), (red, 0, 0)).save(image)
                rows.append(dict(index=index, question="What color is shown?", A="black", B="red",
                                 answer=answer, image_path=str(image), split="test"))
            dataset = ImageMCQDataset.__new__(ImageMCQDataset)
            dataset.dataset_name = "MMVP"
            dataset.meta_only = True
            dataset.img_root = str(root)
            dataset.data = pd.DataFrame(rows)
            evaluation = Path(plan["directory"])
            result_dir = evaluation / "results" / UMM_MODEL_NAME / "cpu-fixture"
            result_dir.mkdir(parents=True)
            infer_data_job(wrapper, str(result_dir), UMM_MODEL_NAME, dataset, seed=42)
            predictions = get_pred_file_path(str(result_dir), UMM_MODEL_NAME, "MMVP")
            self.assertEqual(load(predictions)["prediction"].tolist(), ["A", "B"])
            scores = dataset.evaluate(predictions, model="exact_matching", nproc=1)
            record_evaluation(result_dir, "MMVP", "done", judge="exact_matching", scores=scores)
            (evaluation / "vlmeval.json").write_text(json.dumps(plan["config"]))
            (evaluation / "evaluation.json").write_text(json.dumps(plan))
            result = read_status(plan, ["MMVP"])
            self.assertEqual(result["checkpoint"], str(checkpoint))
            self.assertEqual(result["adapter"], FACTORY)
            self.assertEqual(result["adapter_options"], {})
            self.assertEqual(result["metrics"]["MMVP"]["split=test|Overall"], 1.0)


if __name__ == "__main__":
    unittest.main()
