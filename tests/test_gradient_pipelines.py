import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from omnitaskonomy.data.common import sha256
from omnitaskonomy.gradients.artifacts import load_artifact, save_artifact
from omnitaskonomy.gradients.cli import main
from omnitaskonomy.gradients.manifest import REFERENCE_TASKS, verify_manifest
from omnitaskonomy.taxonomy import load_taxonomy


def synthetic_extract(manifest, model_path, output, *, modules, device):
    frozen = verify_manifest(manifest)
    parameters = [{"name": f"input_{layer}", "module": "llm.input_ln.weight", "layer": layer,
                   "shape": [3], "representation": "raw", "sketch_seed": None} for layer in range(2)]
    parameters.append({"name": "post_0", "module": "llm.post_attn_ln.weight", "layer": 0,
                       "shape": [3], "representation": "raw", "sketch_seed": None})
    parameters = [item for item in parameters if "all" in modules or item["module"] in modules]
    rows = [{**row, "loss": 1., "loss_tokens": int(row["uid"]) + 1} for row in frozen["rows"]]
    arrays = {parameter["name"]: np.array([
        [1 + index / 10, int(row["uid"]) + 1, 1 if row["objective"] == "i2i" else -1]
        for row in rows], dtype=np.float32) for index, parameter in enumerate(parameters)}
    save_artifact(output, rows, parameters, arrays,
                  {"manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(model_path / "ema.safetensors")},
                  folds=frozen["folds"])


class GradientPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "ema.safetensors").write_bytes(b"synthetic checkpoint")
        self.reference = self.root / "reference"
        self.matrix = self.root / "matrix"
        self.extraction = patch("omnitaskonomy.gradients.extract.extract", side_effect=synthetic_extract)
        self.extraction.start()
        self.addCleanup(self.extraction.stop)

    def config(self, name, groups, *, paired=False):
        directory = self.root / name
        directory.mkdir()
        specifications = []
        for group_index, (task, objective) in enumerate(groups):
            records = []
            for index in range(5):
                image = directory / f"{group_index}_{index}.png"
                Image.new("RGB", (2, 2), (group_index, index, 0)).save(image)
                fields = ({"source_image": image.name, "target_image": image.name, "prompt": "Draw."}
                          if objective == "i2i" else {"image": image.name, "conversations": [
                              {"from": "human", "value": "<image> Which?"}, {"from": "gpt", "value": "A"}]})
                records.append({"uid": str(index), "source_uid": f"{task}:{index}", "fold": index, **fields})
            manifest = directory / f"group_{group_index}.jsonl"
            manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
            specifications.append({"task": task, "objective": objective, "manifest": manifest.name})
        path = directory / "config.json"
        path.write_text(json.dumps({"schema_version": 1, "folds": 5, "samples_per_group": 5,
                                    "groups": specifications, "paired_reference": paired}))
        return path

    def modules(self, **options):
        config = self.config("reference_data", [(task, objective) for task in sorted(REFERENCE_TASKS)
                                               for objective in ("i2i", "i2t")], paired=True)
        arguments = ["modules", "--config", str(config), "--model-path", str(self.model),
                     "--output", str(self.reference)]
        for option, value in options.items():
            arguments.extend(["--" + option, str(value)])
        main(arguments)
        return config

    def matrix_arguments(self, config):
        return ["matrix", "--config", str(config), "--reference", str(self.reference),
                "--model-path", str(self.model), "--output", str(self.matrix), "--repeats", "4", "--batch-size", "2"]

    def test_modules_produces_module_and_layer_cosines_and_rejects_overwrite(self):
        config = self.modules()
        report = json.loads((self.reference / "pca/report.json").read_text())
        self.assertEqual({row["module"] for row in report["summary"]},
                         {"llm.input_ln.weight", "llm.post_attn_ln.weight"})
        self.assertEqual({row["layer"] for row in report["summary"] if row["module"] == "llm.input_ln.weight"},
                         {None, 0, 1})
        self.assertTrue(all(row["n"] == 5 and np.isfinite(row["cosine_mean"]) for row in report["summary"]))
        before = sha256(self.reference / "pca/report.json")
        with self.assertRaises(FileExistsError):
            main(["modules", "--config", str(config), "--model-path", str(self.model),
                  "--output", str(self.reference)])
        self.assertEqual(sha256(self.reference / "pca/report.json"), before)

    def test_modules_prepares_default_reference_and_reuses_it(self):
        source = self.config("reference_data", [(task, objective) for task in sorted(REFERENCE_TASKS)
                                               for objective in ("i2i", "i2t")], paired=True)
        transfer = self.root / "data/prepared/gradients/transfer.json"
        transfer.parent.mkdir(parents=True)
        transfer.write_text("existing transfer config\n")
        def prepare(selection, output):
            shutil.copytree(source.parent, output)
            (output / "config.json").rename(output / "reference_config.json")
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("omnitaskonomy.gradients.reference_data.prepare_recipe_reference", side_effect=prepare):
                main(["modules", "--model-path", str(self.model), "--output", str(self.reference)])
            with patch("omnitaskonomy.gradients.reference_data.prepare_recipe_reference",
                       side_effect=AssertionError("Completed reference should be reused")):
                main(["modules", "--model-path", str(self.model), "--output", str(self.root / "repeat")])
        finally:
            os.chdir(previous)
        for output in (self.reference, self.root / "repeat"):
            report = json.loads((output / "pca/report.json").read_text())
            self.assertTrue(report["summary"])
        self.assertEqual(transfer.read_text(), "existing transfer config\n")

    def test_explicit_missing_reference_is_not_replaced_with_default_data(self):
        with patch("omnitaskonomy.gradients.reference_data.prepare_recipe_reference",
                   side_effect=AssertionError("Explicit configuration must be respected")):
            with self.assertRaises(FileNotFoundError):
                main(["modules", "--config", str(self.root / "custom.json"),
                      "--model-path", str(self.model), "--output", str(self.reference)])
        self.assertFalse(self.reference.exists())

    def test_matrix_preserves_reference_folds_and_emits_all_475_cells(self):
        self.modules()
        tasks = load_taxonomy()["tasks"]
        config = self.config("matrix_data", [(task["id"], task["modality"]) for task in tasks])
        content = json.loads(config.read_text())
        content["reference"] = "obsolete_reference.json"
        first = config.parent / content["groups"][0]["manifest"]
        records = [json.loads(line) for line in first.read_text().splitlines()]
        for index, record in enumerate(records):
            record["source_uid"] = f"jigsaw:{index}"
            del record["fold"]
        first.write_text("".join(json.dumps(record) + "\n" for record in records))
        config.write_text(json.dumps(content))
        main(self.matrix_arguments(config))
        report = json.loads((self.matrix / "report.json").read_text())
        self.assertEqual(len(report["cells"]), 475)
        self.assertEqual({row["source_task_id"] for row in report["cells"]},
                         {task["id"] for task in tasks if task["modality"] == "i2i"})
        self.assertEqual({row["node_id"] for row in report["cells"]},
                         {task["id"] for task in tasks if task["modality"] == "i2t"})
        self.assertTrue(all(-1.000001 <= row["alignment"] <= 1.000001 and row["mcse"] >= 0 for row in report["cells"]))
        with (self.matrix / "report_cells.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 475)
        artifact = load_artifact(self.matrix / "artifact")
        self.assertEqual({parameter["module"] for parameter in artifact["metadata"]["parameters"]}, {"llm.input_ln.weight"})
        self.assertEqual({row["uid"]: row["fold"] for row in artifact["metadata"]["rows"]
                          if row["task"] == content["groups"][0]["task"]}, {str(i): i for i in range(5)})
        reference_report = json.loads((self.reference / "pca/report.json").read_text())
        self.assertEqual(set(report["input_audit"]["bases"].values()),
                         {row["sha256"] for row in reference_report["folds"]
                          if row["module"] == "llm.input_ln.weight" and row["parameter"] is None})

    def test_matrix_rejects_subset_before_creating_output(self):
        config = self.config("subset", [("i2i:not_a_task", "i2i"), ("i2t:not_a_task", "i2t")])
        with self.assertRaisesRegex(ValueError, "all 19 I2I and 25 I2T"):
            main(self.matrix_arguments(config))
        self.assertFalse(self.matrix.exists())

    def test_matrix_rejects_changed_checkpoint_before_creating_output(self):
        self.modules()
        config = self.config("matrix_data", [(task["id"], task["modality"]) for task in load_taxonomy()["tasks"]])
        (self.model / "ema.safetensors").write_bytes(b"another checkpoint")
        with self.assertRaisesRegex(ValueError, "reference checkpoint"):
            main(self.matrix_arguments(config))
        self.assertFalse(self.matrix.exists())

    def test_matrix_rejects_swapped_frozen_reference_before_creating_output(self):
        self.modules()
        config = self.config("matrix_data", [(task["id"], task["modality"]) for task in load_taxonomy()["tasks"]])
        frozen_path = self.reference / "frozen.json"
        frozen = json.loads(frozen_path.read_text())
        frozen["sample_seed"] += 1
        frozen_path.write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, "frozen manifest differ"):
            main(self.matrix_arguments(config))
        self.assertFalse(self.matrix.exists())

    def test_matrix_rejects_reference_without_input_layernorm_before_creating_output(self):
        self.modules(modules="llm.post_attn_ln.weight")
        config = self.config("matrix_data", [(task["id"], task["modality"]) for task in load_taxonomy()["tasks"]])
        with self.assertRaisesRegex(ValueError, "Missing concatenated reference PCA"):
            main(self.matrix_arguments(config))
        self.assertFalse(self.matrix.exists())

    def test_modules_rejects_incomplete_reference_before_extraction(self):
        config = self.config("subset", [("jigsaw", "i2i"), ("jigsaw", "i2t")], paired=True)
        with self.assertRaisesRegex(ValueError, "six paired tasks and five folds"):
            main(["modules", "--config", str(config), "--model-path", str(self.model),
                  "--output", str(self.reference)])
        self.assertFalse((self.reference / "artifact").exists())

    def test_custom_model_backward_pca_and_full_matrix(self):
        import torch
        from omnitaskonomy.examples.tiny_umm import TinyUMM

        self.extraction.stop()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(17)
        torch.save(TinyUMM().state_dict(), self.model / "model.pt")
        factory = "omnitaskonomy.examples.tiny_umm:create_adapter"
        self.modules(adapter=factory, device="cpu")
        reference = load_artifact(self.reference / "artifact")
        self.assertEqual({p["module"] for p in reference["metadata"]["parameters"]}, {"shared"})
        self.assertTrue(all(row["raw_norms"]["shared.weight"] > 0 for row in reference["metadata"]["rows"]))
        config = self.config("matrix_data", [(task["id"], task["modality"]) for task in load_taxonomy()["tasks"]])
        arguments = self.matrix_arguments(config) + ["--adapter", factory, "--device", "cpu", "--module", "shared"]
        main(arguments)
        report = json.loads((self.matrix / "report.json").read_text())
        self.assertEqual(len(report["cells"]), 475)
        self.assertTrue(all(-1.000001 <= row["alignment"] <= 1.000001 for row in report["cells"]))


if __name__ == "__main__":
    unittest.main()
