from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from omnitaskonomy.examples.tiny_umm import TinyAdapter, TinyUMM
from omnitaskonomy.gradients.artifacts import load_artifact
from omnitaskonomy.gradients.extract import extract
from omnitaskonomy.gradients.manifest import freeze
from omnitaskonomy.gradients.pca import analyze_reference
from omnitaskonomy.umm import Loss, ParameterSpec, adapter_provenance, load_adapter

FACTORY = "omnitaskonomy.examples.tiny_umm:create_adapter"


class BufferedAdapter:
    def __init__(self, path):
        self.model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Linear(2, 1))
        torch.save(self.model.state_dict(), path)
        self.checkpoint_files = (path,)

    def parameter_specs(self):
        return [ParameterSpec(name, "shared", "shared") for name, _ in self.model.named_parameters()]

    def loss(self, records, objective, context):
        index = int(records[0]["uid"])
        prediction = self.model(torch.tensor([[index + 1., 2.], [3., index + 4.]]))
        return Loss(prediction.square().sum(), 2)


class GenerationAdapter(TinyAdapter):
    def parameter_specs(self):
        for spec in super().parameter_specs():
            yield replace(spec, module="generation" if spec.role == "generation" else None,
                          zero_objectives=("i2t",) if spec.role == "generation" else ())


class DetachedAdapter(GenerationAdapter):
    def loss(self, records, objective, context):
        loss = super().loss(records, objective, context)
        return Loss(loss.total.detach(), loss.count)


class UMMGradientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        torch.manual_seed(17)
        torch.save(TinyUMM().state_dict(), self.base / "model.pt")
        groups = []
        for index in range(2):
            Image.new("RGB", (4, 4), (30 + index * 80, 20, 120)).save(self.root / f"{index}.png")
        for objective in ("i2i", "i2t"):
            rows = []
            for index in range(2):
                fields = ({"source_image": f"{index}.png", "target_image": f"{index}.png", "prompt": "Draw."}
                          if objective == "i2i" else {"image": f"{index}.png", "conversations": [
                              {"from": "human", "value": "<image> Which?"}, {"from": "gpt", "value": "A"}]})
                rows.append({"uid": str(index), "source_uid": f"fixture:{index}", "fold": index, **fields})
            manifest = self.root / f"{objective}.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            groups.append({"task": "fixture", "objective": objective, "manifest": manifest.name})
        config = self.root / "config.json"
        config.write_text(json.dumps({"schema_version": 1, "groups": groups, "folds": 2,
                                      "samples_per_group": 2, "paired_reference": True}))
        self.frozen = self.root / "frozen.json"
        freeze(config, self.frozen)

    def test_analysis_keeps_batchnorm_buffers_and_parameters_unchanged(self):
        adapter = BufferedAdapter(self.root / "buffered.pt")
        adapter.model[0].weight.requires_grad_(False)
        before = {name: value.clone() for name, value in adapter.model.state_dict().items()}
        output = self.root / "buffered-gradients"
        with patch("omnitaskonomy.gradients.extract.load_adapter", return_value=adapter):
            extract(self.frozen, self.base, output, adapter=FACTORY, device="cpu", modules=["shared"])
        for name, value in adapter.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        artifact = load_artifact(output)
        self.assertTrue(np.any(artifact["arrays"]["0.weight"] != 0))

    def test_disconnected_objective_emits_only_explicitly_declared_zero_gradients(self):
        adapter = GenerationAdapter(self.base, None, "cpu")
        output = self.root / "generation-gradients"
        with patch("omnitaskonomy.gradients.extract.load_adapter", return_value=adapter):
            extract(self.frozen, self.base, output, adapter=FACTORY, device="cpu", modules=["generation"])
        artifact = load_artifact(output)
        for index, row in enumerate(artifact["metadata"]["rows"]):
            for array in artifact["arrays"].values():
                if row["objective"] == "i2t":
                    np.testing.assert_array_equal(array[index], 0)
                else:
                    self.assertTrue(np.any(array[index] != 0))
        report = analyze_reference(artifact, self.root / "generation-pca")
        self.assertFalse(report["pairs"])
        self.assertTrue(report["undefined"])
        self.assertTrue(all(row["objectives"] == ["i2t"] for row in report["undefined"]))

    def test_undeclared_detached_objective_fails_without_completed_artifact(self):
        adapter = DetachedAdapter(self.base, None, "cpu")
        output = self.root / "detached-gradients"
        with patch("omnitaskonomy.gradients.extract.load_adapter", return_value=adapter):
            with self.assertRaisesRegex(ValueError, "Detached gradient objective"):
                extract(self.frozen, self.base, output, adapter=FACTORY, device="cpu", modules=["generation"])
        self.assertFalse((output / "metadata.json").exists())

    def test_reference_identity_allows_copied_checkpoint_and_rejects_changed_weights(self):
        original = load_adapter(FACTORY, self.base, device="cpu")
        identity = adapter_provenance(original, FACTORY)["checkpoint_sha256"]
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copyfile(self.base / "model.pt", relocated / "model.pt")
        first, second = self.root / "original-gradients", self.root / "copied-gradients"
        for model, output in ((self.base, first), (relocated, second)):
            extract(self.frozen, model, output, adapter=FACTORY, device="cpu", modules=["shared"],
                    reference_identity=identity)
        first_artifact, second_artifact = load_artifact(first), load_artifact(second)
        for name, array in first_artifact["arrays"].items():
            np.testing.assert_array_equal(array, second_artifact["arrays"][name])
        changed = load_adapter(FACTORY, relocated, device="cpu")
        with torch.no_grad():
            changed.model.shared.weight.add_(1)
        changed.save_checkpoint(relocated)
        rejected = self.root / "rejected-gradients"
        with self.assertRaisesRegex(ValueError, "reference checkpoint"):
            extract(self.frozen, relocated, rejected, adapter=FACTORY, device="cpu", modules=["shared"],
                    reference_identity=identity)
        self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
