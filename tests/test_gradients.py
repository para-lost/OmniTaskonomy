import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from omnitaskonomy.analysis.gradient_transfer import associate
from omnitaskonomy.gradients.artifacts import load_artifact, save_artifact, validate_rows
from omnitaskonomy.gradients.extract import classify_parameter
from omnitaskonomy.gradients.manifest import freeze, verify_manifest
from omnitaskonomy.gradients.minibatch import analyze_minibatches, minibatch_directions, sampling_schedule
from omnitaskonomy.gradients.norms import summarize_norms
from omnitaskonomy.gradients.pca import analyze_reference, fit_pca, load_basis, project_artifact, unit_rows
from omnitaskonomy.gradients.targets import prepare_targets
from omnitaskonomy.data.common import read_jsonl, sha256


def example_rows(n=5, task="toy"):
    return [{"task": task, "objective": objective, "uid": str(i), "source_uid": f"{task}:{i}",
             "fold": i % 5, "loss": 1., "loss_tokens": i + 1, "noise_seed": i}
            for objective in ("i2i", "i2t") for i in range(n)]


PARAMETER = {"name": "input", "module": "llm.input_ln.weight", "layer": 0, "shape": [3],
             "numel": 3, "representation": "raw", "sketch_seed": None}


class GradientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def artifact(self, name, values, rows=None, state="base"):
        return save_artifact(self.root / name, rows or example_rows(), [PARAMETER], {"input": values},
                             {"checkpoint_sha256": "base" if state == "base" else state}, state=state)

    def test_dual_pca_matches_weighted_primal_and_keeps_ties(self):
        rng = np.random.default_rng(17)
        raw = rng.normal(size=(10, 3)) * np.arange(1, 11)[:, None]
        rows = example_rows()
        # Unequal stratum sizes exercise equal task/objective weighting.
        rows[1]["task"] = "extra"
        rows[2]["task"] = "extra"
        fit = fit_pca(raw @ raw.T, rows, 0, energy=.9)
        train = fit["train"]
        directions = unit_rows(raw[train])
        strata = [(rows[i]["task"], rows[i]["objective"]) for i in train]
        weights = np.array([1 / (len(set(strata)) * strata.count(key)) for key in strata])
        moment = (directions * weights[:, None]).T @ directions
        eigenvalues, vectors = np.linalg.eigh(moment)
        primal = vectors[:, ::-1][:, :fit["k"]]
        dual = raw[train].T @ fit["coefficients"]
        np.testing.assert_allclose(dual @ dual.T, primal @ primal.T, atol=1e-12)
        expected = unit_rows(raw[train] @ primal)
        angular = (expected * weights[:, None]).T @ expected
        self.assertAlmostEqual(fit["d_eff"], np.trace(angular)**2 / np.sum(angular**2))
        tied_rows = [{"task": "a", "objective": "i2i", "fold": 1}] * 3
        self.assertEqual(fit_pca(np.eye(3), tied_rows, 0, energy=.5)["k"], 3)

    def test_reference_holds_pairs_out_and_projects_raw_magnitudes(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(10, 3)).astype(np.float32)
        artifact = self.artifact("reference", values)
        report = analyze_reference(artifact, self.root / "pca", layers=False)
        self.assertEqual(len(report["pairs"]), 5)
        bases = load_basis(self.root / "pca", artifact)
        fit = bases[0]
        self.assertTrue(all(artifact["metadata"]["rows"][i]["fold"] != 0 for i in fit["train"]))
        projected = project_artifact(artifact, artifact, fit)
        expected = values.astype(float) @ values[fit["train"]].astype(float).T @ fit["coefficients"]
        np.testing.assert_allclose(projected, expected)
        changed = values.astype(float).copy()
        changed[[0, 5]] *= 100
        new_fit = fit_pca(changed @ changed.T, example_rows(), 0)
        original_fit = fit_pca(values.astype(float) @ values.astype(float).T, example_rows(), 0)
        np.testing.assert_allclose(new_fit["eigenvalues"], original_fit["eigenvalues"], atol=1e-12)
        self.assertAlmostEqual(new_fit["d_eff"], report["folds"][0]["d_eff"])

    def test_weighted_raw_minibatch_not_mean_normalized_gradients(self):
        gradients = np.array([[10., 0.], [0., 1.]])
        tokens = np.array([1, 3])
        indices = np.array([[0, 1], [1, 1]])
        directions = minibatch_directions(gradients, tokens, indices, 2)
        np.testing.assert_allclose(directions[0], np.array([10, 3]) / np.sqrt(109))
        np.testing.assert_allclose(directions[1], [0, 1])
        self.assertFalse(np.allclose(directions[0], np.array([1, 3]) / np.sqrt(10)))
        np.testing.assert_array_equal(sampling_schedule("a", 0, 10, 5), sampling_schedule("a", 0, 10, 9)[:5])

    def test_minibatch_pipeline_has_independent_groups_and_fold_weights(self):
        rng = np.random.default_rng(4)
        reference = self.artifact("reference", rng.normal(size=(10, 3)).astype(np.float32))
        analyze_reference(reference, self.root / "pca", layers=False)
        bases = load_basis(self.root / "pca", reference)
        report = analyze_minibatches(reference, reference, bases, repeats=20, batch_size=2)
        self.assertEqual(len(report["cells"]), 1)
        self.assertAlmostEqual(sum(row["weight"] for row in report["folds"]), 1)
        self.assertEqual(len(report["schedules"]), 10)
        changed = copy.deepcopy(reference["metadata"])
        changed["rows"][0]["fold"] = 2
        with self.assertRaisesRegex(ValueError, "crosses PCA folds"):
            analyze_minibatches({**reference, "metadata": changed}, reference, bases, repeats=20)

    def test_fold_pairing_hash_and_checkpoint_sample_guards(self):
        rows = example_rows()
        rows[5]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "crosses folds"):
            validate_rows(rows, paired=True)
        artifact = self.artifact("raw", np.ones((10, 3), dtype=np.float32))
        path = self.root / "raw" / artifact["metadata"]["parameters"][0]["file"]
        np.save(path, np.zeros((10, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            load_artifact(self.root / "raw")

    def test_norms_average_individual_norms_and_pair_checkpoints(self):
        values = np.tile([[1., 0, 0], [-1, 0, 0]], (5, 1)).astype(np.float32)
        baseline = self.artifact("base", values)
        trained = self.artifact("trained", values * 2, state="3k")
        report = summarize_norms([baseline, trained])
        self.assertEqual([row["mean_l2"] for row in report["summary"]], [1, 1, 2, 2])
        self.assertEqual(report["summary"][-1]["mean_l2_ratio_to_base"], 2)
        trained["metadata"]["rows"][0]["noise_seed"] = 999
        with self.assertRaisesRegex(ValueError, "noise draw"):
            summarize_norms([baseline, trained])

    def test_freeze_is_portable_and_detects_changed_image(self):
        for i in range(5):
            Image.new("RGB", (8, 8), (i, i, i)).save(self.root / f"{i}.png")
        groups = []
        for objective in ("i2i", "i2t"):
            records = [{"uid": str(i), "fold": i,
                        **({"source_image": f"{i}.png", "target_image": f"{i}.png", "prompt": "Draw."}
                           if objective == "i2i" else {"image": f"{i}.png", "conversations":
                               [{"from": "human", "value": "<image> Which?"}, {"from": "gpt", "value": "A"}]})}
                       for i in range(5)]
            filename = f"{objective}.jsonl"
            (self.root / filename).write_text("".join(json.dumps(row) + "\n" for row in records))
            groups.append({"task": "toy", "objective": objective, "manifest": filename})
        config = self.root / "config.json"
        config.write_text(json.dumps({"schema_version": 1, "groups": groups, "samples_per_group": 5, "paired_reference": True}))
        frozen = self.root / "frozen.json"
        report = freeze(config, frozen)
        self.assertEqual(len(report["rows"]), 10)
        verify_manifest(frozen)
        Image.new("RGB", (8, 8), "red").save(self.root / "0.png")
        with self.assertRaisesRegex(ValueError, "image changed"):
            verify_manifest(frozen)

    def test_transfer_join_uses_task_ids_and_preserves_missing_measurements(self):
        gradients = {"schema_version": 1, "input_audit": {"hash": "g"}, "cells": [
            {"source_task_id": "i2i:a", "node_id": "i2t:a", "alignment": .1},
            {"source_task_id": "i2i:b", "node_id": "i2t:a", "alignment": .4}]}
        transfer = {"schema_version": 1, "input_audit": {"hash": "t"}, "transfer_matrix": [
            {"source_task_id": source, "node_id": target, "gain_mean_pp": gain}
            for source, target, gain in [("i2i:a", "i2t:a", 1), ("i2i:b", "i2t:a", 2),
                                        ("i2i:a", "i2t:b", 3), ("i2i:b", "i2t:b", 4)]]}
        report = associate(gradients, transfer, balanced_targets=["i2t:a"])
        self.assertEqual(len(report["cells"]), 4)
        self.assertIsNone(report["cells"][-1]["alignment"])
        self.assertAlmostEqual(report["correlations"]["balanced7"]["pearson"], 1)
        self.assertEqual(report["capability_means"][0]["mean_transfer_gain_pp"], 1.5)

    def test_parameter_families_keep_weight_and_bias_separate(self):
        self.assertEqual(classify_parameter("language_model.model.layers.0.input_layernorm.weight")["module"],
                         "llm.input_ln.weight")
        self.assertIsNone(classify_parameter("language_model.model.layers.0.input_layernorm_moe_gen.weight"))
        self.assertNotEqual(classify_parameter("connector.fc1.weight")["module"],
                            classify_parameter("connector.fc1.bias")["module"])
        self.assertTrue(classify_parameter("language_model.model.layers.27.self_attn.q_proj.weight")["structural_zero_i2i"])

    def test_target_import_authenticates_images_and_preserves_evaluation_prompt(self):
        taxonomy = Path(__file__).resolve().parents[1] / "data/taxonomy"
        retained = next(read_jsonl(taxonomy / "retained_questions.jsonl"))
        path = self.root / "image.png"
        rgb = Image.new("RGB", (10, 12), "red")
        rgb.save(path)
        record = {"uid": retained["uid"], "category_id": retained["task_id"], "fold": 0,
                  "prompt": "Question: Which?\nOptions:\nA. red\nPlease choose. \n", "options": {"A": "red"}, "answer": "A",
                  "images": [{"path": str(path), "sha256": sha256(path),
                              "rgb_sha256": hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()}]}
        source = self.root / "benchmark.json"
        source.write_text(json.dumps({"sampling_seed": 2026091503, "image_transform": {"max_image_size": 980},
                                     "categories": [{"id": retained["task_id"], "samples": [record]}]}))
        config = prepare_targets(source, taxonomy, self.root / "targets")
        group = json.loads(config.read_text())["groups"][0]
        result = next(read_jsonl(config.parent / group["manifest"]))
        self.assertEqual(result["gradient_mcq"]["prompt"], record["prompt"])
        self.assertEqual(result["images"], ["../image.png"])
        Image.new("RGB", (10, 12), "blue").save(path)
        with self.assertRaisesRegex(ValueError, "image hash differs"):
            prepare_targets(source, taxonomy, self.root / "tampered")


if __name__ == "__main__":
    unittest.main()
