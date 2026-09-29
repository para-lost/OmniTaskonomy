import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from omnitaskonomy.data.common import read_jsonl, resolve_image
from omnitaskonomy.gradients.artifacts import stable_seed
from omnitaskonomy.gradients.manifest import freeze, verify_manifest
from omnitaskonomy.gradients.prepare_matrix import IMAGE_TRANSFORM, TARGET_SEED, _target_rank, prepare_matrix
from omnitaskonomy.taxonomy import load_taxonomy


class GradientMatrixDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "gradients/transfer.json"
        tasks = load_taxonomy()["tasks"]
        self.taxonomy = {"tasks": tasks, "retained": []}
        self.rows, jobs = {}, []
        for number, task in enumerate(tasks):
            task_id = task["id"]
            rows = []
            for index in range(7):
                uid = f"{task_id}:{index}"
                images = []
                for view in range(2):
                    image = Image.new("RGB", (4, 4), (number, index, view))
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    images.append({"bytes": stream.getvalue(), "path": None})
                if task["modality"] == "i2i":
                    directory = self.root / task["source_id"]
                    directory.mkdir(exist_ok=True)
                    for view, payload in enumerate(images):
                        (directory / f"{index}_{view}.png").write_bytes(payload["bytes"])
                    rows.append({"uid": uid, "source_image": f"{index}_0.png", "target_image": f"{index}_1.png",
                                 "prompt": "Draw.", "fold": 4})
                else:
                    self.taxonomy["retained"].append({"uid": uid, "task_id": task_id})
                    rows.append({"id": uid, "task_id": task_id, "modality": "i2t", "usage": "val",
                                 "input_images": images, "evaluation_prompt": "Native prompt.\nA. first\nB. second \n",
                                 "options": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
                                 "answer": "B"})
            if task["modality"] == "i2i":
                manifest = directory / "train.jsonl"
                manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
                training = {"i2i_manifest": str(manifest.relative_to(self.root)),
                            "i2i_target_interpolation": "nearest" if task_id == "i2i:semantic_segmentation" else "bicubic"}
                jobs.extend({"id": f"{task_id}:{stage}", "task_id": task_id, "training": training}
                            for stage in (1, 2))
            else:
                self.rows[task_id] = rows
        self.config = self.root / "experiment.json"
        self.config.write_text(json.dumps({"jobs": jobs}))
        self.taxonomy_patch = patch("omnitaskonomy.gradients.prepare_matrix.load_taxonomy", return_value=self.taxonomy)
        self.taxonomy_patch.start()
        self.addCleanup(self.taxonomy_patch.stop)
        self.release_patch = patch("omnitaskonomy.gradients.prepare_matrix.iter_release_rows",
                                   side_effect=lambda task, **kwargs: iter(self.rows[task]))
        self.release = self.release_patch.start()
        self.addCleanup(self.release_patch.stop)

    def prepare(self):
        return prepare_matrix(self.root, self.output, experiment_config=self.config, sample_count=5)

    def test_public_pools_feed_existing_freeze_without_legacy_inputs(self):
        result = json.loads(self.prepare().read_text())
        self.assertEqual(len(result["groups"]), 44)
        self.assertEqual(len({group["task"] for group in result["groups"]}), 44)
        self.assertEqual(self.release.call_count, 25)
        self.assertIn("not the historical", result["provenance"]["selection"])
        for group in result["groups"]:
            manifest = self.output.parent / group["manifest"]
            records = list(read_jsonl(manifest))
            self.assertEqual(len(records), 5)
            self.assertTrue(all("fold" not in row for row in records))
            if group["objective"] == "i2i":
                self.assertTrue(all(resolve_image(manifest, row["source_image"]).is_file() for row in records))
                if group["task"] == "i2i:semantic_segmentation":
                    self.assertEqual(group["target_interpolation"], "nearest")
                continue
            self.assertEqual(group["vit_transform"], IMAGE_TRANSFORM)
            original = {row["id"]: row for row in self.rows[group["task"]]}
            self.assertEqual([row["uid"] for row in records], sorted(original, key=_target_rank)[:5])
            for record in records:
                source = original[record["uid"]]
                self.assertEqual(record["gradient_mcq"], {"prompt": source["evaluation_prompt"], "answer": source["answer"]})
                self.assertEqual(record["loss_seed"], stable_seed(TARGET_SEED, record["uid"]) % (2**32 - 1))
                self.assertEqual([resolve_image(manifest, path).read_bytes() for path in record["images"]],
                                 [value["bytes"] for value in source["input_images"]])
        frozen = self.root / "frozen.json"
        freeze(self.output, frozen)
        rows = verify_manifest(frozen)["rows"]
        self.assertEqual(len(rows), 220)
        for group in result["groups"]:
            self.assertEqual({row["fold"] for row in rows if row["task"] == group["task"]}, set(range(5)))

    def test_missing_source_fails_before_download_or_output(self):
        first = json.loads(self.config.read_text())["jobs"][0]
        (self.root / first["training"]["i2i_manifest"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Prepare the transfer and Taskonomy pools"):
            self.prepare()
        self.release.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_incomplete_public_target_does_not_publish_configuration(self):
        first = next(iter(self.rows))
        removed = self.rows[first].pop()
        with self.assertRaisesRegex(ValueError, "missing retained questions"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.rows[first].append(removed)
        self.assertTrue(self.prepare().is_file())


if __name__ == "__main__":
    unittest.main()
