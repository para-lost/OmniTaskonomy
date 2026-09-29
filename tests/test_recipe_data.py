import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from omnitaskonomy.data.common import read_jsonl, resolve_image, sha256
from omnitaskonomy.data import recipe


class RecipeDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self.subsets = list(dict.fromkeys(recipe.TASKS.values()))
        self.rows = {}
        subsets = []
        for task in self.subsets:
            splits = []
            for split in ("train", "val"):
                rows = [self.record(task, split, index) for index in (7, 2, 10)]
                self.rows[task, split] = rows
                name = f"{task}-{split}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), self.hub / name, row_group_size=1)
                splits.append({"name": split, "num_examples": len(rows), "files": [{
                    "path": name, "bytes": (self.hub / name).stat().st_size,
                    "sha256": sha256(self.hub / name),
                }]})
            subsets.append({"name": task, "splits": splits})
        self.catalog = {"format_version": 1, "dataset": recipe.REPO_ID, "subsets": subsets}
        self.save_catalog()
        self.downloads = []
        self.download = patch("huggingface_hub.hf_hub_download", side_effect=self.fetch)
        self.download.start()
        self.addCleanup(self.download.stop)

    @staticmethod
    def picture(color, size, format):
        stream = io.BytesIO()
        Image.new("RGB", size, color).save(stream, format=format)
        return {"bytes": stream.getvalue(), "path": None}

    def record(self, task, split, index):
        target = self.picture((index, 150, 220), (32, 32), "JPEG")
        return {"id": f"{task}/{split}/{index}", "task": task, "split": split,
                "source_id": str(index), "source_dataset": "fixture",
                "i2i_input_image": self.picture((60, index, 80), (32, 32), "PNG"),
                "i2i_output_image": target, "i2t_input_image": target,
                "i2i_prompt": " Restore the image.\n\n",
                "i2t_prompt": "Choose the correct order.\n\n", "i2t_answer": "Option B.",
                "metadata": json.dumps({"episode": index, "original_index": index, "correct_choice": "B"})}

    def save_catalog(self):
        (self.hub / "release_manifest.json").write_text(json.dumps(self.catalog))

    def fetch(self, repo, filename, *, repo_type, revision):
        self.assertEqual((repo, repo_type, revision),
                         (recipe.REPO_ID, "dataset", recipe.RELEASE_REVISION))
        self.downloads.append(filename)
        return str(self.hub / filename)

    def paths(self, task="jigsaw", split="train"):
        return {kind: self.root / "prepared" / task / split / f"{kind}.jsonl"
                for kind in ("i2i", "i2t")}

    def test_all_subsets_and_splits_preserve_content_in_the_required_order(self):
        for task in self.subsets:
            for split in ("train", "val"):
                with self.subTest(task=task, split=split):
                    paths = recipe.prepare_recipe(task, self.paths(task, split), split=split)
                    image_fields = {"i2i": {"source_image": "i2i_input_image", "target_image": "i2i_output_image"},
                                    "i2t": {"image": "i2t_input_image"}}
                    for kind, path in paths.items():
                        actual = list(read_jsonl(path))
                        episode_order = [10, 2, 7] if task in {"jigsaw", "zoomin"} and split == "train" else [7, 2, 10]
                        sources = {row["source_id"]: row for row in self.rows[task, split]}
                        expected = [sources[str(episode)] for episode in episode_order]
                        self.assertEqual([row["uid"] for row in actual], [row["id"] for row in expected])
                        for row, source in zip(actual, expected):
                            self.assertEqual(row["source_uid"], source["id"])
                            self.assertEqual(row["metadata"], json.loads(source["metadata"]))
                            self.assertEqual(row["split"], split)
                            for key, field in image_fields[kind].items():
                                self.assertFalse(Path(row[key]).is_absolute())
                                image = resolve_image(path, row[key])
                                self.assertEqual(image.read_bytes(), source[field]["bytes"])
                                with Image.open(image) as decoded:
                                    decoded.load()
                            if kind == "i2i":
                                self.assertEqual(row["prompt"], source["i2i_prompt"])
                            else:
                                self.assertEqual(row["conversations"], [
                                    {"from": "human", "value": source["i2t_prompt"]},
                                    {"from": "gpt", "value": source["i2t_answer"]},
                                ])
                        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
                        self.assertEqual(metadata["revision"], recipe.RELEASE_REVISION)
                        self.assertEqual(metadata["count"], len(expected))
                        self.assertEqual(metadata["manifest_sha256"], sha256(path))
                        self.assertEqual(metadata["record_order"],
                                         "episode_lexicographic" if task in {"jigsaw", "zoomin"} and split == "train" else "release")
                    # Identical targets and I2T inputs share one file without re-encoding.
                    generation = next(read_jsonl(paths["i2i"]))
                    understanding = next(read_jsonl(paths["i2t"]))
                    self.assertEqual(generation["target_image"], understanding["image"])

    def test_existing_custom_manifests_and_alias_caches_take_precedence(self):
        for task, subset in (("jigsaw", "jigsaw"), ("zoomin", "zoomin"),
                             ("video_unshuffle_3d", "video_unshuffle"), ("colorization", "visgym_colorization")):
            paths = self.paths(task)
            paths["i2t"].parent.mkdir(parents=True)
            paths["i2t"].write_text('existing custom manifest\n')
            recipe.prepare_recipe(task, paths)
            self.assertEqual(next(read_jsonl(paths["i2i"]))["task"], subset)
            self.assertEqual(paths["i2t"].read_text(), 'existing custom manifest\n')
            with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected network")):
                recipe.prepare_recipe(task, paths)

    def test_incomplete_or_corrupt_release_does_not_publish_manifests(self):
        paths = self.paths()
        spec = self.catalog["subsets"][0]["splits"][0]
        spec["num_examples"] += 1
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            recipe.prepare_recipe("jigsaw", paths)
        self.assertFalse(any(path.exists() for path in paths.values()))
        spec["num_examples"] -= 1
        self.save_catalog()
        shard = self.hub / spec["files"][0]["path"]
        original = shard.read_bytes()
        shard.write_bytes(original[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            recipe.prepare_recipe("jigsaw", paths)
        self.assertFalse(any(path.exists() for path in paths.values()))
        shard.write_bytes(original)
        recipe.prepare_recipe("jigsaw", paths)
        self.assertEqual(len(list(read_jsonl(paths["i2t"]))), 3)

    def test_duplicate_rows_and_wrong_split_fail_before_publication(self):
        for invalid in ("duplicate", "split"):
            rows = [dict(row) for row in self.rows["jigsaw", "train"]]
            if invalid == "duplicate":
                rows[1]["id"] = rows[0]["id"]
            else:
                rows[1]["split"] = "val"
            spec = self.catalog["subsets"][0]["splits"][0]["files"][0]
            shard = self.hub / spec["path"]
            pq.write_table(pa.Table.from_pylist(rows), shard)
            spec.update(bytes=shard.stat().st_size, sha256=sha256(shard))
            self.save_catalog()
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                recipe.prepare_recipe("jigsaw", self.paths())
            self.assertFalse(self.paths()["i2i"].exists())

    def test_bagel_and_custom_pool_consume_the_same_seeded_rows(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Bagel"))
        from data.transforms import ImageTransform
        from omnitaskonomy.datasets import ManifestDataset
        from omnitaskonomy.umm_training import _pool

        class Tokenizer:
            def encode(self, text):
                return list(text.encode())

        transform = ImageTransform(28, 28, 14)
        for task in ("jigsaw", "zoomin"):
            paths = recipe.prepare_recipe(task, self.paths(task))
            for kind, path in paths.items():
                with self.subTest(task=task, kind=kind):
                    dataset = ManifestDataset(kind, path, kind, transform, Tokenizer(), transform,
                                              num_used_data=2, data_seed=42, shuffle_before_slice=True)
                    pool = _pool({"manifest": path, "num_used_data": 2, "data_seed": 42,
                                  "shuffle_before_slice": True})
                    expected = [f"{task}/train/2", f"{task}/train/10"]
                    self.assertEqual([row["uid"] for row in pool], expected)
                    self.assertEqual([dataset.rows[i]["uid"] for i in dataset.indices], expected)
                    parsed = dataset.parse_row(pool[0])
                    self.assertEqual(sum(step["loss"] for step in parsed["sequence_plan"]), 1)
                    if kind == "i2t":
                        self.assertEqual(parsed["text_ids_list"][0], list(self.rows[task, "train"][0]["i2t_prompt"].encode()))
                        self.assertEqual(parsed["text_ids_list"][1], list("Option B.".encode()))


if __name__ == "__main__":
    unittest.main()
