import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from omnitaskonomy.data import transfer
from omnitaskonomy.data.common import read_jsonl, resolve_image, sha256


class TransferDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self.rows = {}
        leaves = []
        for task, task_id in transfer.TASKS.items():
            rows = [self.record(task_id, index) for index in (2, 0, 1)]
            self.rows[task] = rows
            files = []
            for index, portion in enumerate((rows[:1], rows[1:])):
                filename = f"{task}-{index}.parquet"
                path = self.hub / filename
                pq.write_table(pa.Table.from_pylist(portion), path)
                files.append({"path": filename, "rows": len(portion),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
            leaves.append({"task_id": task_id, "modality": "i2i", "usage": "train",
                           "rows": len(rows), "files": files})
        self.catalog = {"dataset": transfer.REPO_ID, "leaves": leaves}
        self.save_catalog()
        self.downloads = []
        download = patch("huggingface_hub.hf_hub_download", side_effect=self.fetch)
        download.start()
        self.addCleanup(download.stop)

    @staticmethod
    def picture(index):
        stream = io.BytesIO()
        Image.new("RGB", (16, 16), (index, 10, 50)).save(stream, format="PNG")
        return {"bytes": stream.getvalue(), "path": None}

    def record(self, task_id, index):
        return {"id": f"{task_id}:occurrence{index}", "task_id": task_id,
                "modality": "i2i", "usage": "train", "source_id": str(index % 2),
                "source_dataset": "fixture", "repeat_index": index // 2,
                "input_images": [self.picture(index)], "output_image": self.picture(index + 20),
                "editing_prompt": " Restore the image.\n\n",
                "source_metadata": json.dumps({"episode": index})}

    def save_catalog(self):
        (self.hub / "release_manifest.json").write_text(json.dumps(self.catalog))

    def fetch(self, repo, filename, *, repo_type, revision):
        expected = (transfer.LLAVA_REPO_ID, transfer.LLAVA_REVISION) if filename.endswith(".json") and filename != "release_manifest.json" else (transfer.REPO_ID, transfer.RELEASE_REVISION)
        self.assertEqual((repo, revision), expected)
        self.assertEqual(repo_type, "dataset")
        self.downloads.append(filename)
        return str(self.hub / filename)

    def test_all_public_pools_preserve_release_order_repetitions_prompts_and_bytes(self):
        for task in transfer.TASKS:
            with self.subTest(task=task):
                output = self.root / "prepared" / task / "train.jsonl"
                self.assertEqual(transfer.prepare_transfer_task(task, output), output)
                actual = list(read_jsonl(output))
                self.assertEqual([r["uid"] for r in actual], [r["id"] for r in self.rows[task]])
                self.assertEqual([r["source_id"] for r in actual], ["0", "0", "1"])
                for row, source in zip(actual, self.rows[task]):
                    self.assertEqual(row["prompt"], source["editing_prompt"])
                    self.assertEqual(row["repeat_index"], source["repeat_index"])
                    self.assertEqual(row["metadata"], json.loads(source["source_metadata"]))
                    for field, picture in (("source_image", source["input_images"][0]),
                                           ("target_image", source["output_image"])):
                        self.assertFalse(Path(row[field]).is_absolute())
                        self.assertEqual(resolve_image(output, row[field]).read_bytes(), picture["bytes"])
                metadata = json.loads(output.with_suffix(".metadata.json").read_text())
                self.assertEqual(metadata["count"], 3)
                self.assertEqual(metadata["manifest_sha256"], sha256(output))
                self.assertEqual(metadata["revision"], transfer.RELEASE_REVISION)
                with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")):
                    transfer.prepare_transfer_task(task, output)

    def test_projected_columns_retain_identity_without_loading_images(self):
        rows = list(transfer.iter_release_rows(transfer.TASKS["jigsaw"], columns=["source_id"]))
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows[0]), {"id", "task_id", "modality", "usage", "source_id"})

    def test_invalid_release_cannot_publish_a_training_manifest(self):
        leaf = self.catalog["leaves"][0]
        task = next(iter(transfer.TASKS))
        output = self.root / "prepared" / task / "train.jsonl"
        leaf["rows"] += 1
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            transfer.prepare_transfer_task(task, output)
        self.assertFalse(output.exists())
        leaf["rows"] -= 1
        leaf["files"][0]["sha256"] = "wrong"
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            transfer.prepare_transfer_task(task, output)
        self.assertFalse(output.exists())

    def test_duplicate_occurrence_ids_are_rejected(self):
        task = next(iter(transfer.TASKS))
        leaf = self.catalog["leaves"][0]
        rows = [dict(row) for row in self.rows[task][1:]]
        rows[0]["id"] = self.rows[task][0]["id"]
        shard = self.hub / leaf["files"][1]["path"]
        pq.write_table(pa.Table.from_pylist(rows), shard)
        leaf["files"][1].update(bytes=shard.stat().st_size, sha256=sha256(shard))
        self.save_catalog()
        output = self.root / "duplicate.jsonl"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            transfer.prepare_transfer_task(task, output)
        self.assertFalse(output.exists())

    def llava_fixture(self):
        data = [{"id": str(index % 3), "image": f"{index:012d}.jpg",
                 "conversations": [{"from": "human", "value": "<image>\nDescribe it. "},
                                   {"from": "gpt", "value": f" Answer {index}.\n"}]}
                for index in range(11)]
        source = self.hub / "llava_instruct_150k.json"
        source.write_text(json.dumps(data))
        return source, data

    def test_llava_matches_historical_double_shuffle_and_keeps_all_conversation_turns(self):
        source, data = self.llava_fixture()
        coco = self.root / "coco" / "train2017"
        coco.mkdir(parents=True)
        for row in data:
            (coco / row["image"]).write_bytes(self.picture(10)["bytes"])
        expected = list(range(len(data)))
        random.Random(0).shuffle(expected)
        expected = expected[:6]
        random.Random(0).shuffle(expected)
        output = self.root / "prepared" / "llava" / "train.jsonl"
        with patch.object(transfer, "LLAVA_COUNT", 6), patch.object(transfer, "urlopen", side_effect=AssertionError("network")):
            transfer.prepare_llava(output, llava_json=source, coco_root=coco.parent)
        rows = list(read_jsonl(output))
        self.assertEqual([row["uid"] for row in rows], [f"llava:{index}" for index in expected])
        self.assertEqual(len({row["uid"] for row in rows}), 6)
        for row, index in zip(rows, expected):
            self.assertEqual(row["conversations"], data[index]["conversations"])
            self.assertEqual(resolve_image(output, row["image"]).resolve(), coco / data[index]["image"])
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["source_sha256"], sha256(source))
        self.assertEqual(metadata["count"], 6)

    def test_missing_llava_sources_download_annotation_and_only_extract_selected_coco_images(self):
        _, data = self.llava_fixture()
        archive = io.BytesIO()
        with ZipFile(archive, "w") as zipped:
            for row in data:
                zipped.writestr("train2017/" + row["image"], self.picture(10)["bytes"])
        coco = self.root / "coco"
        output = self.root / "prepared" / "llava" / "train.jsonl"
        with patch.object(transfer, "LLAVA_COUNT", 6), patch.object(transfer, "urlopen", return_value=io.BytesIO(archive.getvalue())) as download:
            transfer.prepare_llava(output, coco_root=coco)
        download.assert_called_once_with(transfer.COCO_TRAIN_URL, timeout=120)
        self.assertIn("llava_instruct_150k.json", self.downloads)
        rows = list(read_jsonl(output))
        self.assertEqual({p.name for p in (coco / "train2017").iterdir()},
                         {Path(row["image"]).name for row in rows})
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["revision"], transfer.LLAVA_REVISION)
        with patch.object(transfer, "urlopen", side_effect=AssertionError("network")), \
                patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")):
            transfer.prepare_llava(output, coco_root=coco)

    def test_short_llava_source_does_not_publish_manifest_or_download_images(self):
        source, _ = self.llava_fixture()
        output = self.root / "prepared" / "llava" / "train.jsonl"
        with patch.object(transfer, "urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "at least 50000"):
                transfer.prepare_llava(output, llava_json=source, coco_root=self.root / "coco")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
