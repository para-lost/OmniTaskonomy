"""Prepare the published I2I pools and the historical LLaVA 50k transfer pool."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import re
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

from omnitaskonomy.data.common import relative_image, sha256, write_manifest
from omnitaskonomy.data.recipe import _save_image
from omnitaskonomy.data.taskonomy import REPO_ID, RELEASE_REVISION


TASKS = {
    "object_editing": "i2i:object_replacement",
    "attribute_editing": "i2i:attribute_editing",
    "inpainting": "i2i:inpainting",
    "semantic_segmentation": "i2i:semantic_segmentation",
    "object_pointing": "i2i:object_pointing",
    "jigsaw": "i2i:jigsaw",
    "localization": "i2i:localization",
}
LLAVA_REPO_ID = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_REVISION = "9d451dc7629cfe0469f6ae4432b765cd603d5fcb"
LLAVA_COUNT = 50000
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"


def iter_release_rows(task_id, *, columns=None):
    """Read one frozen OmniTaskonomy leaf, retaining its published row order."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    catalog_path = hf_hub_download(REPO_ID, "release_manifest.json", repo_type="dataset",
                                   revision=RELEASE_REVISION)
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog["dataset"] != REPO_ID:
        raise ValueError("Unexpected OmniTaskonomy release manifest")
    spec = next((leaf for leaf in catalog["leaves"] if leaf["task_id"] == task_id), None)
    if spec is None:
        raise ValueError(f"Task is not included in the image release: {task_id}")
    identity = {"id", "task_id", "modality", "usage"}
    selected = None if columns is None else sorted(set(columns) | identity)
    seen = set()
    for item in spec["files"]:
        path = Path(hf_hub_download(REPO_ID, item["path"], repo_type="dataset",
                                    revision=RELEASE_REVISION))
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Release shard checksum mismatch: {item['path']}")
        count = 0
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=32, columns=selected):
                for row in batch.to_pylist():
                    if (row["task_id"] != task_id or row["modality"] != spec["modality"]
                            or row["usage"] != spec["usage"] or row["id"] in seen):
                        raise ValueError(f"Invalid or duplicate release record: {row['id']}")
                    seen.add(row["id"])
                    count += 1
                    yield row
        if count != item["rows"]:
            raise ValueError(f"Release shard count mismatch: {item['path']}")
    if len(seen) != spec["rows"]:
        raise ValueError(f"Release count mismatch: {task_id}")


@contextmanager
def _manifest_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare_transfer_task(task, output):
    """Cache one public 50k I2I pool without reshuffling or re-encoding images."""
    if task not in TASKS:
        raise ValueError(f"Unknown public transfer task: {task}")
    output = Path(output).resolve()
    with _manifest_lock(output):
        if output.is_file():
            return output
        print(f"Preparing {REPO_ID}: {TASKS[task]}", flush=True)

        def records():
            for row in iter_release_rows(TASKS[task]):
                if len(row["input_images"]) != 1:
                    raise ValueError(f"I2I transfer needs one input image: {row['id']}")
                source = _save_image(row["input_images"][0], output.parent / "images")
                target = _save_image(row["output_image"], output.parent / "images")
                yield {"uid": row["id"], "source_uid": row["id"], "task": task,
                       "source_dataset": row["source_dataset"], "source_id": row["source_id"],
                       "split": "train", "repeat_index": row["repeat_index"],
                       "source_image": relative_image(source, output),
                       "target_image": relative_image(target, output), "prompt": row["editing_prompt"],
                       "metadata": json.loads(row["source_metadata"])}

        write_manifest(records(), output, {
            "repo_id": REPO_ID, "revision": RELEASE_REVISION,
            "task_id": TASKS[task], "record_order": "release", "objective": "i2i",
        })
    return output


def _ensure_coco_images(names, coco_root):
    missing = {name for name in names if not (coco_root / "train2017" / name).is_file()}
    if not missing:
        return
    coco_root.mkdir(parents=True, exist_ok=True)
    archive = coco_root / "train2017.zip"
    if not archive.is_file():
        print(f"Downloading COCO train2017 archive to {archive}", flush=True)
        temporary = archive.with_suffix(".zip.part")
        with urlopen(COCO_TRAIN_URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        temporary.replace(archive)
    directory = coco_root / "train2017"
    directory.mkdir(exist_ok=True)
    with ZipFile(archive) as source:
        for name in sorted(missing):
            destination = directory / name
            temporary = destination.with_suffix(".jpg.part")
            with source.open(f"train2017/{name}") as image, temporary.open("wb") as stream:
                shutil.copyfileobj(image, stream)
            temporary.replace(destination)


def prepare_llava(output, *, llava_json=None, coco_root="data/raw/coco"):
    """Match OneVisionJSONIterableDataset's seed-0 shuffle/slice/shuffle pool."""
    output = Path(output).resolve()
    coco_root = Path(coco_root).resolve()
    with _manifest_lock(output):
        if output.is_file():
            return output
        downloaded = llava_json is None
        if downloaded:
            from huggingface_hub import hf_hub_download

            llava_json = hf_hub_download(LLAVA_REPO_ID, "llava_instruct_150k.json",
                                         repo_type="dataset", revision=LLAVA_REVISION)
        source = Path(llava_json).resolve()
        data = json.loads(source.read_text())
        if not isinstance(data, list) or len(data) < LLAVA_COUNT:
            raise ValueError(f"LLaVA source needs at least {LLAVA_COUNT} examples")
        selected = list(enumerate(data))
        random.Random(0).shuffle(selected)
        selected = selected[:LLAVA_COUNT]
        # The historical loader re-seeds before its second, post-selection shuffle.
        random.Random(0).shuffle(selected)
        names = []
        for index, row in selected:
            name = Path(row["image"]).name
            if not re.fullmatch(r"\d{12}\.jpg", name):
                raise ValueError(f"Unexpected COCO train2017 image in LLaVA row {index}: {row['image']}")
            conversations = row["conversations"]
            if (not conversations or conversations[0]["from"] != "human"
                    or not any(turn["from"] == "gpt" for turn in conversations)):
                raise ValueError(f"Invalid LLaVA conversation in row {index}")
            names.append(name)
        _ensure_coco_images(names, coco_root)
        rows = ({"uid": f"llava:{index}", "source_uid": f"llava:{index}",
                 "source_id": str(row["id"]), "source_dataset": LLAVA_REPO_ID,
                 "image": relative_image(coco_root / "train2017" / name, output),
                 "conversations": row["conversations"]}
                for (index, row), name in zip(selected, names))
        write_manifest(rows, output, {
            "repo_id": LLAVA_REPO_ID, "revision": LLAVA_REVISION if downloaded else None,
            "source_json": str(source), "source_sha256": sha256(source),
            "record_order": "shuffle_seed0_take50000_shuffle_seed0", "objective": "i2t",
        })
    return output
