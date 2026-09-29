"""Read the paired recipe release and cache images for the training loaders."""

from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from PIL import Image

from omnitaskonomy.data.common import read_jsonl, relative_image, sha256, write_manifest


REPO_ID = "Wakals/OmniTaskonomy_Recipe_Data"
RELEASE_REVISION = "775f3ffa96bff7fbe204516e5b510e215b6e81a1"
TASKS = {
    "jigsaw": "jigsaw",
    "zoomin": "zoomin",
    "video_unshuffle": "video_unshuffle",
    "video_unshuffle_3d": "video_unshuffle",
    "rotate_qa": "rotate_qa",
    "counting": "counting",
    "visgym_colorization": "visgym_colorization",
    "colorization": "visgym_colorization",
}


def iter_recipe_rows(task, split="train"):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if task not in TASKS or split not in {"train", "val"}:
        raise ValueError(f"Unknown recipe task/split: {task}/{split}")
    subset = TASKS[task]
    catalog_path = hf_hub_download(REPO_ID, "release_manifest.json", repo_type="dataset",
                                   revision=RELEASE_REVISION)
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog["format_version"] != 1 or catalog["dataset"] != REPO_ID:
        raise ValueError("Unsupported recipe release manifest")
    config = next(item for item in catalog["subsets"] if item["name"] == subset)
    spec = next(item for item in config["splits"] if item["name"] == split)
    seen = set()
    for item in spec["files"]:
        path = Path(hf_hub_download(REPO_ID, item["path"], repo_type="dataset",
                                    revision=RELEASE_REVISION))
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Recipe shard checksum mismatch: {item['path']}")
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    if row["task"] != subset or row["split"] != split or row["id"] in seen:
                        raise ValueError(f"Invalid or duplicate recipe record: {row['id']}")
                    seen.add(row["id"])
                    yield row
    if len(seen) != spec["num_examples"]:
        raise ValueError(f"Recipe count mismatch: {subset}/{split}, found {len(seen)}")


def _save_image(value, directory):
    payload = value["bytes"]
    with Image.open(io.BytesIO(payload)) as image:
        suffix = {"JPEG": ".jpg", "PNG": ".png"}[image.format]
    digest = hashlib.sha256(payload).hexdigest()
    path = directory / digest[:2] / (digest + suffix)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
        Path(temporary.name).replace(path)
    return path


def _training_record(row, objective, manifest):
    record = {"uid": row["id"], "source_uid": row["id"], "task": row["task"],
              "split": row["split"], "source_id": row["source_id"],
              "source_dataset": row["source_dataset"], "metadata": json.loads(row["metadata"])}
    fields = {"source_image": "i2i_input_image", "target_image": "i2i_output_image"} if objective == "i2i" else {"image": "i2t_input_image"}
    for destination, source in fields.items():
        image = _save_image(row[source], manifest.parent / "images")
        record[destination] = relative_image(image, manifest)
    if objective == "i2i":
        record["prompt"] = row["i2i_prompt"]
    else:
        record["conversations"] = [{"from": "human", "value": row["i2t_prompt"]},
                                   {"from": "gpt", "value": row["i2t_answer"]}]
    return record


def prepare_recipe(task, manifests, *, split="train"):
    """Cache recipe images and manifests in the order used by the training loaders."""
    if task not in TASKS or split not in {"train", "val"}:
        raise ValueError(f"Unknown recipe task/split: {task}/{split}")
    if not manifests or set(manifests) - {"i2i", "i2t"}:
        raise ValueError("Recipe manifests must specify i2i and/or i2t")
    manifests = {kind: Path(path).resolve() for kind, path in manifests.items()}
    if len(set(manifests.values())) != len(manifests):
        raise ValueError("I2I and I2T need separate manifest paths")
    with ExitStack() as locks:
        for path in sorted(manifests.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = locks.enter_context(path.with_suffix(".lock").open("w"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        missing = {kind: path for kind, path in manifests.items() if not path.is_file()}
        if not missing:
            return manifests
        print(f"Preparing {REPO_ID}: {TASKS[task]}/{split}", flush=True)
        with TemporaryDirectory(prefix=".recipe-", dir=next(iter(missing.values())).parent) as temporary:
            staged = {kind: Path(temporary) / f"{kind}.jsonl" for kind in missing}
            with ExitStack() as streams:
                handles = {kind: streams.enter_context(path.open("w", encoding="utf-8"))
                           for kind, path in staged.items()}
                for row in iter_recipe_rows(task, split):
                    for kind, path in missing.items():
                        record = _training_record(row, kind, path)
                        handles[kind].write(json.dumps(record, ensure_ascii=False) + "\n")
            for kind, path in missing.items():
                rows = read_jsonl(staged[kind])
                order = "release"
                if split == "train" and task in {"jigsaw", "zoomin"}:
                    # Seeded pool selection depends on the original episode ordering.
                    rows = sorted(rows, key=lambda row: str(row["metadata"]["episode"]))
                    order = "episode_lexicographic"
                write_manifest(rows, path, {
                    "repo_id": REPO_ID, "revision": RELEASE_REVISION, "subset": TASKS[task],
                    "split": split, "objective": kind, "record_order": order,
                })
    return manifests
