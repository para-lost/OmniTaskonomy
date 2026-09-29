"""Prepare cross-task gradient pools from training manifests and the public benchmark."""

from collections import defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path

from omnitaskonomy.data.common import read_jsonl, relative_image, resolve_image, sha256, write_manifest
from omnitaskonomy.data.recipe import _save_image
from omnitaskonomy.data.transfer import REPO_ID, RELEASE_REVISION, iter_release_rows
from omnitaskonomy.gradients.artifacts import stable_seed, write_json
from omnitaskonomy.gradients.manifest import _rank
from omnitaskonomy.taxonomy import load_taxonomy


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SEED = 1000
TARGET_SEED = 2026091503
IMAGE_TRANSFORM = {"max_image_size": 980, "min_image_size": 378,
                   "image_stride": 14, "max_pixels": 2007040}


def _target_rank(uid):
    return hashlib.sha256(json.dumps([TARGET_SEED, uid], separators=(",", ":"),
                                     ensure_ascii=False).encode()).digest()


def _sources(config, data_root, task_ids):
    sources = {}
    for job in config["jobs"]:
        task = job.get("task_id")
        if task not in task_ids:
            continue
        training = {**config.get("training", {}), **job["training"]}
        spec = (data_root / training["i2i_manifest"], training.get("i2i_target_interpolation", "bicubic"))
        if task in sources and sources[task] != spec:
            raise ValueError(f"Conflicting transfer source manifests: {task}")
        sources[task] = spec
    if set(sources) != task_ids:
        raise ValueError(f"Transfer configuration is missing I2I tasks: {sorted(task_ids - sources.keys())}")
    missing = [str(path) for path, _ in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepare the transfer and Taskonomy pools first; missing manifests: " + ", ".join(missing))
    return sources


def _target_record(row, manifest):
    uid, answer, prompt = row["id"], row["answer"], row["evaluation_prompt"]
    labels = {item["label"] for item in row["options"]}
    if not isinstance(answer, str) or len(answer) != 1 or answer not in labels or not prompt:
        raise ValueError(f"Invalid public benchmark prompt or correct option: {uid}")
    images = [relative_image(_save_image(value, manifest.parent / "images"), manifest)
              for value in row["input_images"]]
    if not images:
        raise ValueError(f"Benchmark example has no images: {uid}")
    seed = stable_seed(TARGET_SEED, uid) % (2**32 - 1)
    return {"uid": uid, "source_uid": uid, "images": images, "data_seed": seed, "loss_seed": seed,
            "gradient_mcq": {"prompt": prompt, "answer": answer},
            "conversations": [{"from": "human", "value": "<image>\n" * len(images) + prompt},
                              {"from": "gpt", "value": answer}]}


def prepare_matrix(data_root, output, *, experiment_config=None, taxonomy_path=None, sample_count=500):
    if type(sample_count) is not int or sample_count < 5:
        raise ValueError("Gradient pools need at least five samples for five folds")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = Path(experiment_config or ROOT / "configs/experiments/transfer.json")
    taxonomy = load_taxonomy(taxonomy_path)
    source_ids = {task["id"] for task in taxonomy["tasks"] if task["modality"] == "i2i"}
    sources = _sources(json.loads(config_path.read_text()), data_root, source_ids)
    targets = defaultdict(list)
    for row in taxonomy["retained"]:
        targets[row["task_id"]].append(row["uid"])
    selected_sources = {}
    for task, (path, _) in sources.items():
        selected = heapq.nsmallest(sample_count, read_jsonl(path),
                                  key=lambda row: (_rank(SOURCE_SEED, row["uid"]), row["uid"]))
        if len(selected) != sample_count:
            raise ValueError(f"Insufficient gradient source examples: {task}/{len(selected)}")
        selected_sources[task] = selected
    if any(len(uids) < 5 for uids in targets.values()):
        raise ValueError("Every target needs at least five examples for five folds")

    directory = output.parent / (output.stem + "_data")
    directory.mkdir(parents=True, exist_ok=True)
    groups = []
    provenance = {"repo_id": REPO_ID, "revision": RELEASE_REVISION,
                  "source_sample_seed": SOURCE_SEED, "target_sample_seed": TARGET_SEED,
                  "selection": "Rebuilt from public release and prepared training pools; not the historical frozen selection",
                  "folds": "Assigned by the existing freeze step, with reference source and decoded-image constraints"}
    for task, (path, interpolation) in sources.items():
        manifest = directory / (task.replace(":", "_") + ".jsonl")
        records = []
        for original in selected_sources[task]:
            row = dict(original)
            row.pop("fold", None)
            for field in ("source_image", "target_image"):
                image = resolve_image(path, row[field])
                if not image.is_file():
                    raise FileNotFoundError(image)
                row[field] = relative_image(image, manifest)
            records.append(row)
        write_manifest(records, manifest, {"source_manifest_sha256": sha256(path), "sample_seed": SOURCE_SEED})
        groups.append({"task": task, "objective": "i2i", "manifest": os.path.relpath(manifest, output.parent),
                       "samples_per_group": len(records), "target_interpolation": interpolation})

    columns = ["id", "task_id", "modality", "usage", "input_images", "options", "answer", "evaluation_prompt"]
    for task, uids in targets.items():
        chosen = sorted(uids, key=lambda uid: (_target_rank(uid), uid))[:sample_count]
        wanted, population, records, seen = set(chosen), set(uids), {}, set()
        manifest = directory / (task.replace(":", "_") + ".jsonl")
        for row in iter_release_rows(task, columns=columns):
            uid = row["id"]
            if (uid not in population or uid in seen or row["task_id"] != task
                    or row["modality"] != "i2t" or row["usage"] != "val"):
                raise ValueError(f"Public benchmark differs from the released taxonomy: {uid}")
            seen.add(uid)
            if uid in wanted:
                records[uid] = _target_record(row, manifest)
        if seen != population:
            raise ValueError(f"Public benchmark is missing retained questions: {task}")
        write_manifest([records[uid] for uid in chosen], manifest, provenance)
        groups.append({"task": task, "objective": "i2t", "manifest": os.path.relpath(manifest, output.parent),
                       "samples_per_group": len(chosen), "vit_transform": IMAGE_TRANSFORM})
        print(f"Prepared gradient target {task}: {len(chosen)} questions", flush=True)
    write_json(output, {"schema_version": 1, "folds": 5, "samples_per_group": sample_count,
                        "selection_order": "manifest", "sample_seed": SOURCE_SEED,
                        "fold_seed": 2026091601, "groups": groups, "provenance": provenance})
    return output
