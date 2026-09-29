"""Freeze sample selection, source pairing, image identities, and gradient folds."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from omnitaskonomy.data.common import read_jsonl, resolve_image, sha256
from omnitaskonomy.gradients.artifacts import validate_rows, write_json

REFERENCE_TASKS = {"jigsaw", "zoomin", "colorization", "video_unshuffle_3d", "counting", "rotate_qa"}


def _rank(seed, identity):
    return hashlib.sha256(f"{seed}\0{identity}".encode()).digest()


def freeze(config_path, output, *, reference=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1 or not config["groups"]:
        raise ValueError("Expected gradient configuration schema_version 1 and nonempty groups")
    folds, count = config.get("folds", 5), config.get("samples_per_group", 500)
    if type(folds) is not int or folds < 2 or type(count) is not int or count < folds:
        raise ValueError("Invalid fold or sample count")
    rows, inputs, groups, image_assets = [], [], set(), {}
    for spec in config["groups"]:
        task, objective = spec["task"], spec["objective"]
        group_count = spec.get("samples_per_group", count)
        if (task, objective) in groups:
            raise ValueError(f"Duplicate gradient group: {task}/{objective}")
        groups.add((task, objective))
        manifest = (config_path.parent / spec["manifest"]).resolve()
        records = list(read_jsonl(manifest))
        identities = [row["uid"] for row in records]
        if type(group_count) is not int or group_count < folds or len(set(identities)) != len(identities) or len(records) < group_count:
            raise ValueError(f"Duplicate UIDs or insufficient samples: {manifest}")
        order = config.get("selection_order", "seeded_hash")
        if order == "manifest":
            if len(records) != group_count:
                raise ValueError("Frozen manifest order requires exactly samples_per_group records")
            selected = records
        elif order == "seeded_hash":
            selected = sorted(records, key=lambda row: (_rank(config.get("sample_seed", 1000), row["uid"]), row["uid"]))[:group_count]
        else:
            raise ValueError("selection_order must be seeded_hash or manifest")
        inputs.append({"manifest": os.path.relpath(manifest, output.parent), "sha256": sha256(manifest)})
        for record in selected:
            image_paths = ([record["source_image"], record["target_image"]] if objective == "i2i"
                           else record.get("images", [record.get("image")]))
            assets = []
            for value in image_paths:
                image_path = resolve_image(manifest, value).resolve()
                if image_path not in image_assets:
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        decoded = hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()
                    image_assets[image_path] = {"path": os.path.relpath(image_path, output.parent),
                                               "sha256": sha256(image_path), "decoded_sha256": decoded}
                assets.append(image_assets[image_path])
            rows.append({"task": task, "objective": objective, "uid": record["uid"],
                         "sampling_key": spec.get("sampling_key", f"{task}.{objective}"),
                         "source_uid": record.get("source_uid", f"{task}:{record['uid']}"),
                         "fold": record.get("fold"), "image_hashes": [a["decoded_sha256"] for a in assets],
                         "manifest": os.path.relpath(manifest, output.parent), "record": record,
                         "target_interpolation": spec.get("target_interpolation", "bicubic"),
                         "vit_transform": spec.get("vit_transform"),
                         "data_seed": record.get("data_seed"), "loss_seed": record.get("loss_seed"),
                         "assets": assets})

    # Source identities bind paired renders; decoded-image clusters bind repeated
    # benchmark images, even when those files use different names or encodings.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for row in rows:
        source = "source:" + row["source_uid"]
        find(source)
        if row["objective"] == "i2t":
            for digest in row["image_hashes"]:
                parents[find("image:" + digest)] = find(source)
    clusters = defaultdict(list)
    for row in rows:
        clusters[find("source:" + row["source_uid"])].append(row)
    locked = defaultdict(set)
    reference_path = reference if reference is not None else config.get("reference")
    if reference_path:
        reference_rows = json.loads((config_path.parent / reference_path).read_text())["rows"]
        for row in reference_rows:
            locked["source:" + row["source_uid"]].add(row["fold"])
            for digest in row["image_hashes"]:
                locked["image:" + digest].add(row["fold"])
    cluster_locks = defaultdict(set)
    for key, required_folds in locked.items():
        if key in parents:
            cluster_locks[find(key)].update(required_folds)
    for key, members in clusters.items():
        cluster_locks[key].update(row["fold"] for row in members if row["fold"] is not None)
        if len(cluster_locks[key]) > 1:
            raise ValueError("Explicit/reference folds split a source or decoded-image cluster")
    by_task = defaultdict(list)
    for key, members in clusters.items():
        by_task[min(row["task"] for row in members)].append(key)
    for task, keys in by_task.items():
        for rank, key in enumerate(sorted(keys, key=lambda key: (_rank(config.get("fold_seed", 20260826), key), key))):
            fold = next(iter(cluster_locks[key])) if cluster_locks[key] else rank % folds
            for row in clusters[key]:
                row["fold"] = fold
    validate_rows(rows, paired=config.get("paired_reference", False), folds=folds)
    if config.get("paper_reference", False):
        if groups != {(task, objective) for task in REFERENCE_TASKS for objective in ("i2i", "i2t")}:
            raise ValueError("Paper reference requires all six paired tasks")
        if folds != 5 or count != 500:
            raise ValueError("Paper reference requires 500 pairs and five folds")
        for task, objective in groups:
            if any(sum(row["task"] == task and row["objective"] == objective and row["fold"] == f
                       for row in rows) != 100 for f in range(5)):
                raise ValueError(f"Paper reference must have 100 pairs per fold: {task}/{objective}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    result = {"schema_version": 1, "folds": folds, "sample_seed": config.get("sample_seed", 1000),
              "fold_seed": config.get("fold_seed", 20260826), "rows": rows, "inputs": inputs,
              "paper_reference": config.get("paper_reference", False),
              "config_sha256": sha256(config_path), "paper_commit": "5b8799e"}
    if reference is not None:
        result["reference"] = {"path": os.path.relpath(config_path.parent / reference, output.parent),
                               "sha256": sha256(config_path.parent / reference)}
    write_json(output, result)
    return result


def verify_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value["schema_version"] != 1:
        raise ValueError("Unsupported frozen gradient manifest")
    validate_rows(value["rows"], folds=value["folds"])
    for item in value["inputs"]:
        if sha256(path.parent / item["manifest"]) != item["sha256"]:
            raise ValueError("Source manifest changed after gradient selection")
    checked = set()
    for row in value["rows"]:
        for asset in row["assets"]:
            if asset["path"] not in checked and sha256(path.parent / asset["path"]) != asset["sha256"]:
                raise ValueError(f"Gradient image changed: {asset['path']}")
            checked.add(asset["path"])
    return value
