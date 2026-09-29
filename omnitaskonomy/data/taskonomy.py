"""Restore Taskonomy training pools from the published, ordered sample selections."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageFile

from omnitaskonomy.data.common import read_jsonl, resolve_image, sha256, write_manifest


REPO_ID = "Wakals/OmniTaskonomy"
RELEASE_REVISION = "8d9381c14e4c1ece63a2855128920225112eaff4"
TASKS = {
    "colorization": "i2i:colorization",
    "depth_zbuffer": "i2i:z_depth",
    "depth_euclidean": "i2i:euclidean_depth",
    "normal": "i2i:surface_normals",
    "principal_curvature": "i2i:principal_curvature",
    "edge_occlusion": "i2i:occlusion_edges",
    "keypoints3d": "i2i:keypoints_3d",
    "reshading": "i2i:reshading",
    "edge_texture": "i2i:edges_2d",
    "keypoints2d": "i2i:keypoints_2d",
    "segment_unsup2d": "i2i:segmentation_2d",
    "segment_unsup25d": "i2i:segmentation_25d",
}


def _ordered_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update((value + "\n").encode())
    return digest.hexdigest()


def load_selection(task, revision):
    from huggingface_hub import hf_hub_download

    catalog_path = Path(hf_hub_download(REPO_ID, "sources/taskonomy/sources.json",
                                     repo_type="dataset", revision=revision))
    catalog = json.loads(catalog_path.read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported Taskonomy sources schema")
    spec = next(item for item in catalog["tasks"] if item["task_id"] == TASKS[task])
    filename = f"sources/taskonomy/selections/{TASKS[task].split(':')[1]}.jsonl.gz"
    if spec["selection_file"] != filename:
        raise ValueError("Unexpected Taskonomy selection path")
    selection = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision))
    if sha256(selection) != spec["compressed_sha256"]:
        raise ValueError(f"Selection checksum mismatch: {filename}")
    ids = []
    for row in read_jsonl(selection):
        if set(row) != {"id"} or not re.fullmatch(r"[a-z]+/point_\d+_view_\d+", row["id"]):
            raise ValueError(f"Invalid Taskonomy sample ID: {row}")
        ids.append(row["id"])
    if (len(ids) != spec["rows"] or len(set(ids)) != spec["unique_ids"]
            or len(ids) - len(set(ids)) != spec["repeated_rows"]
            or _ordered_hash(ids) != spec["ordered_id_sha256"]):
        raise ValueError(f"Selection count or order mismatch: {filename}")
    expected_domain = "rgb" if task == "colorization" else task
    if spec["input_domain"] != "rgb" or spec["target_domain"] != expected_domain:
        raise ValueError(f"Unexpected image domains for {task}")
    return catalog, spec, ids


def _file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / archive["filename"]
    if path.exists():
        if _file_md5(path) != archive["md5"]:
            raise ValueError(f"Archive checksum mismatch; remove and retry: {path}")
        return path
    temporary = path.with_suffix(".tar.part")
    print(f"Downloading {archive['filename']}", flush=True)
    with urlopen(archive["url"], timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if _file_md5(temporary) != archive["md5"]:
        raise ValueError(f"Archive checksum mismatch: {temporary}")
    temporary.replace(path)
    return path


def _extract_selected(archive_path, domain, building, points, raw_root):
    directory = raw_root / domain / "taskonomy" / building
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {f"{domain}/{point}_domain_{domain}.png": point for point in points}
    found = set()
    with tarfile.open(archive_path, "r|") as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name not in wanted:
                continue
            if not member.isfile() or name in found:
                raise ValueError(f"Invalid selected archive member: {member.name}")
            destination = directory / PurePosixPath(name).name
            temporary = destination.with_suffix(".png.part")
            with archive.extractfile(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
            found.add(name)
    if found != wanted.keys():
        raise ValueError(f"Missing selected archive members in {archive_path}: {sorted(wanted.keys() - found)[:3]}")


def _ensure_sources(catalog, spec, ids, raw_root):
    by_building = defaultdict(set)
    for sample_id in ids:
        building, point = sample_id.split("/")
        by_building[building].add(point)
    required = {f"{building}_{domain}.tar" for building in by_building
                for domain in {spec["input_domain"], spec["target_domain"]}}
    if required != set(spec["archives"]):
        raise ValueError("Taskonomy archive list does not match selected IDs")
    archives = {item["filename"]: item for item in catalog["archives"]}
    for domain in sorted({spec["input_domain"], spec["target_domain"]}):
        for building, points in sorted(by_building.items()):
            directory = raw_root / domain / "taskonomy" / building
            available = set()
            if directory.is_dir():
                with os.scandir(directory) as entries:
                    available = {entry.name for entry in entries if entry.is_file()}
            missing = {point for point in points if f"{point}_domain_{domain}.png" not in available}
            if missing:
                archive = archives[f"{building}_{domain}.tar"]
                cache = raw_root / ".archives"
                cache.mkdir(parents=True, exist_ok=True)
                with (cache / (archive["filename"] + ".lock")).open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    missing = {point for point in missing if not (directory / f"{point}_domain_{domain}.png").is_file()}
                    if missing:
                        path = _download_archive(archive, cache)
                        _extract_selected(path, domain, building, missing, raw_root)


def render_image(image, transform):
    if transform == "rgb_to_gray":
        return image.convert("RGB").convert("L").convert("RGB")
    if transform == "rgb":
        if image.mode == "RGBA" or image.info.get("transparency") is not None:
            rgba = image.convert("RGBA")
            result = Image.new("RGB", image.size, "white")
            result.paste(rgba, mask=rgba.getchannel("A"))
            return result
        return image.convert("RGB")
    if transform == "gray":
        return image.convert("L").convert("RGB")
    array = np.asarray(image).astype(np.float32)
    if transform == "depth16":
        valid = array < 65535
        if valid.sum() < 16:
            valid = np.ones_like(array, dtype=bool)
        lo, hi = float(np.percentile(array[valid], 2)), float(np.percentile(array[valid], 98))
        if hi <= lo:
            hi = lo + 1.0
        array = np.clip((array - lo) / (hi - lo), 0.0, 1.0) * 255.0
    elif transform == "minmax16":
        lo, hi = float(array.min()), float(array.max())
        array = (array - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(array)
    else:
        raise ValueError(f"Unknown Taskonomy transform: {transform}")
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def _prepare_image(path, transform, destination, recovery):
    if recovery and sha256(path) != recovery["source_file_sha256"]:
        raise ValueError(f"Recovery source checksum mismatch: {path}")
    truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = bool(recovery)
    try:
        with Image.open(path) as image:
            rgb = render_image(image, transform)
            unchanged = (transform == "rgb" and image.mode == "RGB"
                         and image.info.get("transparency") is None and not recovery)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = truncated
    if rgb.size != (512, 512):
        raise ValueError(f"Expected 512x512 Taskonomy image: {path}")
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    if recovery and digest != recovery["decoded_rgb_sha256"]:
        raise ValueError(f"Recovered pixel checksum mismatch: {path}")
    if unchanged:
        return str(path), digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.part")
    rgb.save(temporary, format="PNG", compress_level=3)
    temporary.replace(destination)
    return str(destination), digest


def _prepare_pair(sample_id, *, spec, raw_root, image_root, recoveries):
    building, point = sample_id.split("/")
    result = []
    for role in ("input", "target"):
        domain = spec[role + "_domain"]
        path = raw_root / domain / "taskonomy" / building / f"{point}_domain_{domain}.png"
        destination = image_root / role / building / f"{point}.png"
        result.append(_prepare_image(path, spec[role + "_transform"], destination, recoveries.get((sample_id, role))))
    return sample_id, result


def prepare_taskonomy(task, output, *, raw_root=Path("data/raw/taskonomy"),
                      revision=RELEASE_REVISION, workers=8):
    if task not in TASKS:
        raise ValueError(f"Unknown Taskonomy task: {task}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        from huggingface_hub import HfApi
        revision = HfApi().repo_info(REPO_ID, repo_type="dataset", revision=revision).sha
    output, raw_root = Path(output).resolve(), Path(raw_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preparation runs before distributed training; the lock also covers concurrent launchers.
    with (output.parent / ".taskonomy.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog, spec, ids = load_selection(task, revision)
        metadata = {"source_dataset": "Taskonomy", "repo_id": REPO_ID, "revision": revision,
                    "task_id": spec["task_id"], "selection_file": spec["selection_file"],
                    "selection_sha256": spec["compressed_sha256"], "ordered_id_sha256": spec["ordered_id_sha256"],
                    "selection_provenance": spec["selection_provenance"], "raw_root": str(raw_root)}
        if output.exists():
            saved = json.loads(output.with_suffix(".metadata.json").read_text())
            if any(saved[key] != value for key, value in metadata.items()) or sha256(output) != saved["manifest_sha256"]:
                raise ValueError(f"Existing manifest differs; choose a new output path: {output}")
            if all(resolve_image(output, row[field]).is_file() for row in read_jsonl(output)
                   for field in ("source_image", "target_image")):
                return output
            output.unlink()
        _ensure_sources(catalog, spec, ids, raw_root)
        recoveries = {(item["id"], item["role"]): item for item in spec["decode_recoveries"]}
        prepare = partial(_prepare_pair, spec=spec, raw_root=raw_root,
                          image_root=output.parent / "images", recoveries=recoveries)
        unique_ids = list(dict.fromkeys(ids))
        print(f"Preparing {task}: {len(ids)} occurrences, {len(unique_ids)} unique samples", flush=True)
        if workers == 1:
            images = dict(map(prepare, unique_ids))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                images = dict(pool.map(prepare, unique_ids, chunksize=64))
        for index, field in ((0, "input"), (1, "output")):
            expected = spec[f"ordered_{field}_rgb_sha256"]
            if _ordered_hash(images[sample_id][index][1] for sample_id in ids) != expected:
                raise ValueError(f"Prepared {field} pixel checksum mismatch for {task}; manifest was not written")
            metadata[f"ordered_{field}_rgb_sha256"] = expected
        repeats, rows = Counter(), []
        # These paths are already absolute; avoid per-image filesystem resolution on shared storage.
        for sample_id in ids:
            source, target = images[sample_id]
            occurrence = repeats[sample_id]
            repeats[sample_id] += 1
            rows.append({"uid": f"{spec['task_id']}:{sample_id}:{occurrence}", "source_uid": sample_id,
                         "repeat_index": occurrence, "source_image": os.path.relpath(source[0], output.parent),
                         "target_image": os.path.relpath(target[0], output.parent), "prompt": spec["editing_prompt"]})
        # The training launcher treats the manifest's presence as completion.
        prepared = output.with_name(output.stem + ".preparing.jsonl")
        write_manifest(rows, prepared, metadata)
        prepared.with_suffix(".metadata.json").replace(output.with_suffix(".metadata.json"))
        prepared.replace(output)
        return output
