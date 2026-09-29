"""Non-pickle gradient arrays with authenticated row and parameter identities."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from omnitaskonomy.data.common import sha256


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") % (2**63 - 1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_rows(rows, *, paired=False, folds=5):
    seen, sources, images, pairs = set(), {}, {}, defaultdict(dict)
    for row in rows:
        if any(not isinstance(row[field], str) or not row[field] for field in ("task", "uid", "source_uid")):
            raise ValueError("Gradient task and source identities must be nonempty strings")
        key = row["task"], row["objective"], row["uid"]
        if key in seen or row["objective"] not in {"i2i", "i2t"}:
            raise ValueError(f"Duplicate or invalid gradient row: {key}")
        seen.add(key)
        if type(row["fold"]) is not int or not 0 <= row["fold"] < folds:
            raise ValueError(f"Invalid fold for {key}")
        identity = row["source_uid"]
        if sources.setdefault(identity, row["fold"]) != row["fold"]:
            raise ValueError(f"Source crosses folds: {identity}")
        if row["objective"] == "i2t":
            for digest in row.get("image_hashes", []):
                if images.setdefault(digest, row["fold"]) != row["fold"]:
                    raise ValueError("Identical decoded understanding image crosses folds")
        pair = pairs[row["task"], row["uid"]]
        pair[row["objective"]] = row
    if not rows:
        raise ValueError("Gradient rows are empty")
    if paired:
        for key, pair in pairs.items():
            if (set(pair) != {"i2i", "i2t"} or pair["i2i"]["fold"] != pair["i2t"]["fold"]
                    or pair["i2i"]["source_uid"] != pair["i2t"]["source_uid"]):
                raise ValueError(f"Missing or split objective pair: {key}")
        for task in {r["task"] for r in rows}:
            if {r["fold"] for r in rows if r["task"] == task} != set(range(folds)):
                raise ValueError(f"Missing reference fold for {task}")


def load_artifact(path):
    path = Path(path).resolve()
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    if meta["schema_version"] != 1 or meta["status"] != "complete":
        raise ValueError(f"Incomplete or unsupported gradient artifact: {path}")
    validate_rows(meta["rows"], folds=meta["folds"])
    if not meta["provenance"]:
        raise ValueError("Gradient artifact must record extraction/import provenance")
    arrays = {}
    for parameter in meta["parameters"]:
        name = parameter["name"]
        if parameter["representation"] not in {"raw", "countsketch"}:
            raise ValueError(f"Unknown gradient representation: {name}")
        if parameter["representation"] == "raw" and int(np.prod(parameter["shape"])) != parameter["stored_dim"]:
            raise ValueError(f"Raw gradient dimensions differ from parameter shape: {name}")
        if parameter["representation"] == "countsketch" and (
                parameter["sketch_seed"] is None or not parameter.get("sketch_method")
                or len(parameter.get("sketch_mapping_fingerprint", "")) != 64):
            raise ValueError(f"CountSketch parameter lacks mapping provenance: {name}")
        array_path = path / parameter["file"]
        if array_path.resolve().parent != path or sha256(array_path) != parameter["sha256"]:
            raise ValueError(f"Gradient array path/hash differs: {name}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if name in arrays or array.shape != (len(meta["rows"]), parameter["stored_dim"]):
            raise ValueError(f"Gradient array identity/shape differs: {name}")
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Gradient array must contain finite float32 values: {name}")
        arrays[name] = array
    if not arrays:
        raise ValueError("Gradient artifact has no parameters")
    for row in meta["rows"]:
        if not np.isfinite(row["loss"]) or type(row["loss_tokens"]) is not int or row["loss_tokens"] <= 0:
            raise ValueError("Invalid loss or valid-token count")
    return {"metadata": meta, "arrays": arrays,
            "audit": {"path": str(path), "metadata_sha256": sha256(meta_path),
                      "arrays": {p["file"]: p["sha256"] for p in meta["parameters"]}}}


def save_artifact(path, rows, parameters, arrays, provenance, *, state="base", folds=5):
    """Import numerical gradients using the same schema as GPU extraction."""
    validate_rows(rows, folds=folds)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    inventory = []
    for index, parameter in enumerate(parameters):
        array = np.asarray(arrays[parameter["name"]], dtype=np.float32)
        filename = f"parameter_{index:04d}.npy"
        np.save(path / filename, array, allow_pickle=False)
        inventory.append({**parameter, "file": filename, "stored_dim": array.shape[1],
                          "sha256": sha256(path / filename)})
    write_json(path / "metadata.json", {"schema_version": 1, "status": "complete", "state": state,
               "folds": folds, "rows": rows, "parameters": inventory, "provenance": provenance})
    return load_artifact(path)


def parameter_blocks(artifact, layers=True):
    groups = defaultdict(list)
    for parameter in artifact["metadata"]["parameters"]:
        groups[parameter["module"], None].append(parameter["name"])
        if layers:
            groups[parameter["module"], parameter["name"]].append(parameter["name"])
    return groups


def gram(artifact, names):
    n = len(artifact["metadata"]["rows"])
    result = np.zeros((n, n), dtype=np.float64)
    for name in names:
        array = artifact["arrays"][name]
        for start in range(0, array.shape[1], 2048):
            part = np.asarray(array[:, start:start + 2048], dtype=np.float64)
            result += part @ part.T
    return result
