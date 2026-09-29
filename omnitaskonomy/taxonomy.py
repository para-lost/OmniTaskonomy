"""Load the released task catalogue and frozen benchmark question assignments."""

import json
from pathlib import Path
import re

from .data.common import read_jsonl, sha256


DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "data/taxonomy"
FILES = ("tasks.json", "retained_questions.jsonl", "excluded_questions.jsonl")


def _question_keys(rows, label):
    uids, keys = set(), set()
    for row in rows:
        uid, benchmark, index = row["uid"], row["benchmark"], row["index"]
        if not isinstance(index, str) or not index or index != index.strip():
            raise ValueError(f"{label}: index must be a nonempty normalized string")
        if re.fullmatch(r"-?\d+\.0", index):
            raise ValueError(f"{label}: noncanonical numeric index {index!r}")
        if not isinstance(uid, str) or not isinstance(benchmark, str):
            raise ValueError(f"{label}: UID and benchmark must be strings")
        parts = uid.split("::")
        if len(parts) != 3 or parts[0] != benchmark or not parts[1] or parts[2] != index:
            raise ValueError(f"{label}: UID does not match benchmark/split/index: {uid}")
        if "split" in row and row["split"] != parts[1]:
            raise ValueError(f"{label}: split differs from UID: {uid}")
        key = (benchmark, index)
        if uid in uids or key in keys:
            raise ValueError(f"{label}: duplicate question UID or benchmark/index: {uid}")
        uids.add(uid)
        keys.add(key)
    return uids, keys


def load_taxonomy(directory=None):
    """Validate and return a taxonomy bundle; counts and hashes come from its receipt."""
    directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if provenance["schema_version"] != 1:
        raise ValueError("Unsupported taxonomy provenance schema_version")
    for name in FILES:
        if provenance["files"].get(name) != sha256(directory / name):
            raise ValueError(f"Taxonomy SHA-256 mismatch: {name}")

    catalogue = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    if catalogue["schema_version"] != 1:
        raise ValueError("Unsupported taxonomy schema_version")
    families, tasks = catalogue["families"], catalogue["tasks"]
    family_ids = set()
    for family in families:
        family_id = family["id"]
        if family_id not in {"REC", "RCN", "RORG"} or family_id in family_ids:
            raise ValueError(f"Invalid or duplicate family ID: {family_id}")
        for field in ("name", "definition"):
            if not isinstance(family[field], str) or not family[field].strip():
                raise ValueError(f"Family {family_id} needs a nonempty {field}")
        family_ids.add(family_id)

    by_id, source_ids = {}, set()
    for task in tasks:
        task_id, modality = task["id"], task["modality"]
        if (modality not in {"i2i", "i2t"} or not isinstance(task_id, str)
                or not re.fullmatch(rf"{modality}:[A-Za-z][A-Za-z0-9_]*", task_id)):
            raise ValueError(f"Invalid typed task ID or modality: {task_id}")
        if task_id in by_id:
            raise ValueError(f"Duplicate task ID: {task_id}")
        if task["family"] not in family_ids:
            raise ValueError(f"Unknown family for {task_id}: {task['family']}")
        for field in ("name", "definition"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise ValueError(f"Task {task_id} needs a nonempty {field}")
        if modality == "i2i":
            source_id = task["source_id"]
            if (not isinstance(source_id, str) or not source_id.strip()
                    or source_id != source_id.strip() or source_id in source_ids):
                raise ValueError(f"Invalid or duplicate I2I source_id: {source_id}")
            source_ids.add(source_id)
        by_id[task_id] = task

    retained = list(read_jsonl(directory / "retained_questions.jsonl"))
    excluded = list(read_jsonl(directory / "excluded_questions.jsonl"))
    retained_uids, retained_keys = _question_keys(retained, "retained")
    excluded_uids, excluded_keys = _question_keys(excluded, "excluded")
    if retained_uids & excluded_uids or retained_keys & excluded_keys:
        raise ValueError("Retained and excluded question sets overlap")
    for row in retained:
        task = by_id.get(row["task_id"])
        if task is None or task["modality"] != "i2t":
            raise ValueError(f"Retained question must map to an I2T task: {row['uid']}")
        if row["family"] != task["family"]:
            raise ValueError(f"Question family differs from its task: {row['uid']}")
    for row in excluded:
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError(f"Excluded question needs a reason: {row['uid']}")

    counts = {"i2i": sum(task["modality"] == "i2i" for task in tasks),
              "i2t": sum(task["modality"] == "i2t" for task in tasks),
              "retained": len(retained), "excluded": len(excluded)}
    for name, actual in counts.items():
        expected = provenance["counts"][name]
        if type(expected) is not int or expected < 0 or expected != actual:
            raise ValueError(f"Frozen taxonomy count mismatch for {name}: {actual} != {expected}")
    return {"tasks": tasks, "families": families, "retained": retained,
            "excluded": excluded, "provenance": provenance}
