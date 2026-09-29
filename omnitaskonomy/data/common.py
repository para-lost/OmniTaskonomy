from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).parent / path


def relative_image(path, manifest):
    return os.path.relpath(Path(path).resolve(), Path(manifest).resolve().parent)


def write_manifest(rows, output: Path, metadata: dict):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    ids = set()
    order = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            uid = str(row["uid"])
            if uid in ids:
                raise ValueError(f"Duplicate manifest UID: {uid}")
            ids.add(uid)
            order.update((uid + "\n").encode())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    metadata = dict(metadata, count=len(ids), ordered_uid_sha256=order.hexdigest(),
                    manifest_sha256=sha256(output))
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata
