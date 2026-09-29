"""Import frozen benchmark prompts, correct options, and ordered image assets."""

import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from omnitaskonomy.data.common import sha256, write_manifest
from omnitaskonomy.gradients.artifacts import stable_seed, write_json
from omnitaskonomy.taxonomy import load_taxonomy


def prepare_targets(manifest_path, taxonomy_path, output, *, path_map=None):
    """Read the original real_i2t manifest; model responses are never used as labels."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    source = json.loads(manifest_path.read_text())
    taxonomy = load_taxonomy(taxonomy_path)
    retained = {row["uid"]: row["task_id"] for row in taxonomy["retained"]}
    relocations = json.loads(Path(path_map).read_text()) if path_map else {}
    output.mkdir(parents=True, exist_ok=False)
    groups, total, verified_images = [], 0, {}
    for category in source["categories"]:
        target = category["id"]
        manifest = output / (target.replace(":", "_") + ".jsonl")
        records = []
        for row in category["samples"]:
            if retained.get(row["uid"]) != target or row["category_id"] != target:
                raise ValueError(f"Frozen target is absent from the released taxonomy: {row['uid']}")
            if row["answer"] not in row["options"] or len(row["answer"]) != 1 or not row["prompt"]:
                raise ValueError(f"Invalid frozen benchmark answer or prompt: {row['uid']}")
            prompt_digest = hashlib.sha256(json.dumps(row["prompt"], ensure_ascii=False,
                                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if "prompt_sha256" in row and row["prompt_sha256"] != prompt_digest:
                raise ValueError(f"Frozen benchmark prompt hash differs: {row['uid']}")
            images = []
            for asset in row["images"]:
                path = Path(asset["path"])
                for before, after in sorted(relocations.items(), key=lambda pair: len(pair[0]), reverse=True):
                    if path.is_relative_to(before):
                        path = Path(after) / path.relative_to(before)
                        break
                if path not in verified_images:
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        digest = hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()
                    verified_images[path] = sha256(path), digest
                file_digest, rgb_digest = verified_images[path]
                if file_digest != asset["sha256"]:
                    raise ValueError(f"Benchmark image hash differs: {path}")
                if rgb_digest != asset["rgb_sha256"]:
                    raise ValueError(f"Benchmark decoded image differs: {path}")
                images.append(os.path.relpath(path.resolve(), output))
            if not images:
                raise ValueError("Benchmark target requires at least one image")
            records.append({"uid": row["uid"], "source_uid": row["uid"], "fold": row["fold"], "images": images,
                            "data_seed": (seed := stable_seed(source["sampling_seed"], row["uid"]) % (2**32 - 1)),
                            "loss_seed": seed,
                            "gradient_mcq": {"prompt": row["prompt"], "answer": row["answer"]},
                            "conversations": [{"from": "human", "value": "<image>\n" * len(images) + row["prompt"]},
                                              {"from": "gpt", "value": row["answer"]}]})
        write_manifest(records, manifest, {"source_manifest_sha256": sha256(manifest_path), "task": target})
        groups.append({"task": target, "objective": "i2t", "manifest": manifest.name,
                       "samples_per_group": len(records), "vit_transform": source["image_transform"]})
        total += len(records)
        print(f"Prepared gradient target {target}: {len(records)} questions", flush=True)
    write_json(output / "target_config.json", {"schema_version": 1, "groups": groups, "selection_order": "manifest",
                                               "folds": 5, "sample_seed": source["sampling_seed"]})
    write_json(output / "provenance.json", {"source_manifest_sha256": sha256(manifest_path), "n": total,
                                            "taxonomy_provenance_sha256": sha256(Path(taxonomy_path) / "provenance.json"),
                                            "objective": "correct option label plus EOS; frozen evaluation prompts"})
    return output / "target_config.json"
