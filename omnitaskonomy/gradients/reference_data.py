# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Convert the frozen six-task reference selection without the research registry."""

import base64
from collections import defaultdict
import glob
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from omnitaskonomy.data.common import relative_image, sha256, write_manifest
from omnitaskonomy.gradients.artifacts import stable_seed, write_json
from omnitaskonomy.gradients.manifest import REFERENCE_TASKS


I2I_PROMPTS = {
    "jigsaw": "Rearrange the shuffled image patches to reconstruct the original image.",
    "zoomin": "Rearrange the zoomed-in views so they are ordered from least to most zoomed.",
}
I2T_PROMPTS = {
    "jigsaw": (
        "You are solving a 2x2 jigsaw puzzle. The puzzle pieces are currently "
        "scrambled. Your goal is to rearrange the pieces to recover the original "
        "image.\n\nReorder all pieces at once using the format: `('reorder', "
        "[i0, i1, i2, i3])` where the list represents the desired order of pieces "
        "from top-left to bottom-right.\n\nIndex-to-cell mapping (0-based "
        "rows/cols):\n- Index = row * 2 + col.\n- (0,0)->0, (0,1)->1, "
        "(1,0)->2, (1,1)->3.\n\n"
    ),
    "zoomin": (
        "You are given an original image and 4 zoomed-in views laid out left to "
        "right. Your goal is to rearrange them so they are ordered from least to "
        "most zoomed.\n\nReorder all views at once using the format: `('reorder', "
        "[i0, i1, i2, i3])` where the list gives the desired left-to-right "
        "arrangement using 1-based indices.\n\nPosition mapping (1-based, left to "
        "right):\n- Position 1: leftmost view\n- Position 2: second from left\n"
        "- Position 3: third from left\n- Position 4: rightmost view\n\n"
    ),
}

COLOR_PROMPTS = (
    "Adjust the color inside the circled region so it matches the original color at that location.",
    "Correct the color in the marked circular region so it matches the source image.",
    "Recolor the circled area to the original RGB color for that point in the image.",
)
COUNT_PROMPTS = {
    "i2i": (
        "Mark all {category} in this image by placing a dot on each instance.",
        "Place a dot on every {category} visible in the image.",
        "Identify each {category} in the image and mark their locations with dots.",
        "Mark every instance of {category} in the image with a dot.",
        "Add a dot marker on each {category} you can see in this image.",
    ),
    "i2t": (
        "How many {category} are in this image? Answer with a number.",
        "Count the number of {category} in the image. Give the count only.",
        "How many {category} do you see? Respond with just the integer count.",
        "What is the total count of {category} visible in this image?",
        "Count all {category} in the image and state the number.",
    ),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pair_key(task, record):
    fields = ("id", "source_image", "rotation_degrees") if task == "rotate_qa" else ("episode", "episode_seed")
    return {field: record[field] for field in fields}


def _load_selection(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("manifest_schema_version") == "refine_grad_analysis_v1.manifest.v2":
        expected = value["manifest_sha256"]
        if _digest({key: item for key, item in value.items() if key != "manifest_sha256"}) != expected:
            raise ValueError("Legacy gradient selection digest differs")
    elif value.get("format") == "omnitaskonomy-gradient-reference-selection-v1":
        expected = value["selection_sha256"]
        if _digest({key: item for key, item in value.items() if key != "selection_sha256"}) != expected:
            raise ValueError("Released gradient selection digest differs")
    else:
        raise ValueError("Expected frozen refine_grad_analysis_v1 v2 or released reference selection")
    for task, section in value["tasks"].items():
        if task not in REFERENCE_TASKS:
            raise ValueError(f"Unknown reference task: {task}")
        pairs = section["pairs"]
        if [pair["position"] for pair in pairs] != list(range(len(pairs))):
            raise ValueError(f"Non-contiguous selected positions: {task}")
        for pair in pairs:
            if pair["source_id"] != task + ":" + _digest({"task": task, "pair_key": pair["pair_key"]}):
                raise ValueError(f"Frozen source identity differs: {task}")
            if type(pair["fold"]) is not int or not 0 <= pair["fold"] < value["n_folds"]:
                raise ValueError(f"Invalid frozen fold: {task}")
        if len({pair["source_id"] for pair in pairs}) != len(pairs):
            raise ValueError(f"Duplicate frozen reference source: {task}")
    return value


def _rgb(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode == "RGBA" or image.info.get("transparency") is not None:
            rgba = image.convert("RGBA")
            result = Image.new("RGB", rgba.size, "white")
            result.paste(rgba, mask=rgba.getchannel("A"))
            return result
        return image.convert("RGB")


def _square(image):
    width, height = image.size
    dx, dy = max(height - width, 0), max(width - height, 0)
    return Image.fromarray(np.pad(np.asarray(image),
        ((dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2), (0, 0)), mode="edge"))


def _strip_markup(text):
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        text = text.replace(tag, "")
    return text


def _video_prompt(step):
    match = re.search(r"The action being performed in the video is: '(.+?)'", step.get("prompt", ""))
    action = f" The action being performed is: '{match.group(1)}'.\n\n" if match else "\n\n"
    return ("You are given 4 video frames from a short clip laid out left to right in shuffled order." + action +
            "Rearrange the frames so they appear in their original chronological order from left to right.\n\n"
            "Reorder all frames at once using the format: `('reorder', [i0, i1, i2, i3])` "
            "where the list gives the desired left-to-right arrangement using 1-based indices.\n\n"
            "Position mapping (1-based, left to right):\n- Position 1: leftmost frame\n"
            "- Position 2: second from left\n- Position 3: third from left\n- Position 4: rightmost frame\n\n")


def _episode_sample(task, objective, record, data_seed):
    history = record["history"]
    first = history[0]
    if task == "colorization":
        source = first.get("image") or first["image_prev"]
    else:
        source = first.get("image_prev") or first["image"]
    images = [_rgb(base64.b64decode(source, validate=True))]
    if objective == "i2i":
        target = history[-1]["image_next"]
        if task == "counting":
            marks = [step for step in history if "mark" in str(step.get("action", ""))
                     and "unmark" not in str(step.get("action", ""))]
            if not marks:
                raise ValueError("Selected Counting I2I episode has no mark action")
            target = marks[-1]["image_next"]
        elif task == "colorization":
            target = first["image_next"]
        images.append(_rgb(base64.b64decode(target, validate=True)))
        if task == "jigsaw":
            images = [_square(image) for image in images]
    answer = None
    if task in {"jigsaw", "zoomin"}:
        prompt = (I2I_PROMPTS if objective == "i2i" else I2T_PROMPTS)[task]
        if objective == "i2t":
            answer = _strip_markup(first["vlm_output"])
    elif task == "video_unshuffle_3d":
        prompt = ("Rearrange the video frames so they appear in their original chronological order from left to right."
                  if objective == "i2i" else _video_prompt(first))
        if objective == "i2t":
            answer = _strip_markup(first["vlm_output"])
    elif task == "counting":
        match = re.search(r"Count the number of (\S+) in the image", first.get("prompt", ""))
        category = match.group(1).replace("_", " ") if match else "objects"
        prompt = random.Random(data_seed).choice(COUNT_PROMPTS[objective]).format(category=category)
        if objective == "i2t":
            match = re.search(r"'guess',\s*(\d+)", str(first["action"]))
            if not match:
                raise ValueError("Selected Counting I2T episode has no guess action")
            answer = str(int(match.group(1)))
    else:
        if objective == "i2i":
            prompt = random.Random(data_seed).choice(COLOR_PROMPTS)
        else:
            prompt = first["prompt"].strip()
            info = first.get("info") or record.get("extra_state") or {}
            answer = info.get("correct_option_letter") or first.get("action") or _strip_markup(first["vlm_output"])
            answer = answer.strip()
    return images, prompt, answer


def _rotate_sample(objective, record, spec, root, pair):
    target = root / spec["images"] / record["image"]
    paths = {"target_image": target, "input_image": target}
    if objective == "i2i":
        paths["source_image"] = root / spec["source_images"] / record.get("source_split", "train2017") / record["source_image"]
    assets = pair[objective + "_input_assets"]
    expected_roles = {"source_image", "target_image"} if objective == "i2i" else {"input_image"}
    if {asset["role"] for asset in assets} != expected_roles:
        raise ValueError("Rotate-QA selection must fingerprint every external image")
    for asset in assets:
        if sha256(paths[asset["role"]]) != asset["sha256"]:
            raise ValueError(f"Rotate-QA external image differs: {paths[asset['role']]}")
    if objective == "i2t":
        return [_rgb(target.read_bytes())], record["question"].strip(), str(record.get("answer") or record["correct_choice"]).strip()
    size = int(record.get("image_size", 512))
    source = ImageOps.fit(_rgb(paths["source_image"].read_bytes()), (size, size),
                          method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    return [source, _rgb(target.read_bytes())], f"Rotate the image {record['rotation_degrees']} degrees clockwise.", None


def _sample_seeds(task, objective, identity, sample_seed):
    return {name + "_seed": stable_seed(sample_seed, f"{name}:{task}:{objective}:{identity}") % (2**32 - 1)
            for name in ("data", "loss")}


def _write_sample(task, objective, pair, output, sample_seed, images, prompt, answer, provenance):
    identity = pair["source_id"]
    if not prompt.strip() or (objective == "i2t" and not answer.strip()):
        raise ValueError(f"Empty frozen supervision: {task}/{objective}/{identity}")
    manifest = output / task / f"{objective}.jsonl"
    row = {"uid": identity, "source_uid": identity, "fold": pair["fold"],
           "selection_position": pair["position"],
           **_sample_seeds(task, objective, identity, sample_seed), **provenance}
    image_keys = ("source_image", "target_image") if objective == "i2i" else ("image",)
    for key, image in zip(image_keys, images):
        path = output / task / "images" / f"{pair['position']:04d}_{objective}_{key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        row[key] = relative_image(path, manifest)
    if objective == "i2i":
        row["prompt"] = prompt
    else:
        row["conversations"] = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
    return row


def _convert(task, objective, record, spec, root, pair, output, sample_seed, source_path, line):
    seeds = _sample_seeds(task, objective, pair["source_id"], sample_seed)
    images, prompt, answer = (_rotate_sample(objective, record, spec, root, pair) if task == "rotate_qa"
                              else _episode_sample(task, objective, record, seeds["data_seed"]))
    provenance = {"source_record_sha256": pair[objective + "_record_sha256"],
                  "source_jsonl": str(source_path), "source_line": line}
    return _write_sample(task, objective, pair, output, sample_seed, images, prompt, answer, provenance)


def _reference_config(selection, groups, counts):
    if len(set(counts.values())) != 1:
        raise ValueError("Reference tasks must have the same number of pairs")
    return {"schema_version": 1, "groups": groups, "samples_per_group": next(iter(counts.values())),
            "folds": selection["n_folds"], **selection["seeds"], "selection_order": "manifest",
            "paired_reference": True, "paper_reference": set(counts) == REFERENCE_TASKS and
            set(counts.values()) == {500} and selection["n_folds"] == 5}


def prepare_recipe_reference(manifest, output, *, tasks=None):
    """Restore the frozen gradient pairs from the pinned Recipe Data release."""
    from omnitaskonomy.data.recipe import REPO_ID, RELEASE_REVISION, iter_recipe_rows

    selection = _load_selection(manifest)
    tasks = selection["task_order"] if tasks is None else tasks
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - selection["tasks"].keys():
        raise ValueError("Reference tasks must be distinct members of the frozen selection")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"repo_id": REPO_ID, "revision": RELEASE_REVISION,
                  "selection_sha256": sha256(manifest)}
    groups, counts = [], {}
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        for task in tasks:
            pairs = selection["tasks"][task]["pairs"]
            wanted = {_digest(pair["pair_key"]): pair for pair in pairs}
            records = {objective: {} for objective in ("i2i", "i2t")}
            for record in iter_recipe_rows(task):
                metadata = json.loads(record["metadata"])
                key = ({"id": metadata["original_id"], "source_image": metadata["source_image"],
                        "rotation_degrees": metadata["rotation_degrees"]} if task == "rotate_qa"
                       else {field: metadata[field] for field in ("episode", "episode_seed")})
                pair = wanted.get(_digest(key))
                if pair is None:
                    continue
                identity = pair["source_id"]
                if identity in records["i2i"]:
                    raise ValueError(f"Duplicate selected source: {identity}")
                for objective in records:
                    fields = ("i2i_input_image", "i2i_output_image") if objective == "i2i" else ("i2t_input_image",)
                    images = [_rgb(record[field]["bytes"]) for field in fields]
                    prompt = record[objective + "_prompt"]
                    seed = _sample_seeds(task, objective, identity, selection["seeds"]["sample_seed"])["data_seed"]
                    if task == "counting":
                        prompt = random.Random(seed).choice(metadata[objective + "_prompt_variants"]).format(
                            category=metadata["category"])
                    elif task == "colorization" and objective == "i2i":
                        prompt = random.Random(seed).choice(metadata["gradient_i2i_prompt_variants"])
                    answer = record["i2t_answer"] if objective == "i2t" else None
                    records[objective][identity] = _write_sample(
                        task, objective, pair, staged, selection["seeds"]["sample_seed"], images, prompt, answer,
                        {"recipe_id": record["id"], **provenance})
            for objective, selected in records.items():
                if len(selected) != len(pairs):
                    raise ValueError(f"Missing frozen {task}/{objective} records: found {len(selected)} of {len(pairs)}")
                write_manifest([selected[pair["source_id"]] for pair in pairs],
                               staged / task / f"{objective}.jsonl", {"task": task, "objective": objective, **provenance})
                groups.append({"task": task, "objective": objective, "manifest": f"{task}/{objective}.jsonl"})
            counts[task] = len(pairs)
        config = _reference_config(selection, groups, counts)
        write_json(staged / "reference_config.json", config)
        write_json(staged / "provenance.json", {**provenance, "counts": counts, "seeds": selection["seeds"],
            "converter_sha256": sha256(__file__),
            "rendering": "Released paired images; frozen per-objective prompts, data/loss RNG and source positions"})
        staged.rename(output)
    return config


def prepare_reference(manifest, source_root, output, *, sources, tasks=None):
    selection = _load_selection(manifest)
    source_root, output, sources = Path(source_root).resolve(), Path(output).resolve(), Path(sources).resolve()
    source_config = json.loads(sources.read_text())
    if source_config["schema_version"] != 1:
        raise ValueError("Unsupported reference source configuration")
    tasks = selection["task_order"] if tasks is None else tasks
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - selection["tasks"].keys():
        raise ValueError("Reference tasks must be distinct members of the frozen selection")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups, counts = [], {}
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        for task in tasks:
            section = selection["tasks"][task]
            pairs = section["pairs"]
            wanted = {_digest(pair["pair_key"]): pair for pair in pairs}
            records = {objective: {} for objective in ("i2i", "i2t")}
            by_files = defaultdict(list)
            for objective in records:
                spec = source_config["tasks"][task][objective]
                files = sorted({Path(path).resolve() for pattern in spec["jsonl"]
                                for path in glob.glob(str(source_root / pattern))})
                if not files:
                    raise FileNotFoundError(f"No source JSONLs for {task}/{objective}: {spec['jsonl']}")
                by_files[tuple(files)].append(objective)
            for files, objectives in by_files.items():
                for source in files:
                    with source.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            pair = wanted.get(_digest(_pair_key(task, record)))
                            if pair is None:
                                continue
                            digest = _digest(record)
                            for objective in objectives:
                                if digest != pair[objective + "_record_sha256"]:
                                    raise ValueError(f"Frozen {task}/{objective} record changed at {source}:{line_number}")
                                if pair["source_id"] in records[objective]:
                                    raise ValueError(f"Duplicate selected source: {pair['source_id']}")
                                records[objective][pair["source_id"]] = _convert(task, objective, record,
                                    source_config["tasks"][task][objective], source_root, pair, staged,
                                    selection["seeds"]["sample_seed"], source, line_number)
            for objective, selected in records.items():
                if len(selected) != len(pairs):
                    raise ValueError(f"Missing frozen {task}/{objective} records: found {len(selected)} of {len(pairs)}")
                rows = [selected[pair["source_id"]] for pair in pairs]
                write_manifest(rows, staged / task / f"{objective}.jsonl",
                               {"task": task, "objective": objective, "selection_sha256": sha256(manifest)})
                groups.append({"task": task, "objective": objective, "manifest": f"{task}/{objective}.jsonl"})
            counts[task] = len(pairs)
        config = _reference_config(selection, groups, counts)
        write_json(staged / "reference_config.json", config)
        write_json(staged / "provenance.json", {"selection_sha256": sha256(manifest),
            "source_config_sha256": sha256(sources), "converter_sha256": sha256(__file__),
            "source_root": str(source_root), "counts": counts, "seeds": selection["seeds"],
            "rendering": "BAGEL reference adapters; frozen per-objective data/loss RNG and source positions"})
        staged.rename(output)
    return config
