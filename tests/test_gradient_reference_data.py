import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from omnitaskonomy.data.common import read_jsonl, sha256
from omnitaskonomy.gradients.artifacts import stable_seed
from omnitaskonomy.gradients.manifest import freeze, REFERENCE_TASKS
from omnitaskonomy.gradients.reference_data import (
    COLOR_PROMPTS, COUNT_PROMPTS, _digest, _episode_sample, _load_selection, _rotate_sample,
    prepare_recipe_reference, prepare_reference,
)
from omnitaskonomy.train import ROOT


def encoded(color, size=(8, 4)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def fixture(root):
    selection = {"schema_version": 1, "format": "omnitaskonomy-gradient-reference-selection-v1",
        "task_order": sorted(REFERENCE_TASKS), "n_folds": 5,
        "seeds": {"sample_seed": 1000, "fold_seed": 20260826}, "tasks": {}}
    sources = {"schema_version": 1, "tasks": {}}
    for task_index, task in enumerate(selection["task_order"]):
        rows = {objective: [] for objective in ("i2i", "i2t")}
        pairs = []
        for index in range(5):
            source, target = encoded((task_index * 30, index * 30, 0)), encoded((task_index * 30, index * 30, 100))
            pair_key = {"episode": index, "episode_seed": index + 100}
            if task == "rotate_qa":
                coco = root / f"coco/train2017/{index}.png"
                rotated = root / f"rotate/images/{index}.png"
                coco.parent.mkdir(parents=True, exist_ok=True)
                rotated.parent.mkdir(parents=True, exist_ok=True)
                coco.write_bytes(base64.b64decode(source))
                rotated.write_bytes(base64.b64decode(target))
                record = {"id": index, "source_image": f"{index}.png", "rotation_degrees": 45,
                          "image": f"{index}.png", "image_size": 6, "question": "Rotation?", "correct_choice": "B"}
                pair_key = {key: record[key] for key in ("id", "source_image", "rotation_degrees")}
                i2i = i2t = record
            else:
                first = {"image_prev": source, "image_next": target, "prompt": "Arrange views.",
                         "action": "('reorder', [1, 2, 3, 4])", "vlm_output": "<answer>('reorder', [1, 2, 3, 4])</answer>"}
                i2i = {**pair_key, "history": [first]}
                i2t = json.loads(json.dumps(i2i))
                if task == "colorization":
                    i2i["history"][0] = {"image": source, "image_next": target}
                    i2t["history"][0] = {"image": encoded((task_index * 30, index * 30, 200), (12, 4)),
                        "prompt": "Choose the original color.", "info": {"correct_option_letter": "A"}}
                if task == "counting":
                    i2i["history"][0].update(prompt="Count the number of fire_hydrants in the image", action="('mark', (1, 1))")
                    i2i["history"].append({"action": "('guess', 1)", "image_next": source})
                    i2t["history"][0].update(prompt="Count the number of fire_hydrants in the image", action="('guess', 1)")
                if task == "video_unshuffle_3d":
                    i2t["history"][0]["prompt"] = "The action being performed in the video is: 'move a cube'"
            pair = {"position": index, "source_id": task + ":" + _digest({"task": task, "pair_key": pair_key}),
                    "pair_key": pair_key, "fold": index, "i2i_record_sha256": _digest(i2i),
                    "i2t_record_sha256": _digest(i2t), "i2i_input_assets": [], "i2t_input_assets": []}
            if task == "rotate_qa":
                pair["i2i_input_assets"] = [{"role": "source_image", "sha256": sha256(coco)},
                                             {"role": "target_image", "sha256": sha256(rotated)}]
                pair["i2t_input_assets"] = [{"role": "input_image", "sha256": sha256(rotated)}]
            pairs.append(pair)
            rows["i2i"].append(i2i)
            rows["i2t"].append(i2t)
        selection["tasks"][task] = {"n_pairs": 5, "pairs": pairs}
        sources["tasks"][task] = {}
        for objective in rows:
            path = root / f"{task}_{objective}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows[objective])))
            sources["tasks"][task][objective] = {"jsonl": [path.name]}
            if task == "rotate_qa":
                sources["tasks"][task][objective].update(images="rotate/images", source_images="coco")
    selection["selection_sha256"] = _digest(selection)
    selected, source_config = root / "selection.json", root / "sources.json"
    selected.write_text(json.dumps(selection))
    source_config.write_text(json.dumps(sources))
    return selected, source_config


def recipe_rows(root, selected, sources):
    selection = _load_selection(selected)
    specs = json.loads(sources.read_text())["tasks"]
    result = {}
    for task, section in selection["tasks"].items():
        pairs = list(reversed(section["pairs"]))
        raw = {objective: list(read_jsonl(root / specs[task][objective]["jsonl"][0]))
               for objective in ("i2i", "i2t")}
        rows = []
        for index, pair in enumerate(pairs):
            metadata = dict(pair["pair_key"])
            if task == "rotate_qa":
                metadata["original_id"] = metadata.pop("id")
            if task == "counting":
                metadata.update(category="fire hydrants", i2i_prompt_variants=COUNT_PROMPTS["i2i"],
                                i2t_prompt_variants=COUNT_PROMPTS["i2t"])
            if task == "colorization":
                metadata["gradient_i2i_prompt_variants"] = COLOR_PROMPTS
            row = {"id": f"{task}/train/{index}", "metadata": json.dumps(metadata)}
            for objective in raw:
                record = raw[objective][index]
                images, prompt, answer = (_rotate_sample(objective, record, specs[task][objective], root, pair)
                    if task == "rotate_qa" else _episode_sample(task, objective, record, 42))
                fields = ("i2i_input_image", "i2i_output_image") if objective == "i2i" else ("i2t_input_image",)
                for field, image in zip(fields, images):
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    row[field] = {"bytes": buffer.getvalue(), "path": None}
                row[objective + "_prompt"] = prompt
                if objective == "i2t":
                    row["i2t_answer"] = answer
            if task == "colorization":
                row["i2i_prompt"] = "Inpaint the gray circle, preserving its outline."
            rows.append(row)
        result[task] = rows
    return result


class ReferencePreparationTests(unittest.TestCase):
    def test_hf_pairs_match_raw_reference_images_prompts_order_folds_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, sources = fixture(root)
            rows = recipe_rows(root, selected, sources)
            native = root / "native"
            released = root / "released"
            expected = prepare_reference(selected, root, native, sources=sources)
            with patch("omnitaskonomy.data.recipe.iter_recipe_rows", side_effect=lambda task: iter(rows[task])):
                actual = prepare_recipe_reference(selected, released)
            self.assertEqual(actual, expected)
            for task in REFERENCE_TASKS:
                for objective in ("i2i", "i2t"):
                    left = list(read_jsonl(native / task / f"{objective}.jsonl"))
                    right = list(read_jsonl(released / task / f"{objective}.jsonl"))
                    fields = ("source_image", "target_image") if objective == "i2i" else ("image",)
                    for raw, restored in zip(left, right):
                        for key in ("uid", "source_uid", "fold", "selection_position", "data_seed", "loss_seed"):
                            self.assertEqual(restored[key], raw[key])
                        key = "prompt" if objective == "i2i" else "conversations"
                        self.assertEqual(restored[key], raw[key])
                        self.assertNotIn("source_record_sha256", restored)
                        for field in fields:
                            with Image.open(native / task / raw[field]) as a, Image.open(released / task / restored[field]) as b:
                                self.assertEqual(a.size, b.size)
                                self.assertEqual(a.tobytes(), b.tobytes())
            frozen = freeze(released / "reference_config.json", root / "frozen.json")
            self.assertEqual(len(frozen["rows"]), 60)
            provenance = json.loads((released / "provenance.json").read_text())
            self.assertEqual(provenance["selection_sha256"], sha256(selected))
            self.assertEqual(provenance["repo_id"], "Wakals/OmniTaskonomy_Recipe_Data")

    def test_hf_missing_or_duplicate_selected_pairs_leave_no_partial_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, sources = fixture(root)
            rows = recipe_rows(root, selected, sources)["jigsaw"]
            for bad_rows, message in ((rows[:-1], "Missing frozen"), (rows + rows[:1], "Duplicate selected")):
                with patch("omnitaskonomy.data.recipe.iter_recipe_rows", return_value=iter(bad_rows)):
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_recipe_reference(selected, root / "prepared", tasks=["jigsaw"])
                self.assertFalse((root / "prepared").exists())

    def test_all_six_sources_preserve_order_folds_and_distinct_modality_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, sources = fixture(root)
            output = root / "prepared"
            result = prepare_reference(selected, root, output, sources=sources)
            self.assertEqual(len(result["groups"]), 12)
            self.assertEqual(result["selection_order"], "manifest")
            frozen = freeze(output / "reference_config.json", root / "frozen.json")
            self.assertEqual(len(frozen["rows"]), 60)
            for task in sorted(REFERENCE_TASKS):
                i2i, i2t = (list(read_jsonl(output / task / f"{objective}.jsonl")) for objective in ("i2i", "i2t"))
                self.assertEqual([row["uid"] for row in i2i], [row["uid"] for row in i2t])
                self.assertEqual([row["fold"] for row in i2i], list(range(5)))
                self.assertEqual([row["selection_position"] for row in i2i], list(range(5)))
                self.assertEqual(i2i[0]["data_seed"], stable_seed(1000, f"data:{task}:i2i:{i2i[0]['uid']}") % (2**32 - 1))
                self.assertEqual(i2t[0]["loss_seed"], stable_seed(1000, f"loss:{task}:i2t:{i2t[0]['uid']}") % (2**32 - 1))
                if task == "colorization":
                    with Image.open(output / task / i2i[0]["source_image"]) as image:
                        self.assertEqual(image.size, (8, 4))
                    with Image.open(output / task / i2t[0]["image"]) as image:
                        self.assertEqual(image.size, (12, 4))
                    self.assertEqual(i2t[0]["conversations"][-1]["value"], "A")
                if task == "jigsaw":
                    with Image.open(output / task / i2i[0]["source_image"]) as image:
                        self.assertEqual(image.size, (8, 8))
                    with Image.open(output / task / i2t[0]["image"]) as image:
                        self.assertEqual(image.size, (8, 4))
                if task == "counting":
                    self.assertIn("fire hydrants", i2i[0]["prompt"])
                    self.assertEqual(i2t[0]["conversations"][-1]["value"], "1")
                    with Image.open(output / task / i2i[0]["target_image"]) as image:
                        self.assertEqual(image.getpixel((0, 0))[2], 100)
                if task == "rotate_qa":
                    with Image.open(output / task / i2i[0]["source_image"]) as image:
                        self.assertEqual(image.size, (6, 6))
                    self.assertEqual(i2i[0]["prompt"], "Rotate the image 45 degrees clockwise.")
                if task == "video_unshuffle_3d":
                    self.assertIn("move a cube", i2t[0]["conversations"][0]["value"])

    def test_changed_selected_record_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, sources = fixture(root)
            path = root / "colorization_i2t.jsonl"
            rows = list(read_jsonl(path))
            rows[0]["history"][0]["info"]["correct_option_letter"] = "B"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "record changed"):
                prepare_reference(selected, root, root / "prepared", sources=sources, tasks=["colorization"])
            self.assertFalse((root / "prepared").exists())

    def test_rotate_external_image_is_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, sources = fixture(root)
            Image.new("RGB", (8, 4), "white").save(root / "coco/train2017/0.png")
            with self.assertRaisesRegex(ValueError, "external image differs"):
                prepare_reference(selected, root, root / "prepared", sources=sources, tasks=["rotate_qa"])
            self.assertFalse((root / "prepared").exists())

    def test_released_frozen_selection_has_exact_paper_population(self):
        selected = _load_selection(ROOT / "data/gradients/reference_selection.json.gz")
        self.assertEqual(set(selected["tasks"]), REFERENCE_TASKS)
        for task, section in selected["tasks"].items():
            self.assertEqual(len(section["pairs"]), 500)
            self.assertEqual([sum(pair["fold"] == fold for pair in section["pairs"]) for fold in range(5)], [100] * 5)
            self.assertNotIn("/home/", json.dumps(section))


if __name__ == "__main__":
    unittest.main()
