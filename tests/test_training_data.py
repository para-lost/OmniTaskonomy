import ast
import copy
from itertools import islice
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Bagel"))
from data.dataset_base import CurriculumPackedDataset, DataConfig, PackedDataset


class Tokenizer:
    def encode(self, text):
        return [10 + ord(char) % 30 for char in text]


class BatchStream(torch.utils.data.IterableDataset):
    def __init__(self, world_size, rank, batch_size, generation):
        self.world_size, self.local_rank = world_size, rank
        self.batch_size, self.generation = batch_size, generation

    def __iter__(self):
        while True:
            groups = ["i2t"] * self.batch_size
            if self.generation:
                groups += ["i2i"] * self.batch_size
            yield {"batch_data_indexes": [{"dataset_name": group} for group in groups]}


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        Image.new("RGB", (32, 32), (40, 80, 160)).save(self.root / "image.png")
        for kind in ("i2i", "i2t"):
            rows = []
            for index in range(32):
                row = {"uid": f"{kind}-{index}"}
                if kind == "i2i":
                    row.update(source_image="image.png", target_image="image.png", prompt="Complete.")
                else:
                    row.update(image="image.png", conversations=[
                        {"from": "human", "value": "<image>\nQuestion?"},
                        {"from": "gpt", "value": "Answer."},
                    ])
                rows.append(row)
            (self.root / f"{kind}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def group(self, kind, **kwargs):
        return {"manifest": str(self.root / f"{kind}.jsonl"), "kind": kind,
                "num_used_data": 8, "fixed_batch_size": 1, "weight": 1,
                "image_transform_args": {"image_stride": 16, "min_image_size": 32, "max_image_size": 32},
                "vit_image_transform_args": {"image_stride": 14, "min_image_size": 28, "max_image_size": 28},
                **kwargs}

    def packed(self, groups, world_size=1):
        return PackedDataset(DataConfig(copy.deepcopy(groups)), tokenizer=Tokenizer(),
            special_tokens={"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
            local_rank=0, world_size=world_size, num_workers=1)

    @staticmethod
    def drawn_ids(packed, count=8):
        by_group = {dataset.dataset_name: [] for dataset in packed.grouped_datasets}
        for batch in islice(iter(packed), count):
            for item in batch["batch_data_indexes"]:
                by_group[item["dataset_name"]].append(item["uid"])
        return by_group

    def test_full_packer_selects_independent_group_pools_and_seeds(self):
        results = {}
        for seed in (42, 123, 456):
            packed = self.packed({
                "i2i": self.group("i2i", data_seed=42, shuffle_before_slice=True),
                "i2t": self.group("i2t", data_seed=None, shuffle_before_slice=True),
            })
            packed.set_epoch(seed)
            results[seed] = self.drawn_ids(packed)
        self.assertEqual(results[42]["i2i"], results[123]["i2i"])
        self.assertEqual(results[42]["i2i"], results[456]["i2i"])
        for seed, result in results.items():
            selected = list(range(32))
            random.Random(seed).shuffle(selected)
            self.assertEqual(result["i2t"], [f"i2t-{i}" for i in selected[:8]])
        self.assertNotEqual(set(results[42]["i2t"]), set(results[123]["i2t"]))
        self.assertNotEqual(set(results[123]["i2t"]), set(results[456]["i2t"]))

    def test_transfer_still_slices_first_and_fixed_seed_zero_is_respected(self):
        for fixed_seed in (None, 0):
            packed = self.packed({"i2t": self.group("i2t", data_seed=fixed_seed)})
            packed.set_epoch(123)
            selected = list(range(8))
            random.Random(123 if fixed_seed is None else fixed_seed).shuffle(selected)
            self.assertEqual(self.drawn_ids(packed)["i2t"], [f"i2t-{i}" for i in selected])

    def test_worker_capacity_is_checked_after_pool_selection(self):
        with self.assertRaisesRegex(ValueError, "one row per distributed data worker"):
            self.packed({"i2t": self.group("i2t", num_used_data=2, shuffle_before_slice=True)}, world_size=4)

    def test_nearest_targets_preserve_palette_and_source_packing(self):
        palette = [(0, 0, 0), (255, 0, 0), (0, 255, 255)]
        image = Image.new("P", (19, 23))
        image.putpalette([channel for color in palette for channel in color])
        image.putdata([(x // 5 + y // 7) % 3 for y in range(23) for x in range(19)])
        image.save(self.root / "target.png")
        image.convert("RGB").save(self.root / "image.png")
        rows = [json.loads(line) for line in (self.root / "i2i.jsonl").read_text().splitlines()]
        for row in rows:
            row["target_image"] = "target.png"
        (self.root / "i2i.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

        batches = []
        for interpolation in (None, "nearest"):
            options = {} if interpolation is None else {"target_interpolation": interpolation}
            packed = self.packed({"i2i": self.group("i2i", **options)})
            packed.data_config.text_cond_dropout_prob = 0
            packed.data_config.vae_cond_dropout_prob = 0
            packed.data_config.vit_cond_dropout_prob = 0
            dataset = packed.grouped_datasets[0]
            sample = dataset.parse_row(rows[0])
            torch.testing.assert_close(sample["image_tensor_list"][0],
                                       dataset.transform(image.convert("RGB")), rtol=0, atol=0)
            random.seed(42)
            np.random.seed(42)
            status = packed.pack_sequence(sample, packed.set_sequence_status())
            batches.append(packed.to_tensor(status))

        bicubic, nearest = batches
        torch.testing.assert_close(nearest["padded_images"][0], bicubic["padded_images"][0], rtol=0, atol=0)
        target = nearest["padded_images"][1]
        expected = image.convert("RGB").resize((32, 32), Image.Resampling.NEAREST)
        expected = torch.from_numpy(np.array(expected)).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        colors = (target.add(1).mul(127.5).round().byte().permute(1, 2, 0).reshape(-1, 3)).unique(dim=0)
        self.assertEqual({tuple(color) for color in colors.tolist()}, set(palette))
        self.assertFalse(torch.equal(target, bicubic["padded_images"][1]))
        torch.testing.assert_close(
            {key: value for key, value in nearest.items() if key != "padded_images"},
            {key: value for key, value in bicubic.items() if key != "padded_images"}, rtol=0, atol=0,
        )

    def test_unknown_target_interpolation_fails_before_loading_samples(self):
        with self.assertRaisesRegex(ValueError, "Unknown target interpolation"):
            self.packed({"i2i": self.group("i2i", target_interpolation="linear")})

    def trainer_datasets(self, phases):
        (self.root / "mixed.yaml").write_text(yaml.safe_dump({
            "i2i": self.group("i2i"), "i2t": self.group("i2t"),
        }))
        namespace = dict(
            DataConfig=DataConfig, PackedDataset=PackedDataset, CurriculumPackedDataset=CurriculumPackedDataset,
            model_args=SimpleNamespace(text_cond_dropout_prob=0.1, vae_cond_dropout_prob=0.2,
                vit_cond_dropout_prob=0.3, vit_patch_size=14, vit_max_num_patch_per_side=70,
                latent_patch_size=2, max_latent_size=64, interpolate_pos=False),
            training_args=SimpleNamespace(visual_und=True, visual_gen=True,
                expected_num_tokens=4096, use_flex=False),
            vae_config=SimpleNamespace(downsample=8), tokenizer=Tokenizer(),
            new_token_ids={"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
            dist=SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 1),
            logger=logging.getLogger(__name__), os=os, yaml=yaml,
            data_args=SimpleNamespace(dataset_config_file=str(self.root / "curriculum.yaml"), num_workers=1,
                max_num_tokens_per_sample=4096, max_num_tokens=4096, max_buffer_size=50, prefer_buffer_before=4096),
            dataset_meta={"curriculum": phases}, data_status=None,
        )
        # Execute the real nested dataset setup without initializing CUDA/FSDP or model weights.
        path = ROOT / "Bagel/train/pretrain_unified_navit.py"
        tree = ast.parse(path.read_text())
        make = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_make_packed_dataset")
        curriculum = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.If) and ast.unparse(node.test) == "'curriculum' in dataset_meta")
        module = ast.Module(body=[make, curriculum], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        return namespace

    def test_phase_dropout_changes_real_packing_and_cache_identity(self):
        phases = [{"dataset_config_file": "mixed.yaml", "num_samples": 8,
                   "target_dataset_name": "i2t", "conditioning_dropout_prob": p} for p in (0.0, 1.0, 0.0)]
        namespace = self.trainer_datasets(phases)
        first, second, third = [phase[0] for phase in namespace["train_dataset"].phases]
        self.assertIs(first, third)
        self.assertIsNot(first, second)
        batches = [next(iter(dataset)) for dataset in (first, second)]
        self.assertEqual(batches[0]["packed_label_ids"].tolist(), batches[1]["packed_label_ids"].tolist())
        self.assertEqual(batches[0]["mse_loss_indexes"].numel(), batches[1]["mse_loss_indexes"].numel())
        # I2T conditioning remains present; only the I2I source and instruction disappear.
        self.assertEqual(batches[0]["packed_vit_tokens"].shape[0], 2 * batches[1]["packed_vit_tokens"].shape[0])
        self.assertEqual(batches[0]["padded_images"].shape[0], 2)
        self.assertEqual(batches[1]["padded_images"].shape[0], 1)
        default = namespace["_make_packed_dataset"]({"i2t": self.group("i2t")}).data_config
        self.assertEqual((default.text_cond_dropout_prob, default.vae_cond_dropout_prob,
                          default.vit_cond_dropout_prob), (0.1, 0.2, 0.3))
        for invalid in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                namespace["_make_packed_dataset"]({}, conditioning_dropout_prob=invalid)

    def test_curriculum_boundaries_survive_prefetch_and_gradient_accumulation(self):
        # The launcher rounds the 15,000 requested visits to 235 complete batch-64 updates.
        for world_size, batch_size, accumulation in ((4, 16, 1), (4, 4, 4), (1, 16, 4)):
            for rank in sorted({0, world_size - 1}):
                with self.subTest(world_size=world_size, batch_size=batch_size, rank=rank):
                    mixed = BatchStream(world_size, rank, batch_size, True)
                    pure = BatchStream(world_size, rank, batch_size, False)
                    curriculum = CurriculumPackedDataset([(mixed, 7552, "i2t"), (pure, 7488, "i2t")])
                    loader = torch.utils.data.DataLoader(curriculum, batch_size=None, num_workers=1, prefetch_factor=2)
                    microbatches = []
                    visits = 0
                    for batch in loader:
                        names = [item["dataset_name"] for item in batch["batch_data_indexes"]]
                        visits += names.count("i2t") * world_size
                        microbatches.append("i2i" in names)
                        if visits >= 15040:
                            break
                    self.assertEqual(visits, 15040)
                    self.assertEqual(microbatches, [True] * (118 * accumulation) + [False] * (117 * accumulation))
                    self.assertEqual(len(microbatches) % accumulation, 0)


if __name__ == "__main__":
    unittest.main()
