import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from omnitaskonomy.train import BAGEL, build_plan, command, dataset_configs, main, parser, run


def arguments(root, *extra):
    return parser().parse_args([
        "--task", "jigsaw", "--i2i-manifest", str(root / "i2i.jsonl"),
        "--i2t-manifest", str(root / "i2t.jsonl"),
        "--model-path", str(root / "base"), "--output-dir", str(root / "runs"), *extra,
    ])


def flags(argv):
    start = argv.index(str(BAGEL / "train/pretrain_unified_navit.py")) + 1
    return dict(zip(argv[start::2], argv[start + 1::2]))


class TrainingPlanTests(unittest.TestCase):
    def test_transfer_stages_reset_optimizer_and_disable_i2t_dropout(self):
        plan = build_plan(arguments(Path("/fixture")))
        self.assertEqual([s["name"] for s in plan["stages"]], ["i2i", "i2t"])
        first, second = plan["stages"]
        one = flags(command(plan, first, 42, Path("/out/first"), plan["model_path"]))
        two = flags(command(plan, second, 42, Path("/out/second"), "/out/first/checkpoints/0000781"))
        self.assertEqual(one["--finetune_from_ema"], "True")
        self.assertEqual(two["--finetune_from_ema"], "False")
        self.assertEqual(two["--resume_model_only"], "True")
        self.assertEqual(two["--visual_gen"], "False")
        self.assertEqual([first["budget"], second["budget"]], [50000, 50000])
        self.assertEqual(plan["batch_size"], 4)
        self.assertEqual(plan["gradient_accumulation"], 4)
        self.assertEqual(second["dataset_config"]["i2t"]["num_used_data"], 50000)
        for launch in (one, two):
            self.assertEqual(launch["--warmup_steps"], "50")
            self.assertEqual(launch["--text_cond_dropout_prob"], "0.0")
            self.assertEqual(launch["--freeze_shared_for_i2i"], "False")
            self.assertEqual(launch["--freeze_vit"], "False")
            self.assertEqual(launch["--freeze_und"], "False")

    def test_all_controlled_recipes_and_frozen_scope(self):
        expected = {"r1": ["i2t"], "r2": ["i2i", "i2t"], "r3": ["mixed", "i2t"],
                    "r4": ["i2i", "mixed"], "r5": ["mixed"], "r6": ["i2i", "mixed"]}
        for recipe, names in expected.items():
            with self.subTest(recipe=recipe):
                plan = build_plan(arguments(Path("/fixture"), "--recipe", recipe, "--suite", "controlled"))
                self.assertEqual([s["name"] for s in plan["stages"]], names)
                self.assertEqual(plan["requested_i2t_budget"], 15000)
                i2t_stages = [stage for stage in plan["stages"] if stage["name"] != "i2i"]
                self.assertEqual(sum(stage["budget"] for stage in i2t_stages), 15040)
                self.assertEqual(plan["batch_size"], 16)
                self.assertEqual(plan["gradient_accumulation"], 1)
                for stage in plan["stages"]:
                    self.assertEqual(stage["condition_dropout"], 0.0 if stage["name"] == "i2t" else 0.1)
                    launch = flags(command(plan, stage, 42, Path("/out"), plan["model_path"]))
                    frozen = recipe == "r4" and stage["name"] == "i2i"
                    self.assertEqual(launch["--freeze_shared_for_i2i"], str(frozen))
                    self.assertEqual(launch["--freeze_vit"], str(frozen))
                    self.assertEqual(launch["--freeze_und"], str(frozen))
                    self.assertEqual(launch["--freeze_vae"], "True")
                    self.assertNotIn("--freeze_llm", launch)
                    self.assertEqual(stage["warmup_steps"], 50 if stage["name"] == "i2i" else 8)
                    for groups in dataset_configs(plan, stage, 42).values():
                        if "i2t" in groups:
                            self.assertEqual(groups["i2t"]["num_used_data"], 1000)
                if recipe == "r3":
                    mixed, pure = plan["stages"]
                    self.assertEqual([mixed["budget"], pure["budget"]], [7552, 7488])
                    self.assertEqual(mixed["dataset_config"]["i2i"]["fixed_batch_size"], 32)
                    self.assertEqual(list(pure["dataset_config"]), ["i2t"])

    def test_controlled_separates_model_and_modality_seeds(self):
        plan = build_plan(arguments(Path("/fixture"), "--suite", "controlled", "--recipe", "r3",
                                    "--seeds", "42", "123", "456"))
        mixed, pure = plan["stages"]
        self.assertEqual(plan["model_seeds"], [4396, 4396, 4396])
        for seed in plan["seeds"]:
            mixed_config = dataset_configs(plan, mixed, seed)["dataset.yaml"]
            pure_config = dataset_configs(plan, pure, seed)["dataset.yaml"]
            self.assertEqual(mixed_config["i2i"]["data_seed"], 42)
            self.assertEqual(mixed_config["i2t"]["data_seed"], seed)
            self.assertEqual(pure_config["i2t"]["data_seed"], seed)
            self.assertTrue(pure_config["i2t"]["shuffle_before_slice"])
            argv = flags(command(plan, mixed, seed, Path("/out"), plan["model_path"]))
            self.assertEqual(argv["--global_seed"], "4396")
        self.assertNotIn("data_seed", mixed["dataset_config"]["i2t"])
        self.assertNotIn("data_seed", pure["dataset_config"]["i2t"])

        explicit = build_plan(arguments(Path("/fixture"), "--seeds", "42", "123", "789",
                                        "--model-seeds", "4396", "1111", "3333", "--i2i-seed", "7"))
        self.assertEqual(explicit["model_seeds"], [4396, 1111, 3333])
        self.assertEqual(dataset_configs(explicit, explicit["stages"][0], 123)["dataset.yaml"]["i2i"]["data_seed"], 7)
        self.assertFalse(explicit["stages"][1]["dataset_config"]["i2t"]["shuffle_before_slice"])

    def test_target_interpolation_only_reaches_generation_groups(self):
        default = build_plan(arguments(Path("/fixture"), "--task", "semseg"))
        self.assertEqual(default["stages"][0]["dataset_config"]["i2i"]["target_interpolation"], "bicubic")
        for recipe in ("r1", "r2", "r3", "r5", "r6"):
            with self.subTest(recipe=recipe):
                plan = build_plan(arguments(Path("/fixture"), "--recipe", recipe,
                                            "--i2i-target-interpolation", "nearest"))
                for stage in plan["stages"]:
                    for groups in dataset_configs(plan, stage, 42).values():
                        if "curriculum" in groups:
                            continue
                        for group in groups.values():
                            if group["kind"] == "i2i":
                                self.assertEqual(group["target_interpolation"], "nearest")
                            else:
                                self.assertNotIn("target_interpolation", group)

    def test_instance_freezes_half_of_layers_only_in_first_stage(self):
        half = build_plan(arguments(Path("/fixture"), "--stage1-freeze-last-half-llm"))
        first = flags(command(half, half["stages"][0], 42, Path("/out"), half["model_path"]))
        second = flags(command(half, half["stages"][1], 42, Path("/out2"), "/stage1"))
        self.assertEqual(first["--freeze_llm_layers_ratio_from_end"], "0.5")
        self.assertNotIn("--freeze_llm_layers_ratio_from_end", second)

    def test_stage1_only_does_not_require_mixed_batch_compatibility(self):
        plan = build_plan(arguments(Path("/fixture"), "--recipe", "r4", "--suite", "controlled",
                                    "--i2i-budget", "3000", "--stop-after-stage1"))
        self.assertEqual([s["name"] for s in plan["stages"]], ["i2i"])
        self.assertTrue(plan["stop_after_stage1"])

    def test_controlled_r3_stages_use_complete_updates_with_accumulation(self):
        for extra in [[], ["--batch-size", "4"], ["--nproc-per-node", "1"]]:
            plan = build_plan(arguments(Path("/fixture"), "--suite", "controlled", "--recipe", "r3", *extra))
            batch = plan["batch_size"] * plan["nproc_per_node"] * plan["gradient_accumulation"]
            self.assertEqual([stage["budget"] // batch for stage in plan["stages"]], [118, 117])
            self.assertTrue(all(stage["budget"] % batch == 0 for stage in plan["stages"]))

    def test_r3_starts_i2t_from_mixed_weights_with_fresh_optimizer(self):
        for suite, budgets in (("controlled", [7552, 7488]), ("transfer", [25024, 24976])):
            with self.subTest(suite=suite):
                plan = build_plan(arguments(Path("/fixture"), "--suite", suite, "--recipe", "r3"))
                mixed, pure = plan["stages"]
                self.assertEqual([mixed["budget"], pure["budget"]], budgets)
                effective_batch = plan["batch_size"] * plan["nproc_per_node"] * plan["gradient_accumulation"]
                self.assertEqual(mixed["budget"] % effective_batch, 0)
                first = flags(command(plan, mixed, 42, Path("/out/first"), plan["model_path"]))
                second = flags(command(plan, pure, 42, Path("/out/second"), "/out/first/checkpoints/final"))
                self.assertEqual(first["--finetune_from_ema"], "True")
                self.assertEqual(second["--resume_from"], "/out/first/checkpoints/final")
                self.assertEqual(second["--finetune_from_ema"], "False")
                self.assertEqual(second["--resume_model_only"], "True")
                self.assertEqual(second["--warmup_steps"], first["--warmup_steps"])
                self.assertEqual(second["--visual_gen"], "False")
                for field in ("text_cond_dropout_prob", "vae_cond_dropout_prob", "vit_cond_dropout_prob"):
                    self.assertEqual(second["--" + field], "0.0")

    def test_controlled_stage1_rounds_visits_without_changing_pool(self):
        for recipe in ["r2", "r4", "r6"]:
            topologies = [[], ["--nproc-per-node", "1"]]
            if recipe == "r2":
                topologies.append(["--batch-size", "4"])
            for pool, visits in [(1876, 1920), (9380, 9408), (100000, 100032)]:
                for topology in topologies:
                    plan = build_plan(arguments(Path("/fixture"), "--suite", "controlled", "--recipe", recipe,
                        "--i2i-budget", str(pool), *topology))
                    stage = plan["stages"][0]
                    self.assertEqual(plan["requested_i2i_budget"], pool)
                    self.assertEqual(stage["budget"], visits)
                    self.assertEqual(stage["dataset_config"]["i2i"]["num_used_data"], pool)
                    self.assertEqual(stage["budget"] % 64, 0)

    def test_r3_100k_uses_archived_mixed_batch(self):
        plan = build_plan(arguments(Path("/fixture"), "--suite", "controlled", "--recipe", "r3",
                                    "--i2i-budget", "100000", "--cpu-offload"))
        stage = plan["stages"][0]
        self.assertEqual(stage["dataset_config"]["i2i"]["fixed_batch_size"] * 4, 428)
        self.assertEqual(flags(command(plan, stage, 42, Path("/out"), plan["model_path"]))["--cpu_offload"], "True")

    def test_single_gpu_accumulates_to_same_effective_batch(self):
        plan = build_plan(arguments(Path("/fixture"), "--nproc-per-node", "1"))
        self.assertEqual(plan["batch_size"] * plan["gradient_accumulation"], 64)
        self.assertEqual(plan["gradient_accumulation"], 16)

    def test_multiple_seeds_are_isolated_and_rng_changes(self):
        plan = build_plan(arguments(Path("/fixture"), "--seeds", "42", "123"))
        first = flags(command(plan, plan["stages"][0], 42, Path("/out/seed_42"), plan["model_path"]))
        second = flags(command(plan, plan["stages"][0], 123, Path("/out/seed_123"), plan["model_path"]))
        self.assertEqual(first["--global_seed"], "4396")
        self.assertEqual(second["--global_seed"], "4477")
        self.assertNotEqual(first["--checkpoint_dir"], second["--checkpoint_dir"])

    def test_invalid_configuration_fails_before_launch(self):
        for extra in [["--batch-size", "0"], ["--max-tokens-per-sample", "0"], ["--seeds", "42", "42"],
                      ["--condition-dropout", "1"], ["--i2t-budget", "1"],
                      ["--cpu-offload"],
                      ["--seeds", "42", "43", "--model-seeds", "1"], ["--model-seeds", "-1"],
                      ["--i2i-seed", "-1"], ["--suite", "controlled", "--recipe", "r3", "--i2t-budget", "64"],
                      ["--suite", "controlled", "--recipe", "r3", "--i2i-budget", "100000", "--batch-size", "8"],
                      ["--recipe", "r1", "--stage1-checkpoint", "/weights"],
                      ["--recipe", "r1", "--stop-after-stage1"],
                      ["--stop-after-stage1", "--stage1-checkpoint", "/weights"],
                      ["--recipe", "r4", "--stage1-freeze-last-half-llm"],
                      ["--stage1-checkpoint", "/weights", "--stage1-freeze-last-half-llm"],
                      ["--stage1-checkpoint", "/weights", "--freeze-input-ln"],
                      ["--recipe", "r3", "--suite", "controlled", "--i2i-budget", "3000"]]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_plan(arguments(Path("/fixture"), *extra))

    def test_dry_run_needs_no_models_or_cuda_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                main(["--task", "normal", "--recipe", "R1", "--i2t-manifest", str(root / "missing.jsonl"),
                      "--model-path", str(root / "missing-model"), "--output-dir", str(root / "out"),
                      "--seeds", "42", "123", "--dry-run"])
            payload = json.loads(capture.getvalue())
            self.assertEqual(len(payload["commands"]), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_bagel_weights_fail_before_data_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite, task, module, function in (
                ("transfer", "normal", "taskonomy", "prepare_taskonomy"),
                ("controlled", "jigsaw", "recipe", "prepare_recipe"),
            ):
                plan = build_plan(arguments(root, "--suite", suite, "--task", task))
                with self.subTest(suite=suite), patch(
                    f"omnitaskonomy.data.{module}.{function}", side_effect=AssertionError("unexpected preparation")
                ):
                    with self.assertRaisesRegex(FileNotFoundError, "Missing base model file"):
                        run(plan)
            self.assertEqual(list(root.iterdir()), [])

    def test_reused_stage1_dry_run_requires_only_stage2_inputs(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["--task", "normal", "--recipe", "r2", "--i2t-manifest", "/missing.jsonl",
                  "--model-path", "/base", "--stage1-checkpoint", "/stage1",
                  "--output-dir", "/out", "--seeds", "42", "43", "44", "--dry-run"])
        payload = json.loads(capture.getvalue())
        self.assertEqual([s["name"] for s in payload["plan"]["stages"]], ["i2t"])
        self.assertIsNone(payload["plan"]["i2i_manifest"])
        for argv, seed in zip(payload["commands"], [42, 43, 44]):
            launch = flags(argv)
            self.assertEqual(launch["--resume_from"], "/stage1")
            self.assertEqual(launch["--finetune_from_ema"], "False")
            self.assertEqual(launch["--global_seed"], str(4396 + seed - 42))
            self.assertIn(f"seed_{seed}/stage2_i2t", launch["--checkpoint_dir"])

    def test_stage1_reuse_records_identity_and_rejects_replaced_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, stage1 = root / "base", root / "stage1"
            base.mkdir()
            stage1.mkdir()
            for name in ["llm_config.json", "vit_config.json"]:
                (base / name).write_text("{}")
            (stage1 / "model.safetensors").write_bytes(b"fixed stage one")
            (root / "i2t.jsonl").write_text('{"uid": "fixture"}\n')
            args = parser().parse_args(["--task", "normal", "--model-path", str(base),
                "--i2t-manifest", str(root / "i2t.jsonl"), "--stage1-checkpoint", str(stage1),
                "--output-dir", str(root / "runs"), "--seeds", "42", "43"])
            plan = build_plan(args)

            def complete_stage(argv, **kwargs):
                launch = flags(argv)
                stage_dir = Path(launch["--results_dir"]).parent
                checkpoint = Path(launch["--checkpoint_dir"]) / "0000780"
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.safetensors").write_bytes(b"trained fixture")
                (stage_dir / "logs").mkdir()
                (stage_dir / "logs/completion.json").write_text(json.dumps({
                    "checkpoint": str(checkpoint), "sample_visits": 50000,
                    "optimizer_updates": 781, "pending_microbatches": 1}))

            with patch("omnitaskonomy.train.subprocess.run", side_effect=complete_stage):
                manifest = run(plan)
            records = json.loads(manifest.read_text())["records"]
            self.assertEqual([row["stage"] for row in records], ["i2t", "i2t"])
            self.assertEqual([row["model_seed"] for row in records], [4396, 4397])
            for record in records:
                self.assertEqual(record["initialization"], str(stage1))
                self.assertEqual(len(record["stage1_checkpoint_sha256"]), 64)
                stage_dir = Path(record["checkpoint"]).parents[1]
                config = yaml.safe_load((stage_dir / "dataset.yaml").read_text())
                self.assertEqual(config["i2t"]["data_seed"], record["seed"])
                self.assertFalse((stage_dir.parent / "stage1_i2i").exists())
            with patch("omnitaskonomy.train.subprocess.run", side_effect=AssertionError("completed stage relaunched")):
                self.assertEqual(run(plan), manifest)
                (stage1 / "model.safetensors").write_bytes(b"different initialization")
                with self.assertRaisesRegex(ValueError, "another configuration"):
                    run(plan)


class ManifestPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BAGEL))
        import torch
        from PIL import Image
        from omnitaskonomy.datasets import ManifestDataset
        from data.dataset_base import DataConfig, PackedDataset
        cls.torch, cls.Image, cls.Dataset = torch, Image, ManifestDataset
        cls.DataConfig, cls.PackedDataset = DataConfig, PackedDataset

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.Image.new("RGB", (32, 32), (40, 80, 160)).save(self.root / "image.png")

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, kind, rows, rank=0, world=1):
        path = self.root / f"{kind}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        torch = self.torch
        class Transform:
            stride = 16
            def __call__(self, image):
                return torch.zeros(3, 32, 32)
        class Tokenizer:
            def encode(self, text):
                return [10 + ord(char) % 30 for char in text]
        return self.Dataset(kind, path, kind, Transform(), Tokenizer(), Transform(),
                            local_rank=rank, world_size=world, num_workers=1)

    def i2i(self, uid="pair"):
        return {"uid": uid, "source_image": "image.png", "target_image": "image.png", "prompt": "Complete."}

    def i2t(self):
        return {"uid": "answer", "image": "image.png", "conversations": [
            {"from": "human", "value": "<image>\nQuestion?"}, {"from": "gpt", "value": "Answer."}]}

    def test_modalities_and_dropout_gates_match_original_contract(self):
        generation = next(iter(self.dataset("i2i", [self.i2i()])))
        plans = generation["sequence_plan"]
        self.assertEqual([p["type"] for p in plans], ["vae_image", "vit_image", "text", "vae_image"])
        self.assertEqual([p["enable_cfg"] for p in plans], [1, 1, 1, 0])
        self.assertEqual([p["loss"] for p in plans], [0, 0, 0, 1])
        understanding = next(iter(self.dataset("i2t", [self.i2t()])))
        self.assertEqual([p["type"] for p in understanding["sequence_plan"]], ["vit_image", "text", "text"])
        self.assertTrue(all(p["enable_cfg"] == 0 for p in understanding["sequence_plan"]))
        self.assertEqual([p["loss"] for p in understanding["sequence_plan"]], [0, 0, 1])

    def test_actual_packing_masks_only_answer_and_generation_target(self):
        # Exercise the original packer, without constructing a language model.
        packer = self.PackedDataset.__new__(self.PackedDataset)
        packer.use_flex = False
        packer.data_config = self.DataConfig({}, text_cond_dropout_prob=1,
                                            vit_cond_dropout_prob=1, vae_cond_dropout_prob=1,
                                            vit_patch_size=16)
        from data.data_utils import get_flattened_position_ids_extrapolate
        packer.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        for name, value in {"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3,
                            "end_of_image": 4}.items():
            setattr(packer, name, value)
        sample = next(iter(self.dataset("i2t", [self.i2t()])))
        status = packer.pack_sequence(sample, packer.set_sequence_status())
        self.assertEqual(len(status["ce_loss_indexes"]), len("Answer.") + 1)
        self.assertGreater(len(status["packed_vit_token_indexes"]), 0)
        self.assertEqual(status["mse_loss_indexes"], [])
        sample = next(iter(self.dataset("i2i", [self.i2i()])))
        status = packer.pack_sequence(sample, packer.set_sequence_status())
        self.assertEqual(status["ce_loss_indexes"], [])
        self.assertEqual(status["packed_vit_token_indexes"], [])
        self.assertGreater(len(status["mse_loss_indexes"]), 0)

    def test_full_packed_loader_uses_manifest_without_registry(self):
        dataset = self.dataset("i2t", [self.i2t()])
        config = self.DataConfig({"i2t": {
            "manifest": str(dataset.manifest), "kind": "i2t", "fixed_batch_size": 1,
            "weight": 1, "image_transform_args": {"image_stride": 14, "min_image_size": 28, "max_image_size": 28},
            "vit_image_transform_args": {"image_stride": 14, "min_image_size": 28, "max_image_size": 28},
        }})
        packed = self.PackedDataset(config, tokenizer=dataset.tokenizer,
            special_tokens={"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
            local_rank=0, world_size=1, num_workers=1)
        packed.set_epoch(123)
        batch = next(iter(packed))
        self.assertEqual(batch["batch_data_indexes"][0]["uid"], "answer")
        self.assertEqual(batch["packed_label_ids"].numel(), len("Answer.") + 1)
        self.assertEqual(len(batch["sample_lens"]), 1)

    def test_shards_cover_remainder_rows_without_duplicate_uids(self):
        rows = [self.i2i(str(i)) for i in range(5)]
        left, right = self.dataset("i2i", rows, 0, 2), self.dataset("i2i", rows, 1, 2)
        l_iter, r_iter = iter(left), iter(right)
        ids = [next(l_iter)["data_indexes"]["uid"] for _ in range(3)]
        ids += [next(r_iter)["data_indexes"]["uid"] for _ in range(2)]
        self.assertEqual(set(ids), {str(i) for i in range(5)})

    def test_missing_images_and_bad_conversations_fail(self):
        row = self.i2i(); row["source_image"] = "missing.png"
        with self.assertRaises(FileNotFoundError):
            next(iter(self.dataset("i2i", [row])))
        row = self.i2t(); row["conversations"] = [{"from": "human", "value": "question"}]
        with self.assertRaisesRegex(ValueError, "no supervised"):
            next(iter(self.dataset("i2t", [row])))


if __name__ == "__main__":
    unittest.main()
