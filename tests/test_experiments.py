import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omnitaskonomy.experiments import execute, load_experiment, main, select_jobs
from omnitaskonomy.train import BAGEL, ROOT


CONFIGS = ROOT / "configs/experiments"


class PaperExperimentTests(unittest.TestCase):
    def test_every_preset_expands_without_data_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config in CONFIGS.glob("*.json"):
                with self.subTest(config=config.name):
                    spec = load_experiment(config)
                    preview = execute(spec, config, root / "data", root / "model", root / "output", dry_run=True)
                    self.assertEqual(len(preview["jobs"]), len(spec["jobs"]))
                    self.assertTrue(all(job["commands"] for job in preview["jobs"]))
                    self.assertEqual(list(root.iterdir()), [])

    def test_scaling_shares_first_stage_and_preserves_three_seed_rngs(self):
        spec = load_experiment(CONFIGS / "controlled_scaling.json")
        selected = select_jobs(spec, ["jigsaw_r2_3k", "jigsaw_r6_3k"])
        self.assertEqual([j["id"] for j in selected], ["jigsaw_stage1_3k", "jigsaw_r2_3k", "jigsaw_r6_3k"])
        result = execute(spec, CONFIGS / "controlled_scaling.json", "/data", "/base", "/out",
                         jobs=["jigsaw_r2_3k", "jigsaw_r6_3k"], dry_run=True)
        initial, r2, r6 = [job["plan"] for job in result["jobs"]]
        self.assertEqual([stage["name"] for stage in initial["stages"]], ["i2i"])
        self.assertEqual(initial["seeds"], [42])
        self.assertEqual(initial["stages"][0]["budget"], 1920)
        self.assertEqual(r2["stage1_checkpoint"], r6["stage1_checkpoint"])
        for plan in (r2, r6):
            self.assertEqual(plan["seeds"], [42, 123, 456])
            self.assertEqual(plan["model_seeds"], [4396] * 3)
            self.assertEqual(plan["stages"][0]["budget"], 15040)

    def test_transfer_covers_taxonomy_with_paired_baseline(self):
        spec = load_experiment(CONFIGS / "transfer.json")
        catalogue = json.loads((ROOT / "data/taxonomy/tasks.json").read_text())
        expected = {t["id"] for t in catalogue["tasks"] if t["modality"] == "i2i"}
        sources = [j for j in spec["jobs"] if "stage1_from" in j]
        self.assertEqual({j["task_id"] for j in sources}, expected)
        self.assertEqual(len(sources), 19)
        result = execute(spec, CONFIGS / "transfer.json", "/data", "/base", "/out", dry_run=True)
        plans = {j["id"]: j["plan"] for j in result["jobs"]}
        baseline = plans["i2t_baseline"]
        for source in sources:
            plan = plans[source["id"]]
            self.assertEqual(plan["seeds"], baseline["seeds"])
            self.assertEqual(plan["model_seeds"], baseline["model_seeds"])
            self.assertEqual([s["name"] for s in plan["stages"]], ["i2t"])
            self.assertEqual(plan["stages"][0]["dataset_config"], baseline["stages"][0]["dataset_config"])
        semseg = plans["semseg_stage1"]["stages"][0]["dataset_config"]["i2i"]
        self.assertEqual(semseg["target_interpolation"], "nearest")
        self.assertIn("colorization/train.jsonl", plans["colorization_stage1"]["i2i_manifest"])

    def test_r4_uses_its_generation_only_checkpoint_then_unfreezes_mixed(self):
        config = CONFIGS / "controlled_scaling.json"
        spec = load_experiment(config)
        for task in ("jigsaw", "zoomin"):
            for budget in ("3k", "10k", "30k", "100k"):
                with self.subTest(task=task, budget=budget):
                    result = execute(spec, config, "/data", "/base", "/out",
                                     jobs=[f"{task}_r4_{budget}", f"{task}_r6_{budget}"], dry_run=True)
                    plans = {job["id"]: job["plan"] for job in result["jobs"]}
                    initial = plans[f"{task}_stage1_{budget}_frozen"]
                    r4, r6 = (plans[f"{task}_{recipe}_{budget}"] for recipe in ("r4", "r6"))
                    self.assertEqual([stage["name"] for stage in initial["stages"]], ["i2i"])
                    self.assertTrue(initial["stages"][0]["freeze"])
                    self.assertEqual(initial["seeds"], [42])
                    self.assertIn(initial["output_dir"], r4["stage1_checkpoint"])
                    self.assertNotEqual(r4["stage1_checkpoint"], r6["stage1_checkpoint"])
                    self.assertEqual(r4["stages"], r6["stages"])
                    self.assertEqual([stage["name"] for stage in r4["stages"]], ["mixed"])
                    self.assertFalse(r4["stages"][0]["freeze"])

    def test_instance_uses_paper_training_protocol_and_preserves_seed_selection(self):
        config = CONFIGS / "instance_paper_15ep.json"
        spec = load_experiment(config)
        self.assertEqual(spec["evaluation"]["all_data_seeds"], [42, 123, 456, 789])
        self.assertEqual(spec["evaluation"]["paper_table_data_seeds"], [42, 123, 789])
        result = execute(spec, config, "/data", "/base", "/out", dry_run=True)
        self.assertEqual(len(result["jobs"]), 8)
        for job in result["jobs"]:
            plan = job["plan"]
            self.assertEqual(plan["i2i_seed"], plan["seeds"][0])
            self.assertIsNone(plan["stage1_checkpoint"])
            self.assertEqual(plan["requested_i2t_budget"], 15000)
            self.assertEqual(plan["stages"][1]["dataset_config"]["i2t"]["num_used_data"], 1000)
            self.assertEqual(plan["stages"][1]["budget"], 15040)
            self.assertEqual([stage["condition_dropout"] for stage in plan["stages"]], [0.1, 0.0])
            for command in job["commands"]:
                start = command.index(str(BAGEL / "train/pretrain_unified_navit.py")) + 1
                flags = dict(zip(command[start::2], command[start + 1::2]))
                for flag in ("--freeze_shared_for_i2i", "--freeze_vit", "--freeze_und", "--freeze_llm_input_ln"):
                    self.assertEqual(flags[flag], "False")
                self.assertNotIn("--freeze_llm_layers_ratio_from_end", flags)
                self.assertEqual(flags["--freeze_vae"], "True")

    def test_shared_stage1_resolves_saved_checkpoint_and_reuses_it(self):
        spec = load_experiment(CONFIGS / "controlled_scaling.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            for name in ("llm_config.json", "vit_config.json", "ema.safetensors", "ae.safetensors"):
                (base / name).write_text("fixture")
            data = root / "data/jigsaw/train"
            data.mkdir(parents=True)
            for name in ("i2i.jsonl", "i2t.jsonl"):
                (data / name).write_text('{"uid":"fixture"}\n')

            def train_fixture(argv, **kwargs):
                start = argv.index(str(BAGEL / "train/pretrain_unified_navit.py")) + 1
                flags = dict(zip(argv[start::2], argv[start + 1::2]))
                checkpoint = Path(flags["--checkpoint_dir"]) / "arbitrary_actual_checkpoint"
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.safetensors").write_text("trained")
                logs = Path(flags["--results_dir"])
                logs.mkdir()
                (logs / "completion.json").write_text(json.dumps({"checkpoint": str(checkpoint),
                    "sample_visits": int(flags["--total_data_num"]), "optimizer_updates": 1,
                    "pending_microbatches": 0}))

            arguments = (spec, CONFIGS / "controlled_scaling.json", root / "data", base, root / "runs")
            with patch("omnitaskonomy.train.subprocess.run", side_effect=train_fixture):
                execute(*arguments, jobs=["jigsaw_r2_3k"])
            result_dir = root / "runs/controlled_scaling/jigsaw_r2_3k"
            records = json.loads((result_dir / "checkpoints.json").read_text())["records"]
            self.assertEqual([record["seed"] for record in records], [42, 123, 456])
            self.assertEqual(len({record["initialization"] for record in records}), 1)
            self.assertEqual(len({record["stage1_checkpoint_sha256"] for record in records}), 1)
            self.assertTrue(records[0]["initialization"].endswith("arbitrary_actual_checkpoint"))
            with patch("omnitaskonomy.train.subprocess.run", side_effect=AssertionError("completed experiment relaunched")):
                execute(*arguments, jobs=["jigsaw_r2_3k"])

    def test_i2t_scaling_holds_visits_fixed_and_labels_three_seed_extension(self):
        for name, seeds in [("controlled_i2t_scaling_historical", [42, 123]),
                            ("controlled_i2t_scaling_three_seed", [42, 123, 456])]:
            spec = load_experiment(CONFIGS / f"{name}.json")
            result = execute(spec, CONFIGS / f"{name}.json", "/data", "/base", "/out", dry_run=True)
            final_plans = [job["plan"] for job in result["jobs"] if not job["plan"]["stop_after_stage1"]]
            self.assertEqual(len(final_plans), 16)
            for plan in final_plans:
                self.assertEqual(plan["requested_i2t_budget"], 30000)
                self.assertEqual(plan["stages"][-1]["budget"], 30016)
                self.assertEqual(plan["seeds"], seeds)

    def test_gradient_diagnostic_checkpoints_are_independent_literal_budgets(self):
        config = CONFIGS / "gradient_checkpoints.json"
        spec = load_experiment(config)
        preview = execute(spec, config, "/data", "/base", "/out", dry_run=True)
        self.assertEqual(len(preview["jobs"]), 6)
        self.assertEqual({job["plan"]["requested_i2i_budget"] for job in preview["jobs"]}, {3000, 10000, 30000})
        for job in preview["jobs"]:
            plan = job["plan"]
            self.assertIsNone(plan["stage1_checkpoint"])
            self.assertEqual([stage["name"] for stage in plan["stages"]], ["i2i"])
            self.assertTrue(plan["stages"][0]["freeze_last_half_llm"])
            self.assertEqual(plan["stages"][0]["budget"], ((plan["requested_i2i_budget"] + 63) // 64) * 64)

    def test_cli_job_selection_and_unknown_dependency_fail_before_writes(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["--config", str(CONFIGS / "transfer.json"), "--model-path", "/absent",
                  "--jobs", "normal", "--dry-run"])
        self.assertEqual(json.loads(capture.getvalue())["selected_jobs"], ["normal_stage1", "normal"])
        with self.assertRaisesRegex(ValueError, "Unknown experiment jobs"):
            select_jobs(load_experiment(CONFIGS / "transfer.json"), ["missing"])
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.json"
            config.write_text(json.dumps({"schema_version": 1, "id": "fixture", "training": {},
                                          "jobs": [{"id": "later", "stage1_from": "missing"}]}))
            with self.assertRaisesRegex(ValueError, "earlier job"):
                load_experiment(config)


if __name__ == "__main__":
    unittest.main()
