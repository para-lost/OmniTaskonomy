import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from omnitaskonomy.data.common import sha256
from omnitaskonomy.examples.tiny_umm import TinyUMM, create_adapter
from omnitaskonomy.experiments import execute, load_experiment
from omnitaskonomy.train import build_plan, main, parser, preview_commands, run
from omnitaskonomy.umm import LossContext


ADAPTER = "omnitaskonomy.examples.tiny_umm:create_adapter"


class UMMTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.base = self.root / "base"
        self.base.mkdir()
        torch.manual_seed(42)
        torch.save(TinyUMM().state_dict(), self.base / "model.pt")
        Image.new("RGB", (4, 4), (20, 80, 140)).save(self.root / "image.png")
        for objective in ("i2i", "i2t"):
            rows = []
            for index in range(32):
                row = {"uid": f"{objective}-{index}"}
                if objective == "i2i":
                    row.update(source_image="image.png", target_image="image.png", prompt="Restore.")
                else:
                    row.update(image="image.png", conversations=[
                        {"from": "human", "value": "<image>\nChoose."}, {"from": "gpt", "value": "A"},
                    ])
                rows.append(row)
            (self.root / f"{objective}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def plan(self, recipe="r2", suite="controlled", output="run", extra=()):
        return build_plan(parser().parse_args([
            "--task", "tiny", "--recipe", recipe, "--suite", suite,
            "--i2i-manifest", str(self.root / "i2i.jsonl"),
            "--i2t-manifest", str(self.root / "i2t.jsonl"),
            "--i2i-pool", "16", "--i2t-pool", "8",
            "--i2i-budget", "1876" if suite == "controlled" else "8",
            "--i2t-budget", "128" if suite == "controlled" else "8",
            "--batch-size", "64" if suite == "controlled" else "2",
            "--gradient-accumulation", "1" if suite == "controlled" else "2",
            "--condition-dropout", "0", "--model-path", str(self.base),
            "--output-dir", str(self.root / output), "--adapter", ADAPTER,
            "--device", "cpu", "--nproc-per-node", "1", *extra,
        ]))

    @staticmethod
    def weights(checkpoint):
        return torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=True)

    def test_all_six_recipes_execute_and_r4_unfreezes_second_stage(self):
        base = self.weights(self.base)
        expected_stages = {"r1": ["i2t"], "r2": ["i2i", "i2t"], "r3": ["mixed", "i2t"],
                           "r4": ["i2i", "mixed"], "r5": ["mixed"], "r6": ["i2i", "mixed"]}
        for recipe, expected in expected_stages.items():
            with self.subTest(recipe=recipe):
                extra = ("--condition-dropout", "0.1") if recipe == "r3" else ()
                with torch.no_grad():
                    records = json.loads(run(self.plan(recipe, output=recipe, extra=extra)).read_text())["records"]
                self.assertEqual([row["stage"] for row in records], expected)
                for record in records:
                    self.assertEqual(record["adapter"], ADAPTER)
                    self.assertEqual(record["pending_microbatches"], 0)
                    self.assertTrue((Path(record["checkpoint"]) / "model.pt").is_file())
                    torch.testing.assert_close(self.weights(record["checkpoint"])["scale"], base["scale"], rtol=0, atol=0)
                if recipe == "r4":
                    first, final = [self.weights(record["checkpoint"]) for record in records]
                    for name in base:
                        self.assertEqual(not torch.equal(first[name], base[name]), name.startswith("generation."))
                    for name in ("shared.weight", "understanding.weight"):
                        self.assertFalse(torch.equal(first[name], final[name]))
                    policies = [json.loads(Path(row["trainable_parameters"]).read_text())["policy"]
                                for row in records]
                    self.assertEqual(policies, ["generation", "all"])
                if recipe == "r3":
                    phases = [json.loads(Path(record["trainable_parameters"]).with_name(
                        "completion.json").read_text())["phases"][0] for record in records]
                    self.assertEqual([(phase["optimizer_updates_before"], phase["optimizer_updates_after"])
                                      for phase in phases], [(0, 1), (0, 1)])
                    self.assertEqual([phase["condition_dropout"] for phase in phases], [0.1, 0.0])
                    # Each one-update stage starts warmup at zero learning rate.
                    torch.testing.assert_close(self.weights(records[-1]["checkpoint"])["shared.weight"],
                                               base["shared.weight"], rtol=0, atol=0)

    def test_r3_reloads_mixed_weights_and_restarts_optimizer_and_warmup(self):
        plan = self.plan("r3", extra=("--i2t-budget", "256", "--condition-dropout", "0.1"))
        with patch("omnitaskonomy.umm_training.torch.optim.AdamW", wraps=torch.optim.AdamW) as optimizer:
            records = json.loads(run(plan).read_text())["records"]
        self.assertEqual(optimizer.call_count, 2)
        first, final = records
        self.assertEqual(final["initialization"], first["checkpoint"])
        self.assertEqual([row["sample_visits"] for row in records], [128, 128])
        self.assertEqual([row["optimizer_updates"] for row in records], [2, 2])
        self.assertEqual([row["objective_visits"] for row in records],
                         [{"i2i": 16, "i2t": 128}, {"i2i": 0, "i2t": 128}])
        for record in records:
            log = Path(record["trainable_parameters"]).with_name("progress.jsonl")
            progress = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row["next_learning_rate"] for row in progress],
                             [plan["learning_rate"] / 8, plan["learning_rate"] * 2 / 8])
        mixed_weights = self.weights(first["checkpoint"])
        final_weights = self.weights(final["checkpoint"])
        self.assertFalse(torch.equal(mixed_weights["shared.weight"], self.weights(self.base)["shared.weight"]))
        self.assertFalse(torch.equal(final_weights["shared.weight"], mixed_weights["shared.weight"]))
        torch.testing.assert_close(final_weights["generation.weight"], mixed_weights["generation.weight"],
                                   rtol=0, atol=0)

    @torch.enable_grad()
    def test_mixed_objectives_have_equal_weight_despite_different_batch_sizes(self):
        plan = self.plan("r5")
        records = json.loads(run(plan).read_text())["records"]
        self.assertEqual(records[0]["objective_visits"], {"i2i": 8, "i2t": 128})
        adapter = create_adapter(model_path=self.base, device="cpu")
        optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=0, betas=(0.9, 0.95),
                                      eps=1e-15, weight_decay=0)
        i2i = json.loads((self.root / "i2i.jsonl").read_text().splitlines()[0])
        i2t = json.loads((self.root / "i2t.jsonl").read_text().splitlines()[0])
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            losses = [adapter.loss([row], objective, LossContext(self.root / f"{objective}.jsonl", 42)).mean
                      for objective, row in (("i2i", i2i), ("i2t", i2t))]
            sum(losses).backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            optimizer.param_groups[0]["lr"] = plan["learning_rate"] * step / 8
            optimizer.step()
        for name, actual in self.weights(records[0]["checkpoint"]).items():
            torch.testing.assert_close(actual, adapter.model.state_dict()[name], rtol=0, atol=1e-7)

    def test_pools_follow_suite_seed_rules_and_completed_runs_resume(self):
        plans = [self.plan("r1", suite, suite, ("--seeds", "42", "123"))
                 for suite in ("controlled", "transfer")]
        for plan in plans:
            result = run(plan)
            original = result.read_bytes()
            self.assertEqual(run(plan).read_bytes(), original)
            pools = []
            for record in json.loads(original)["records"]:
                completion = Path(record["trainable_parameters"]).with_name("completion.json")
                pools.append(json.loads(completion.read_text())["phases"][0]["pool_uids"]["i2t"])
            self.assertNotEqual(pools[0], pools[1])
            if plan["suite"] == "transfer":
                self.assertEqual(set(pools[0]), {f"i2t-{index}" for index in range(8)})
                self.assertEqual(set(pools[0]), set(pools[1]))
            else:
                self.assertNotEqual(set(pools[0]), set(pools[1]))
        record = json.loads(result.read_text())["records"][0]
        with (Path(record["checkpoint"]) / "model.pt").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Saved checkpoint files changed"):
            run(plans[-1])

    def test_transfer_presets_reuse_adapter_checkpoint_directories(self):
        spec = {"schema_version": 1, "id": "fixture", "paper_commit": "fixture",
                "training": {"suite": "transfer", "task": "tiny", "recipe": "r2", "seeds": [42],
                             "i2i_manifest": "i2i.jsonl", "i2t_manifest": "i2t.jsonl", "i2i_budget": 8,
                             "i2t_budget": 8, "i2i_pool": 16, "i2t_pool": 8, "batch_size": 2,
                             "gradient_accumulation": 2},
                "jobs": [{"id": "source", "training": {"stop_after_stage1": True}},
                         {"id": "final", "stage1_from": "source", "training": {}}]}
        config = self.root / "experiment.json"
        config.write_text(json.dumps(spec))
        execute(load_experiment(config), config, self.root, self.base, self.root / "out",
                adapter=ADAPTER, device="cpu", nproc=1)
        source = json.loads((self.root / "out/fixture/source/checkpoints.json").read_text())["records"][0]
        final = json.loads((self.root / "out/fixture/final/checkpoints.json").read_text())["records"][0]
        self.assertEqual(final["initialization"], source["checkpoint"])
        self.assertEqual(final["stage"], "i2t")
        self.assertEqual(final["sample_visits"], 8)

    def test_missing_taskonomy_manifest_is_prepared_before_adapter_training(self):
        manifest = self.root / "normal.jsonl"
        plan = self.plan("r2", "transfer", extra=("--task", "normal", "--i2i-manifest", str(manifest)))

        def prepare(task, output):
            self.assertEqual(task, "normal")
            shutil.copyfile(self.root / "i2i.jsonl", output)
            return output

        with patch("omnitaskonomy.data.taskonomy.prepare_taskonomy", side_effect=prepare):
            records = json.loads(run(plan).read_text())["records"]
        self.assertEqual([row["stage"] for row in records], ["i2i", "i2t"])
        self.assertEqual(records[0]["i2i_manifest"], str(manifest))
        completion = Path(records[0]["trainable_parameters"]).with_name("completion.json")
        pool = json.loads(completion.read_text())["phases"][0]["pool_uids"]["i2i"]
        self.assertEqual(set(pool), {f"i2i-{index}" for index in range(16)})
        trained = self.weights(records[0]["checkpoint"])
        self.assertFalse(torch.equal(trained["generation.weight"], self.weights(self.base)["generation.weight"]))

    def test_existing_taskonomy_manifest_is_used_unchanged(self):
        manifest = self.root / "i2i.jsonl"
        original = manifest.read_bytes()
        plan = self.plan("r2", "transfer", extra=("--task", "normal",))
        with patch("omnitaskonomy.data.taskonomy.prepare_taskonomy", side_effect=AssertionError("unexpected preparation")):
            records = json.loads(run(plan).read_text())["records"]
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(records[0]["objective_visits"], {"i2i": 8, "i2t": 0})

    def test_missing_recipe_manifests_feed_training_and_provenance(self):
        for recipe in ("r1", "r4"):
            with self.subTest(recipe=recipe):
                manifests = {kind: self.root / f"{recipe}-{kind}.jsonl" for kind in ("i2i", "i2t")}
                plan = self.plan(recipe, output=recipe, extra=(
                    "--task", "jigsaw", "--i2i-manifest", str(manifests["i2i"]),
                    "--i2t-manifest", str(manifests["i2t"])))

                def prepare(task, paths):
                    self.assertEqual(task, "jigsaw")
                    self.assertEqual(paths, manifests)
                    for kind, path in paths.items():
                        shutil.copyfile(self.root / f"{kind}.jsonl", path)
                    return paths

                with patch("omnitaskonomy.data.recipe.prepare_recipe", side_effect=prepare):
                    result = run(plan)
                records = json.loads(result.read_text())["records"]
                self.assertEqual([record["stage"] for record in records],
                                 ["i2t"] if recipe == "r1" else ["i2i", "mixed"])
                provenance = json.loads((self.root / recipe / "plan.json").read_text())
                self.assertEqual(provenance["manifest_sha256"],
                                 {f"{kind}_manifest": sha256(path) for kind, path in manifests.items()})
                final = self.weights(records[-1]["checkpoint"])
                self.assertFalse(torch.equal(final["understanding.weight"],
                                             self.weights(self.base)["understanding.weight"]))
                if recipe == "r4":
                    first = self.weights(records[0]["checkpoint"])
                    self.assertFalse(torch.equal(first["generation.weight"],
                                                 self.weights(self.base)["generation.weight"]))
                    torch.testing.assert_close(first["shared.weight"], self.weights(self.base)["shared.weight"])

                with patch("omnitaskonomy.data.recipe.prepare_recipe", side_effect=AssertionError("unexpected preparation")):
                    self.assertEqual(run(plan).read_bytes(), result.read_bytes())

    def test_recipe_dry_run_does_not_prepare_missing_data(self):
        plan = self.plan(extra=("--task", "zoomin", "--i2i-manifest", str(self.root / "missing-i2i.jsonl"),
                                "--i2t-manifest", str(self.root / "missing-i2t.jsonl")))
        output = io.StringIO()
        with patch("omnitaskonomy.data.recipe.prepare_recipe", side_effect=AssertionError("unexpected preparation")):
            with contextlib.redirect_stdout(output):
                main(preview_commands(plan)[0][2:] + ["--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["plan"], plan)
        self.assertFalse((self.root / "missing-i2i.jsonl").exists())
        self.assertFalse((self.root / "missing-i2t.jsonl").exists())
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_taskonomy_preparation_is_not_used_for_controlled_or_custom_data(self):
        for suite, task in (("controlled", "tiny"), ("transfer", "tiny"), ("transfer", "normal")):
            recipe = "r1" if task == "normal" else "r2"
            with self.subTest(suite=suite, task=task):
                plan = self.plan(recipe, suite, output=task, extra=(
                    "--task", task, "--i2i-manifest", str(self.root / "missing.jsonl")))
                with patch("omnitaskonomy.data.taskonomy.prepare_taskonomy", side_effect=AssertionError("unexpected preparation")):
                    with self.assertRaises(FileNotFoundError):
                        run(plan)
                self.assertFalse(Path(plan["output_dir"]).exists())

    def test_recipe_preparation_does_not_replace_transfer_pools(self):
        for task in ("jigsaw", "counting"):
            with self.subTest(task=task):
                plan = self.plan("r2", "transfer", output=task, extra=(
                    "--task", task, "--i2i-manifest", str(self.root / "missing.jsonl")))
                with patch("omnitaskonomy.data.recipe.prepare_recipe", side_effect=AssertionError("unexpected preparation")):
                    with self.assertRaises(FileNotFoundError):
                        run(plan)
                self.assertFalse(Path(plan["output_dir"]).exists())

    def test_taskonomy_dry_run_does_not_prepare_missing_data(self):
        manifest = self.root / "normal.jsonl"
        plan = self.plan("r2", "transfer", extra=("--task", "normal", "--i2i-manifest", str(manifest)))
        output = io.StringIO()
        with patch("omnitaskonomy.data.taskonomy.prepare_taskonomy", side_effect=AssertionError("unexpected preparation")):
            with contextlib.redirect_stdout(output):
                main(preview_commands(plan)[0][2:] + ["--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["plan"], plan)
        self.assertFalse(manifest.exists())
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_partial_accumulation_is_reported_without_an_extra_update(self):
        short = json.loads(run(self.plan("r1", "transfer", "short")).read_text())["records"][0]
        partial = json.loads(run(self.plan("r1", "transfer", "partial",
                                           ("--i2t-budget", "10"))).read_text())["records"][0]
        self.assertEqual((partial["sample_visits"], partial["optimizer_updates"], partial["pending_microbatches"]),
                         (10, 2, 1))
        partial_weights = self.weights(partial["checkpoint"])
        for name, parameter in self.weights(short["checkpoint"]).items():
            torch.testing.assert_close(parameter, partial_weights[name], rtol=0, atol=0)

    def test_preview_does_not_load_adapter_and_rejects_unsupported_flags(self):
        plan = self.plan(extra=("--adapter", "absent.module:create_adapter", "--seeds", "42", "123",
                               "--model-seeds", "4396", "4397"))
        command = preview_commands(plan)[0]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(command[2:] + ["--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["plan"], plan)
        self.assertFalse(Path(plan["output_dir"]).exists())
        for flag in ("--cpu-offload", "--freeze-input-ln", "--stage1-freeze-last-half-llm"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "does not support BAGEL"):
                self.plan(extra=(flag,))
        with self.assertRaisesRegex(ValueError, "nproc-per-node 1"):
            self.plan(extra=("--nproc-per-node", "2"))
        with self.assertRaisesRegex(ValueError, "select custom weights through adapter options"):
            self.plan(extra=("--base-weights", "model"))


if __name__ == "__main__":
    unittest.main()
