"""CPU integration with the experimental scorers; no benchmark downloads."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from omnitaskonomy.evaluate import (
    BENCHMARK_CONFIGS, MODEL_NAME, ROOT, aggregate_results, read_status,
)
from omnitaskonomy.eval_status import record_evaluation


class VLMEvalScoringIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "VLMEvalKit"))
        import pandas as pd
        from vlmeval.dataset.image_mcq import CVBench, ImageMCQDataset
        from vlmeval.dataset.utils.multiple_choice import MMT_abbrs
        from vlmeval.smp import dump
        cls.pandas = pd
        cls.classes = {"ImageMCQDataset": ImageMCQDataset, "CVBench": CVBench}
        cls.mmt_categories = list(MMT_abbrs)
        cls.dump = staticmethod(dump)

    def fixture(self, name):
        categories = self.mmt_categories if name == "MMT-Bench_VAL" else [None] * 4
        records = []
        for index, category in enumerate(categories):
            row = {
                "index": index, "question": f"Which color is shown in question {index}?",
                "A": "red", "B": "blue", "answer": "A",
                "prediction": "B" if index % 4 == 3 else "A", "split": "test",
            }
            if name == "MMVP":
                row["question"] = f"Which color is shown in pair {index // 2}?"
            elif name == "MMT-Bench_VAL":
                # Its real reporter expects all 32 top-level categories to be present.
                row.update({"category": f"subtask_{index}", "l2-category": category})
            elif name == "CV-Bench-2D":
                row.update({"split": "2D", "source": "COCO" if index < 3 else "ADE20K"})
            elif name == "CV-Bench-3D":
                row.update({"split": "3D", "source": "Omni3D"})
            records.append(row)
        return self.pandas.DataFrame(records)

    def test_all_eight_real_evaluators_write_status_consumable_by_release(self):
        expected = {
            "MMVP": ("split=test|Overall", 0.75),
            "CV-Bench-2D": ("split=2D|Overall", 0.5),
            "CV-Bench-3D": ("split=3D|Overall", 0.75),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"model": {MODEL_NAME: {"seed": 42}}, "data": BENCHMARK_CONFIGS}
            (root / "vlmeval.json").write_text(json.dumps(config))
            run = root / "results" / MODEL_NAME / "fixture-run"
            run.mkdir(parents=True)
            for name, spec in BENCHMARK_CONFIGS.items():
                with self.subTest(benchmark=name):
                    data = self.fixture(name)
                    cls = self.classes[spec["class"]]
                    self.assertIn(name, cls.DATASET_URL)
                    dataset = cls.__new__(cls)
                    dataset.dataset_name = name
                    dataset.data = data.drop(columns="prediction")
                    predictions = run / f"{MODEL_NAME}_{name}.xlsx"
                    self.dump(data, str(predictions))
                    scores = dataset.evaluate(str(predictions), model="exact_matching", nproc=1)
                    metric, value = expected.get(name, ("split=test|Overall", 0.75))
                    status = record_evaluation(
                        run, name, "done", scores=scores, judge="exact_matching",
                    )
                    self.assertEqual(status["datasets"][name]["metrics"][metric], value)
                    if name == "MMVP":
                        self.assertNotIn("Average", scores.columns)
                        self.assertEqual(scores.iloc[0]["Overall"], 0.75)
                    if name == "CV-Bench-2D":
                        self.assertEqual(scores.iloc[0]["COCO"], 1.0)
                        self.assertEqual(scores.iloc[0]["ADE20K"], 0.0)
            plan = {"directory": str(root), "config": config, "seed": 42,
                    "checkpoint": "/fixture/model.safetensors", "judge": "exact_matching",
                    "vlmeval_source_sha256": "fixture-source"}
            (root / "evaluation.json").write_text(json.dumps(plan))
            result = read_status(plan, list(BENCHMARK_CONFIGS))
            self.assertEqual(set(result["metrics"]), set(BENCHMARK_CONFIGS))
            summary = aggregate_results([result])
            self.assertEqual(summary["benchmarks"]["MMVP"]["split=test|Overall"]["mean"], 0.75)
            self.assertEqual(summary["benchmarks"]["CV-Bench-2D"]["split=2D|Overall"]["mean"], 0.5)

    def test_receipts_clear_stale_metrics_and_preserve_other_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record_evaluation(run, "BLINK", "done", judge="exact_matching", scores={"Overall": 0.75})
            record_evaluation(run, "MMVP", "done", judge="exact_matching", scores={"Overall": 0.5})
            record_evaluation(run, "MMVP", "running")
            state = json.loads((run / "status.json").read_text())
            self.assertEqual(state["datasets"]["MMVP"], {"status": "running"})
            self.assertEqual(state["datasets"]["BLINK"]["metrics"], {"Overall": 0.75})
            record_evaluation(run, "MMVP", "failed", error_message=ValueError("prediction failed"))
            state = json.loads((run / "status.json").read_text())
            self.assertEqual(state["datasets"]["MMVP"]["error_message"], "prediction failed")
            record_evaluation(run, "MMVP", "done", skip_reason="mode_infer")
            state = json.loads((run / "status.json").read_text())
            self.assertEqual(state["datasets"]["MMVP"], {"status": "done", "skip_reason": "mode_infer"})
            self.assertFalse((run / "status.tmp").exists())

    def test_split_metrics_and_invalid_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = self.pandas.DataFrame([
                {"split": "test", "Overall": 0.75, "category": float("nan")},
                {"split": "ALL", "Overall": 0.5, "category": 1.0},
            ])
            state = record_evaluation(directory, "MMT-Bench_VAL", "done", scores=scores)
            self.assertEqual(state["datasets"]["MMT-Bench_VAL"]["metrics"], {
                "split=test|Overall": 0.75, "split=ALL|Overall": 0.5, "split=ALL|category": 1.0,
            })
            path = Path(directory) / "status.json"
            previous = path.read_text()
            for scores, error in [
                ({"Overall": float("inf")}, ValueError),
                ({"Overall": "not a score"}, TypeError),
                (self.pandas.DataFrame([{"Overall": 0.5}, {"Overall": 0.75}]), ValueError),
            ]:
                with self.subTest(scores=scores), self.assertRaises(error):
                    record_evaluation(directory, "MMVP", "done", scores=scores)
                self.assertEqual(path.read_text(), previous)

    def test_experimental_exact_matching_keeps_its_original_answer_parser(self):
        from vlmeval.utils.matching_util import can_infer_option

        choices = {"A": "red", "B": "blue"}
        self.assertEqual(can_infer_option("B", choices), "B")
        self.assertFalse(can_infer_option(
            "The correct answer is B because this is followed by a long explanation.", choices,
        ))


if __name__ == "__main__":
    unittest.main()
