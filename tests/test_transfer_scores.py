import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from omnitaskonomy.analysis.vlmeval_scores import compare_questions, read_scores
from omnitaskonomy.data.common import sha256

ROOT = Path(__file__).resolve().parents[1]


class ScoreReaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, rows, name="scores.csv"):
        path = self.root / name
        frame = pd.DataFrame(rows)
        if path.suffix == ".xlsx":
            frame.to_excel(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return path

    @staticmethod
    def row(index="001", **changes):
        return {"index": index, "hit": 1, "question": "Which  color\nis shown?",
                "answer": " A ", "A": "red\tobject", "B": "NA", **changes}

    def test_excel_and_csv_preserve_string_ids_and_normalize_only_historical_fields(self):
        indexes = ["001", "1.0", "CV-Bench-2D::COCO_000001", "NA", "1.00", "-2.0"]
        for suffix in (".xlsx", ".csv"):
            with self.subTest(format=suffix):
                path = self.write([self.row(index, C="") for index in indexes], "scores" + suffix)
                scores = read_scores(path, expected_sha256=sha256(path))
                self.assertEqual(list(scores["rows"]), ["001", "1", "CV-Bench-2D::COCO_000001", "NA", "1.00", "-2"])
                self.assertEqual(scores["path"], str(path.resolve()))
                self.assertEqual(scores["sha256"], sha256(path))
                self.assertEqual(scores["rows"]["001"], {
                    "hit": 1, "question": "Which color is shown?", "answer": "A",
                    "options": {"A": "red object", "B": "NA", "C": ""},
                })

    def test_comparison_covers_full_population_and_option_schema(self):
        reference = read_scores(self.write([self.row("01"), self.row("02")]))["rows"]
        candidate = read_scores(self.write([
            self.row("02", question="Which color is shown?", hit=0),
            self.row("01", A="red object", log="Different scoring log"),
        ], "candidate.xlsx"))["rows"]
        compare_questions(reference, candidate)
        for field, replacement in [("question", "Different?"), ("answer", "B"),
                                   ("options", {"A": "red object", "B": "NA", "C": ""})]:
            changed = copy.deepcopy(candidate)
            changed["01"][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "UID '01'"):
                compare_questions(reference, changed)
        for changed in ({"01": candidate["01"]}, {**candidate, "03": candidate["01"]}):
            with self.assertRaisesRegex(ValueError, "Question-set drift.*0[23]"):
                compare_questions(reference, changed)

    def test_duplicate_and_empty_indexes_are_rejected(self):
        for rows, error in [([self.row("1"), self.row(" 1.0 ")], "Duplicate.*'1'"),
                            ([self.row("")], "Empty question index"),
                            ([self.row(" \t")], "Empty question index")]:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, error):
                read_scores(self.write(rows))

    def test_missing_fields_and_nonbinary_hits_fail_without_using_predictions(self):
        for field in ("index", "hit", "question", "answer"):
            row = self.row(prediction="A")
            del row[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Missing score columns"):
                read_scores(self.write([row]))
        for hit in (None, "", "not a score", -1, 0.5, 2, float("inf")):
            with self.subTest(hit=hit), self.assertRaisesRegex(ValueError, "Nonbinary hit.*'001'"):
                read_scores(self.write([self.row(hit=hit, prediction="A")]))
        for field in ("question", "answer"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Missing question or answer.*'001'"):
                read_scores(self.write([self.row(**{field: " \t"})]))

    def test_logs_are_preserved_and_never_override_stored_hits(self):
        logs = ["Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.",
                "Failed to predict, thus randomly generate one. "]
        result = read_scores(self.write([self.row(str(i), hit=i, log=log, prediction="A")
                                         for i, log in enumerate(logs)]))
        self.assertEqual([row["hit"] for row in result["rows"].values()], [0, 1])
        self.assertEqual([row["log"] for row in result["rows"].values()], logs)

    def test_hash_pins_and_changes_during_read_are_rejected(self):
        path = self.write([self.row()])
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            read_scores(path, expected_sha256="0" * 64)
        original_reader = pd.read_csv

        def read_then_change(*args, **kwargs):
            frame = original_reader(*args, **kwargs)
            path.write_text(path.read_text() + "\n")
            return frame

        with patch("omnitaskonomy.analysis.vlmeval_scores.pd.read_csv", side_effect=read_then_change):
            with self.assertRaisesRegex(ValueError, "changed while reading"):
                read_scores(path)

    def test_unknown_format_and_empty_workbook_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported score format"):
            read_scores(self.root / "scores.json")
        path = self.root / "empty.xlsx"
        pd.DataFrame(columns=["index", "hit", "question", "answer"]).to_excel(path, index=False)
        with self.assertRaisesRegex(ValueError, "No score rows"):
            read_scores(path)


class HistoricalWorkbookTests(unittest.TestCase):
    def test_real_mmvp_and_cvbench_scorers_produce_readable_per_question_workbooks(self):
        sys.path.insert(0, str(ROOT / "VLMEvalKit"))
        from vlmeval.dataset.image_mcq import CVBench, ImageMCQDataset
        from vlmeval.smp import dump, get_intermediate_file_path

        with tempfile.TemporaryDirectory() as directory:
            for name, cls in (("MMVP", ImageMCQDataset), ("CV-Bench-2D", CVBench)):
                with self.subTest(benchmark=name):
                    rows = []
                    for index, prediction in enumerate(("A", "B", "undecidable", "A")):
                        row = {"index": index if name == "MMVP" else f"COCO_00000{index}",
                               "question": f"Which color is shown in question {index}?",
                               "answer": "A", "A": "red", "B": "blue",
                               "prediction": prediction, "split": "test"}
                        if name == "CV-Bench-2D":
                            row.update(split="2D", source="COCO" if index < 3 else "ADE20K")
                        rows.append(row)
                    data = pd.DataFrame(rows)
                    dataset = cls.__new__(cls)
                    dataset.dataset_name = name
                    dataset.data = data.drop(columns="prediction")
                    predictions = Path(directory) / f"fixture_{name}.xlsx"
                    dump(data, str(predictions))
                    dataset.evaluate(str(predictions), model="exact_matching", nproc=1)
                    workbook = Path(get_intermediate_file_path(str(predictions), "_exact_matching_result"))
                    scores = read_scores(workbook, expected_sha256=sha256(workbook))
                    self.assertEqual(len(scores["rows"]), 4)
                    for source, hit in zip(rows, (1, 0, 0, 1)):
                        self.assertEqual(scores["rows"][str(source["index"])]["hit"], hit)
                    unmatched = scores["rows"][str(rows[2]["index"])]
                    self.assertIn("exact_matching", unmatched["log"])
                    self.assertEqual(unmatched["hit"], 0)
                    csv_path = Path(directory) / f"{name}.csv"
                    pd.read_excel(workbook, dtype=object, keep_default_na=False).to_csv(csv_path, index=False)
                    compare_questions(scores["rows"], read_scores(csv_path)["rows"])


if __name__ == "__main__":
    unittest.main()
