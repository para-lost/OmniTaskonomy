from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from omnitaskonomy.data.common import sha256
from omnitaskonomy.taxonomy import load_taxonomy


class FrozenTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy()

    def test_public_task_identities_and_v15_sources(self):
        tasks = {task["id"]: task for task in self.taxonomy["tasks"]}
        self.assertEqual(Counter(task["modality"] for task in tasks.values()), {"i2i": 19, "i2t": 25})
        self.assertEqual({task["source_id"] for task in tasks.values() if task["modality"] == "i2i"}, {
            "category", "appearance", "colorization", "depth_zbuffer", "depth_euclidean",
            "normal", "principal_curvature", "edge_occlusion", "keypoints3d", "reshading",
            "inpainting", "semseg", "edge_texture", "keypoints2d", "segment_unsup2d",
            "segment_unsup25d", "counting", "jigsaw", "localization",
        })
        self.assertEqual(tasks["i2i:localization"]["name"], tasks["i2t:REFERENTIAL_LOCALIZATION"]["name"])
        self.assertEqual(tasks["i2i:semantic_segmentation"]["family"], "RORG")
        self.assertEqual(tasks["i2t:SEMANTIC_SCENE_PARSING"]["family"], "REC")
        self.assertEqual(tasks["i2i:object_replacement"]["source_id"], "category")
        self.assertIn("Add objects", tasks["i2i:object_replacement"]["definition"])
        self.assertIn("replace", tasks["i2i:object_replacement"]["definition"])
        self.assertIn("grayscale", tasks["i2i:colorization"]["definition"])

    def test_frozen_cohorts_and_stable_question_assignments(self):
        retained, excluded = self.taxonomy["retained"], self.taxonomy["excluded"]
        self.assertEqual((len(retained), len(excluded)), (9444, 978))
        self.assertEqual(Counter(row["benchmark"] for row in retained), {
            "BLINK": 1758, "CV-Bench-2D": 1434, "CV-Bench-3D": 1200, "MMStar": 986,
            "MMT-Bench_VAL": 2840, "MMVP": 290, "RealWorldQA": 745, "VStarBench": 191,
        })
        self.assertEqual(Counter(row["family"] for row in retained), {"REC": 3475, "RCN": 3038, "RORG": 2931})
        self.assertEqual(Counter(row["agreement_pattern"] for row in retained), {"AAA": 8811, "AAB": 633})
        self.assertTrue(all(row["agreement_count"] == {"AAA": 3, "AAB": 2}[row["agreement_pattern"]]
                            for row in retained))
        self.assertEqual(Counter(row["exclusion_type"] for row in excluded),
                         {"inherited_v10": 949, "v11_no_leaf_majority": 29})
        supported = [count for count in Counter(row["task_id"] for row in retained).values() if count > 100]
        self.assertEqual((len(supported), sum(supported)), (19, 9145))
        by_uid = {row["uid"]: row for row in retained}
        self.assertEqual(by_uid["MMStar::val::0"]["task_id"], "i2t:QUALITATIVE_SPATIAL_RELATIONS")
        self.assertEqual(by_uid["BLINK::val::val_Art_Style_1"]["task_id"], "i2t:COLOR_MATERIAL")


class TaxonomyValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.catalogue = {
            "schema_version": 1,
            "families": [
                {"id": "REC", "name": "Recognition", "definition": "Recognize visible attributes."},
                {"id": "RORG", "name": "Reorganization", "definition": "Organize visible entities."},
            ],
            "tasks": [
                {"id": "i2i:localization", "name": "Localization", "modality": "i2i",
                 "family": "RORG", "definition": "Mark a referred region.", "source_id": "localization"},
                {"id": "i2t:REFERENTIAL_LOCALIZATION", "name": "Localization", "modality": "i2t",
                 "family": "RORG", "definition": "Locate a referred object."},
                {"id": "i2t:COLOR_MATERIAL", "name": "Appearance", "modality": "i2t",
                 "family": "REC", "definition": "Recognize color and material."},
            ],
        }
        self.retained = [
            {"uid": "Example::val::1", "benchmark": "Example", "index": "1",
             "task_id": "i2t:REFERENTIAL_LOCALIZATION", "family": "RORG"},
            {"uid": "Example::val::01", "benchmark": "Example", "index": "01",
             "task_id": "i2t:COLOR_MATERIAL", "family": "REC"},
            {"uid": "BLINK::val::val_Art_Style_1", "benchmark": "BLINK", "index": "val_Art_Style_1",
             "task_id": "i2t:COLOR_MATERIAL", "family": "REC"},
        ]
        self.excluded = [{"uid": "Example::test::2", "benchmark": "Example", "index": "2",
                          "reason": "No majority assignment."}]

    def write_bundle(self, catalogue=None, retained=None, excluded=None):
        catalogue = self.catalogue if catalogue is None else catalogue
        retained = self.retained if retained is None else retained
        excluded = self.excluded if excluded is None else excluded
        (self.directory / "tasks.json").write_text(json.dumps(catalogue), encoding="utf-8")
        for filename, rows in (("retained_questions.jsonl", retained), ("excluded_questions.jsonl", excluded)):
            (self.directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "files": {name: sha256(self.directory / name)
                      for name in ("tasks.json", "retained_questions.jsonl", "excluded_questions.jsonl")},
            "counts": {"i2i": sum(task["modality"] == "i2i" for task in catalogue["tasks"]),
                       "i2t": sum(task["modality"] == "i2t" for task in catalogue["tasks"]),
                       "retained": len(retained), "excluded": len(excluded)},
        }
        (self.directory / "provenance.json").write_text(json.dumps(receipt), encoding="utf-8")

    def test_small_bundle_preserves_leading_zeroes_and_symbolic_indexes(self):
        self.write_bundle()
        result = load_taxonomy(self.directory)
        self.assertEqual(result["retained"], self.retained)
        self.assertEqual(result["provenance"]["counts"], {"i2i": 1, "i2t": 2, "retained": 3, "excluded": 1})

    def test_rejects_modified_bytes_and_wrong_counts(self):
        for filename in ("tasks.json", "retained_questions.jsonl", "excluded_questions.jsonl"):
            with self.subTest(filename=filename):
                self.write_bundle()
                with (self.directory / filename).open("a", encoding="utf-8") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_taxonomy(self.directory)
        self.write_bundle()
        path = self.directory / "provenance.json"
        receipt = json.loads(path.read_text())
        receipt["counts"]["retained"] = 9444
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "count mismatch for retained"):
            load_taxonomy(self.directory)

    def test_rejects_duplicate_or_invalid_catalogue_identities(self):
        variants = []
        for field, value, message in (("id", "i2t:localization", "typed task ID"),
                                      ("id", "i2i:bad id", "typed task ID"),
                                      ("family", "UNKNOWN", "Unknown family"),
                                      ("source_id", "", "source_id")):
            catalogue = deepcopy(self.catalogue)
            catalogue["tasks"][0][field] = value
            variants.append((catalogue, message))
        for section, message in (("tasks", "Duplicate task ID"), ("families", "duplicate family ID")):
            catalogue = deepcopy(self.catalogue)
            catalogue[section].append(deepcopy(catalogue[section][0]))
            variants.append((catalogue, message))
        catalogue = deepcopy(self.catalogue)
        catalogue["tasks"].append(dict(catalogue["tasks"][0], id="i2i:other"))
        variants.append((catalogue, "duplicate I2I source_id"))
        for catalogue, message in variants:
            with self.subTest(message=message):
                self.write_bundle(catalogue=catalogue)
                with self.assertRaisesRegex(ValueError, message):
                    load_taxonomy(self.directory)

    def test_rejects_non_i2t_routes_and_wrong_families(self):
        for field, value, message in (("task_id", "i2i:localization", "I2T task"),
                                      ("task_id", "i2t:MISSING", "I2T task"),
                                      ("family", "REC", "family differs")):
            with self.subTest(field=field, value=value):
                retained = deepcopy(self.retained)
                retained[0][field] = value
                self.write_bundle(retained=retained)
                with self.assertRaisesRegex(ValueError, message):
                    load_taxonomy(self.directory)

    def test_rejects_question_identity_collisions_and_overlap(self):
        for uid in ("Example::val::1", "Example::test::1"):
            with self.subTest(uid=uid):
                duplicate = dict(self.retained[0], uid=uid)
                self.write_bundle(retained=[*self.retained, duplicate])
                with self.assertRaisesRegex(ValueError, "duplicate question"):
                    load_taxonomy(self.directory)
                self.write_bundle(excluded=[dict(duplicate, reason="Excluded.")])
                with self.assertRaisesRegex(ValueError, "sets overlap"):
                    load_taxonomy(self.directory)

    def test_rejects_unnormalized_or_inconsistent_question_identity(self):
        for index in (1, " 1 ", "1.0", "-01.0", ""):
            with self.subTest(index=index):
                retained = deepcopy(self.retained)
                retained[0].update(index=index, uid=f"Example::val::{index}")
                self.write_bundle(retained=retained)
                with self.assertRaisesRegex(ValueError, "index"):
                    load_taxonomy(self.directory)
        for changes, message in (({"uid": "Example::val::3"}, "UID does not match"),
                                 ({"split": "test"}, "split differs")):
            retained = deepcopy(self.retained)
            retained[0].update(changes)
            self.write_bundle(retained=retained)
            with self.assertRaisesRegex(ValueError, message):
                load_taxonomy(self.directory)
        self.write_bundle(excluded=[dict(self.excluded[0], reason=" ")])
        with self.assertRaisesRegex(ValueError, "needs a reason"):
            load_taxonomy(self.directory)


if __name__ == "__main__":
    unittest.main()
