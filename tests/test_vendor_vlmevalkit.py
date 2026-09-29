"""Exercise the exporter without a research checkout, credentials or model dependencies."""

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SOURCE = runpy.run_path(str(ROOT / "scripts/vendor_vlmevalkit.py"))["export_source"]
PATCHED_FILES = {"vlmeval/config.py", "run.py", "vlmeval/vlm/bagel_vlm.py", "setup.py"}

SETUP = '''from setuptools import setup

with open('README.md', encoding="utf-8") as f:
    readme = f.read()


def do_setup():
    setup(
        name='fixture-vlmevalkit',
        version='0.1.0',
        long_description=readme,
        long_description_content_type='text/markdown',
    )

if __name__ == '__main__':
    do_setup()
'''

CONFIG = '''from functools import partial
from vlmeval.vlm.bagel_vlm import BAGEL
BAGEL_ROOT = None
unrelated_registry = {"fixture": "preserve this value"}
bagel_series = {"private-experiment": partial(BAGEL, model_path="/private/checkpoint")}
o1_key = None
supported_VLM = {**unrelated_registry, **bagel_series}
'''

RUNNER = '''import os
from vlmeval.smp import *

def main():
    for model_name in []:
        eval_id = f"T{date}_G{commit_id}"
        for dataset_name in []:
            try:
                pred_format = get_pred_file_format()
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(reuse=args.reuse)
                if RANK == 0:
                    if args.mode == 'infer':
                        continue
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
            except Exception as e:
                logger.exception("Fixture dataset failed")
'''

BAGEL_ADAPTER = '''import os

class BAGEL:
    def __init__(self, model_path):
        # config (llm_config.json and visual encoders)
        self.ori_model_path = "/private/base-model"
        self.use_cpu_offload = False

    def predict(self, item):
        return item
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_export_integrity(testcase, destination, patched_files=PATCHED_FILES):
    metadata = json.loads((destination / "VENDORED.json").read_text())
    manifest_path = destination / metadata["source_manifest"]
    testcase.assertEqual(metadata["source_manifest_sha256"], digest(manifest_path))
    testcase.assertEqual(metadata["license_sha256"], digest(destination / metadata["license"]))
    testcase.assertEqual(metadata["source_kind"], "tracked working-tree files")
    testcase.assertEqual(set(metadata["changes"]), patched_files)
    records = json.loads(manifest_path.read_text())
    names = [row["path"] for row in records]
    testcase.assertEqual(len(names), len(set(names)))
    changed = set()
    for row in records:
        with testcase.subTest(path=row["path"]):
            relative = Path(row["path"])
            testcase.assertFalse(relative.is_absolute())
            testcase.assertNotIn("..", relative.parts)
            path = destination / relative
            testcase.assertTrue(path.is_file())
            testcase.assertFalse(path.is_symlink())
            testcase.assertEqual(row["release_sha256"], digest(path))
            if row["source_sha256"] != row["release_sha256"]:
                changed.add(row["path"])
            if row["path"] not in patched_files:
                testcase.assertEqual(row["source_sha256"], row["release_sha256"])
    testcase.assertEqual(changed, patched_files)

    files = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination)
        parts = {part.lower() for part in relative.parts}
        # Editable installs and evaluator tests create metadata and bytecode.
        if "__pycache__" in parts or relative.parts[0] == "vlmeval.egg-info":
            continue
        with testcase.subTest(exported_path=str(relative)):
            testcase.assertFalse(parts & {".git", "results", "your_results", "outputs", "weights", "checkpoints"})
            testcase.assertFalse(any(part.startswith(".env") for part in relative.parts))
            testcase.assertNotIn(path.suffix.lower(), {".ipynb", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".md"})
        if path.is_file():
            files.add(str(relative))
    testcase.assertEqual(files, set(names) | {"SOURCE_MANIFEST.json", "VENDORED.json"})
    return records, metadata


class VendorExporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repository = root / "research"
        self.repository.mkdir()
        self.source = self.repository / "VLMEvalKit"
        self.destination = root / "release"
        self.git("init", "--quiet")
        self.allowed = {
            "LICENSE": "Fixture license\n",
            "requirements.txt": "pillow\n",
            "run.py": RUNNER,
            "setup.py": SETUP,
            "vlmeval/config.py": CONFIG,
            "vlmeval/vlm/bagel_vlm.py": BAGEL_ADAPTER,
            "vlmeval/dataset/scorer.py": "def score(answer):\n    return answer == 'A'\n",
            "assets/palette.json": '{"color": [1, 2, 3]}\n',
            "requirements/optional.txt": "numpy\n",
            "vlmeval/vlm/bagel/modeling/model.py": "MODEL_VERSION = 1\n",
            "vlmeval/vlm/bagel/data/data_utils.py": "def image_size(image):\n    return image.size\n",
        }
        self.excluded = {
            "README.md": "# Fixture VLMEvalKit\n",
            "docs/inference.md": "Run the benchmark.\n",
            "docs/palette.json": '{"color": [1, 2, 3]}\n',
            "vlmeval/README.MD": "Nested upstream instructions.\n",
            ".env": "FIXTURE_VALUE=not-a-secret\n",
            "vlmeval/.env.production": "FIXTURE_VALUE=not-a-secret\n",
            "docs/.env.example": "FIXTURE_VALUE=example\n",
            "results/predictions.json": "{}\n",
            "assets/results/predictions.json": "{}\n",
            "vlmeval/results/scores.csv": "score\n1\n",
            "docs/your_results/scores.json": "{}\n",
            "vlmeval/outputs/predictions.json": "{}\n",
            "assets/weights/model.bin": "fixture model bytes\n",
            "vlmeval/weights/README.md": "Private model artifacts.\n",
            "assets/model.ckpt": "fixture checkpoint bytes\n",
            "assets/model.pt": "fixture checkpoint bytes\n",
            "assets/model.safetensors": "fixture checkpoint bytes\n",
            "assets/analysis.ipynb": "{}\n",
            "vlmeval/private.key": "fixture key placeholder\n",
            "vlmeval/dataset/scores.xlsx": "fixture spreadsheet bytes\n",
            "vlmeval/vlm/bagel/train.py": "raise RuntimeError('training extra')\n",
            "vlmeval/vlm/bagel/data/private_registry.py": "PRIVATE_DATA = True\n",
            "scripts/cluster_launcher.sh": "exit 1\n",
        }
        for name, text in {**self.allowed, **self.excluded}.items():
            self.write(name, text)
        (self.repository / "outside_source.py").write_text("OUTSIDE = True\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Fixture source")

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.repository), *arguments], text=True).strip()

    def write(self, name, text):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_exports_only_tracked_allowed_files_and_excludes_nested_git(self):
        self.write("vlmeval/untracked_experiment.py", "UNTRACKED = True\n")
        nested = self.source / "docs/nested"
        nested.mkdir()
        subprocess.check_call(["git", "-C", str(nested), "init", "--quiet"])
        self.write("docs/nested/untracked_module.py", "NESTED = True\n")
        self.assertEqual(set(self.git("ls-files", "VLMEvalKit").splitlines()),
                         {"VLMEvalKit/" + name for name in self.allowed.keys() | self.excluded.keys()})
        EXPORT_SOURCE(self.source, self.destination)
        records, metadata = assert_export_integrity(self, self.destination)
        self.assertEqual({row["path"] for row in records}, set(self.allowed))
        self.assertEqual(metadata["revision"], self.git("rev-parse", "HEAD"))
        for row in records:
            self.assertEqual(row["source_sha256"], digest(self.source / row["path"]))

    def test_exports_uncommitted_tracked_content_instead_of_head(self):
        source = self.write("vlmeval/dataset/scorer.py", "def score(answer):\n    return answer == 'B'\n")
        head_content = self.git("show", "HEAD:VLMEvalKit/vlmeval/dataset/scorer.py")
        self.assertNotEqual(head_content, source.read_text().strip())
        EXPORT_SOURCE(self.source, self.destination)
        self.assertEqual((self.destination / "vlmeval/dataset/scorer.py").read_bytes(), source.read_bytes())
        records, _ = assert_export_integrity(self, self.destination)
        record = next(row for row in records if row["path"] == "vlmeval/dataset/scorer.py")
        self.assertEqual(record["source_sha256"], digest(source))
        self.assertEqual(record["source_sha256"], record["release_sha256"])

    def test_portability_patches_keep_unrelated_code_and_remain_valid_python(self):
        EXPORT_SOURCE(self.source, self.destination)
        config = (self.destination / "vlmeval/config.py").read_text()
        adapter = (self.destination / "vlmeval/vlm/bagel_vlm.py").read_text()
        runner = (self.destination / "run.py").read_text()
        self.assertIn("if RANK == 0 and args.reuse and len(prev_pred_roots):", runner)
        self.assertNotIn("private-experiment", config)
        self.assertIn('unrelated_registry = {"fixture": "preserve this value"}', config)
        self.assertIn('supported_VLM = {**unrelated_registry, **bagel_series}', config)
        self.assertNotIn("/private/base-model", adapter)
        self.assertIn("BAGEL_ORI_MODEL_PATH", adapter)
        self.assertIn("    def predict(self, item):\n        return item", adapter)
        self.assertIn("OMNITASKONOMY_EVAL_ID", runner)
        self.assertIn("eval_results = dataset.evaluate(result_file, **judge_kwargs)", runner)
        self.assertIn("'failed', error_message=str(e)", runner)
        for name in PATCHED_FILES:
            compile((self.destination / name).read_text(), name, "exec")

    def test_existing_destination_is_rejected_without_modification(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("Keep the user's existing directory.\n")
        original = marker.read_bytes()
        with self.assertRaisesRegex(ValueError, "Destination already exists"):
            EXPORT_SOURCE(self.source, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), original)

    def test_exported_package_metadata_does_not_need_a_vendored_readme(self):
        EXPORT_SOURCE(self.source, self.destination)
        metadata = subprocess.check_output(
            [sys.executable, "setup.py", "--name", "--version"], cwd=self.destination, text=True,
        )
        self.assertEqual(metadata.splitlines(), ["fixture-vlmevalkit", "0.1.0"])
        self.assertFalse(list(self.destination.rglob("*.md")))

    def test_patch_anchor_drift_fails_before_creating_destination(self):
        self.write("run.py", RUNNER.replace("from vlmeval.smp import *", "from vlmeval.smp import get_pred_file_format"))
        with self.assertRaisesRegex(ValueError, "expected one patch anchor"):
            EXPORT_SOURCE(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_tracked_symlink_cannot_copy_a_file_outside_source(self):
        outside = self.repository.parent / "outside.txt"
        outside.write_text("Outside the selected source tree.\n")
        link = self.source / "assets/linked.txt"
        link.symlink_to(outside)
        self.git("add", "VLMEvalKit/assets/linked.txt")
        with self.assertRaisesRegex(ValueError, "Expected a regular source file"):
            EXPORT_SOURCE(self.source, self.destination)
        self.assertFalse((self.destination / "assets/linked.txt").exists())


class VendoredReleaseTests(unittest.TestCase):
    def test_shipped_source_matches_its_complete_hash_manifest(self):
        destination = ROOT / "VLMEvalKit"
        release_patches = PATCHED_FILES | {
            "vlmeval/api/gpt.py", "vlmeval/dataset/image_mcq.py", "vlmeval/dataset/utils/multiple_choice.py",
        }
        records, metadata = assert_export_integrity(self, destination, release_patches)
        names = {row["path"] for row in records}
        self.assertEqual(metadata["source"], "gen4und/VLMEvalKit")
        self.assertTrue({"vlmeval/dataset/image_mcq.py", "vlmeval/dataset/utils/multiple_choice.py",
                         "vlmeval/utils/matching_util.py", "vlmeval/vlm/bagel_vlm.py"} <= names)


if __name__ == "__main__":
    unittest.main()
