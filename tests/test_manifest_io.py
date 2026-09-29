import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from omnitaskonomy.data.common import resolve_image, sha256, write_manifest


class ManifestIOTests(unittest.TestCase):
    def test_manifest_order_duplicate_rejection_and_moving_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle/train.jsonl"
            image = output.parent / "images/a.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (40, 30), (80, 120, 160)).save(image)
            metadata = write_manifest([{"uid": "a", "image": "images/a.png"}], output, {"seed": 0})
            self.assertEqual(metadata["manifest_sha256"], sha256(output))
            moved = root / "moved"
            shutil.move(output.parent, moved)
            self.assertTrue(resolve_image(moved / "train.jsonl", "images/a.png").is_file())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                write_manifest([{"uid": "a"}, {"uid": "a"}], root / "duplicate.jsonl", {})
