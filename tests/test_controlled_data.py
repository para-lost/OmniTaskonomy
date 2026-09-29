import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Bagel"))
from data.transforms import ImageTransform
from omnitaskonomy.datasets import ManifestDataset


class Tokenizer:
    def encode(self, text):
        return list(text.encode())


class MultiImageManifestTests(unittest.TestCase):
    def test_multiimage_order_and_supervised_text_match_conversation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, color in (("dark", 10), ("light", 240)):
                Image.new("RGB", (28, 28), (color, color, color)).save(root / f"{name}.png")
            row = dict(uid="two-views", images=["dark.png", "light.png"], conversations=[
                {"from": "human", "value": "First <image> second <image> Compare."},
                {"from": "gpt", "value": "The second image is brighter."}])
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            transform = ImageTransform(28, 28, 14)
            dataset = ManifestDataset("multi", manifest, "i2t", transform, Tokenizer(), transform)
            parsed = dataset.parse_row(row)
            self.assertEqual([step["type"] for step in parsed["sequence_plan"]],
                             ["text", "vit_image", "text", "vit_image", "text", "text"])
            self.assertEqual([step["loss"] for step in parsed["sequence_plan"]], [0, 0, 0, 0, 0, 1])
            self.assertLess(parsed["image_tensor_list"][0].mean(), parsed["image_tensor_list"][1].mean())
            row["conversations"][0]["value"] = "Only one image: <image>"
            with self.assertRaisesRegex(ValueError, "placeholders"):
                dataset.parse_row(row)


if __name__ == "__main__":
    unittest.main()
