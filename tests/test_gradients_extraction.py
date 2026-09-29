import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Bagel"))
from data.dataset_base import DataConfig, PackedDataset
from omnitaskonomy.gradients.extract import countsketch, objective_loss


class CharacterTokenizer:
    def encode(self, text):
        return list(text.encode())


class SmallVAE(nn.Module):
    def encode(self, images):
        return F.avg_pool2d(images.float(), 8)


class LossModel(nn.Module):
    """Analytic differentiable losses isolate the real packer's loss reduction."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.last_inputs = None

    def forward(self, **inputs):
        self.last_inputs = inputs
        if "mse_loss_indexes" in inputs:
            count = len(inputs["mse_loss_indexes"])
            offsets = torch.arange(count * 3).reshape(count, 3).float()
            return {"mse": (self.weight - offsets).square(), "ce": None}
        labels = inputs["packed_label_ids"] % 2
        logits = torch.stack((self.weight.expand(len(labels)), -self.weight.expand(len(labels))), dim=-1)
        return {"mse": None, "ce": F.cross_entropy(logits, labels, reduction="none")}


class GradientPackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, color in (("dark", 10), ("light", 240)):
            Image.new("RGB", (32, 32), (color, color, color)).save(self.root / f"{name}.png")
        self.runtime = SimpleNamespace(device=torch.device("cpu"), model=LossModel(), vae_model=SmallVAE())

    def packed(self, row, kind):
        manifest = self.root / f"{kind}.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        groups = {"gradient": {"manifest": str(manifest), "kind": kind,
            "image_transform_args": {"max_image_size": 32, "min_image_size": 32, "image_stride": 16},
            "vit_image_transform_args": {"max_image_size": 28, "min_image_size": 28, "image_stride": 14},
            "fixed_batch_size": 1, "weight": 1}}
        config = DataConfig(groups, text_cond_dropout_prob=0, vit_cond_dropout_prob=0,
                            vae_cond_dropout_prob=0, max_latent_size=64)
        return PackedDataset(config, CharacterTokenizer(), {"bos_token_id": 1, "eos_token_id": 2,
            "start_of_image": 3, "end_of_image": 4}, 0, 1, 1)

    def test_countsketch_is_fixed_signed_linear_map(self):
        gradient = torch.arange(1, 8).float()
        actual, fingerprint = countsketch(gradient, 3, 17)
        generator = torch.Generator().manual_seed(17)
        buckets = torch.randint(3, (7,), generator=generator, dtype=torch.int32)
        signs = torch.randint(2, (7,), generator=generator, dtype=torch.int8) * 2 - 1
        expected = torch.tensor([sum(float(gradient[i] * signs[i]) for i in range(7) if buckets[i] == bucket)
                                 for bucket in range(3)])
        torch.testing.assert_close(actual, expected)
        doubled, second_fingerprint = countsketch(gradient * 2, 3, 17)
        torch.testing.assert_close(doubled, actual * 2)
        self.assertEqual(fingerprint, second_fingerprint)

    def test_i2i_averages_latent_coordinates_and_supervised_tokens(self):
        row = dict(uid="pair", source_image="dark.png", target_image="light.png", prompt="Reorder.")
        loss, tokens = objective_loss(self.runtime, self.packed(row, "i2i"), row, "i2i", 42)
        self.assertEqual(tokens, 4)
        offsets = torch.arange(12).float()
        expected = (2 - offsets).square().mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        loss.backward()
        self.assertAlmostEqual(self.runtime.model.weight.grad.item(), (2 * (2 - offsets)).mean().item())
        self.assertEqual(tuple(self.runtime.model.last_inputs["padded_latent"].shape), (2, 3, 4, 4))
        self.assertNotIn("padded_images", self.runtime.model.last_inputs)

    def test_multiimage_i2t_uses_all_ordered_views_and_answer_tokens(self):
        row = dict(uid="two", images=["dark.png", "light.png"], conversations=[
            {"from": "human", "value": "<image> then <image> Which is brighter?"},
            {"from": "gpt", "value": "B"}])
        loss, tokens = objective_loss(self.runtime, self.packed(row, "i2t"), row, "i2t", 42)
        inputs = self.runtime.model.last_inputs
        self.assertEqual(tokens, 2)
        self.assertEqual(inputs["packed_label_ids"].tolist(), [ord("B"), 2])
        self.assertEqual(inputs["vit_token_seqlens"].tolist(), [4, 4])
        self.assertLess(inputs["packed_vit_tokens"][:4].mean(), inputs["packed_vit_tokens"][4:].mean())
        expected = F.cross_entropy(torch.tensor([[2., -2.], [2., -2.]]), torch.tensor([0, 0]))
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        loss.backward()
        self.assertGreater(abs(self.runtime.model.weight.grad.item()), 0)

    def test_frozen_mcq_preserves_prompt_whitespace_and_only_answer_plus_eos(self):
        prompt = "  Compare both images.\nAnswer with one letter.\n\n "
        row = dict(uid="BLINK:0", images=["dark.png", "light.png"],
                   gradient_mcq={"prompt": prompt, "answer": "A"})
        with torch.no_grad():
            loss, tokens = objective_loss(self.runtime, self.packed(row, "i2t"), row, "i2t", 42)
        inputs = self.runtime.model.last_inputs
        self.assertEqual(tokens, 2)
        self.assertEqual(inputs["packed_label_ids"].tolist(), [ord("A"), 2])
        ids = inputs["packed_text_ids"].tolist()
        prompt_ids = list(prompt.encode())
        self.assertTrue(any(ids[start:start + len(prompt_ids)] == prompt_ids for start in range(len(ids))))
        self.assertEqual(inputs["vit_token_seqlens"].tolist(), [4, 4])
        loss.backward()
        self.assertTrue(torch.isfinite(self.runtime.model.weight.grad))


if __name__ == "__main__":
    unittest.main()
