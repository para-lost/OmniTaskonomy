from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

from omnitaskonomy.examples.tiny_umm import TinyUMM
from omnitaskonomy.umm import Loss, LossContext, ParameterInventory, ParameterSpec, adapter_provenance, load_adapter


FACTORY = "omnitaskonomy.examples.tiny_umm:create_adapter"


class ParameterRoleTests(unittest.TestCase):
    @torch.enable_grad()
    def test_r4_updates_generation_then_restores_initial_scope(self):
        model = TinyUMM()
        specs = [ParameterSpec(name, name.split(".")[0] if name != "scale" else "fixed", None)
                 for name, _ in model.named_parameters()]
        inventory = ParameterInventory(model, specs)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        inventory.apply("generation")
        optimizer = torch.optim.AdamW(inventory.parameters(), lr=.1)
        model(torch.ones(2, 4), "i2i").square().sum().backward()
        optimizer.step()
        for name, value in model.named_parameters():
            self.assertEqual(not torch.equal(value, before[name]), name.startswith("generation."))
        inventory.apply("all")
        self.assertTrue(all(value.requires_grad for name, value in model.named_parameters() if name != "scale"))
        self.assertFalse(model.scale.requires_grad)
        self.assertTrue(all(value.grad is None for value in model.parameters()))

    def test_tied_parameters_are_counted_once_and_conflicting_roles_fail(self):
        model = nn.Module()
        model.left = nn.Linear(2, 2, bias=False)
        model.right = nn.Linear(2, 2, bias=False)
        model.right.weight = model.left.weight
        specs = [ParameterSpec(name, "shared", "projection") for name, _ in model.named_parameters(remove_duplicate=False)]
        inventory = ParameterInventory(model, specs)
        self.assertEqual(len(inventory.parameters()), 1)
        self.assertEqual(len(inventory.apply("all")["trainable"]), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ParameterInventory(model, [specs[0], ParameterSpec("right.weight", "generation", "projection")])
        with self.assertRaisesRegex(ValueError, "omitted"):
            ParameterInventory(model, [])
        with self.assertRaisesRegex(ValueError, "no eligible"):
            inventory.apply("generation")
        self.assertTrue(model.left.weight.requires_grad)

    @torch.enable_grad()
    def test_fixed_operations_preserve_backward_to_generation(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
        inventory = ParameterInventory(model, [ParameterSpec("0.weight", "generation", None),
                                               ParameterSpec("1.weight", "understanding", None)])
        inventory.apply("generation")
        model(torch.ones(1, 2)).sum().backward()
        self.assertGreater(model[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model[1].weight.grad)


class AdapterTests(unittest.TestCase):
    @torch.enable_grad()
    def test_losses_checkpoint_roundtrip_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(17)
            torch.save(TinyUMM().state_dict(), root / "model.pt")
            Image.new("RGB", (4, 4), (70, 120, 210)).save(root / "image.png")
            adapter = load_adapter(FACTORY, root, device="cpu")
            context = LossContext(root / "data.jsonl", 42)
            row = {"image": "image.png", "conversations": [
                {"from": "human", "value": "Which?"}, {"from": "gpt", "value": "B"}]}
            one = adapter.loss([row], "i2t", context)
            two = adapter.loss([row, row], "i2t", context)
            torch.testing.assert_close(one.mean, two.mean)
            self.assertEqual(two.count, 2)
            two.mean.backward()
            self.assertGreater(adapter.model.shared.weight.grad.abs().sum().item(), 0)
            messages = [{"type": "image", "value": str(root / "image.png")}, {"type": "text", "value": "Which?"}]
            expected = adapter.generate(messages)
            saved = root / "saved"
            adapter.save_checkpoint(saved)
            restored = load_adapter(FACTORY, root, checkpoint=saved, device="cpu")
            self.assertEqual(restored.generate(messages), expected)
            self.assertEqual(restored.generate(messages, output="image").size, (8, 8))
            before = adapter_provenance(restored, FACTORY)
            with torch.no_grad():
                restored.model.shared.weight.add_(1)
            restored.save_checkpoint(saved)
            self.assertNotEqual(adapter_provenance(restored, FACTORY)["checkpoint_sha256"], before["checkpoint_sha256"])

    def test_invalid_loss_is_rejected_before_backward(self):
        for loss in (Loss(torch.tensor(1.), 0), Loss(torch.tensor(float("nan")), 1), Loss(torch.ones(2), 1)):
            with self.assertRaises(ValueError):
                _ = loss.mean


if __name__ == "__main__":
    unittest.main()
