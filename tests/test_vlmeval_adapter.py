"""Actual CPU checkpoint loading preserves BAGEL's nonpersistent RoPE buffers."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from accelerate import init_empty_weights
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLMEvalKit"))
sys.path.insert(0, str(ROOT / "VLMEvalKit/vlmeval/vlm/bagel"))
from vlmeval.vlm.bagel.modeling.qwen2.configuration_qwen2 import Qwen2Config
from vlmeval.vlm.bagel.modeling.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from omnitaskonomy.vlmeval_adapter import _load_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.rotary = Qwen2RotaryEmbedding(config=Qwen2Config(
            hidden_size=128, num_attention_heads=1, rope_theta=1000000.0,
        ))
        self.register_buffer("count", torch.tensor(3, dtype=torch.int64))


class AdapterCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "weights.safetensors"
        self.weight = torch.tensor([[1.003, 2.009], [3.017, 4.031]], dtype=torch.float32)

    def state(self):
        return {"projection.weight": self.weight, "count": torch.tensor(3, dtype=torch.int64)}

    def model(self):
        with init_empty_weights():
            return TinyModel()

    def test_preserves_native_rope_and_loads_bf16_checkpoint_parameters(self):
        state = {"_fsdp_wrapped_module.module." + key: value for key, value in self.state().items()}
        for prefix in ("time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"):
            state[prefix + ".unused"] = torch.ones(1)
        save_file(state, self.checkpoint)
        model = self.model()
        native_inv_freq = model.rotary.inv_freq.clone()
        self.assertTrue(model.projection.weight.is_meta)
        self.assertNotIn("rotary.inv_freq", model.state_dict())
        self.assertFalse(torch.equal(native_inv_freq, native_inv_freq.bfloat16().float()))

        model = _load_checkpoint(model, self.checkpoint, "cpu")
        self.assertFalse(model.training)
        self.assertEqual(model.projection.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(model.projection.weight, self.weight.bfloat16(), rtol=0, atol=0)
        self.assertEqual(model.count.dtype, torch.int64)
        self.assertEqual(model.count.item(), 3)
        self.assertEqual(model.rotary.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(model.rotary.inv_freq, native_inv_freq, rtol=0, atol=0)

        reference = TinyModel().rotary
        query = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
        positions = torch.tensor([[0, 32, 512, 4095]])
        for actual, expected in zip(model.rotary(query, positions), reference(query, positions)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_missing_unexpected_and_wrong_shape_weights(self):
        for problem in ("missing", "unexpected", "shape"):
            with self.subTest(problem=problem):
                state = self.state()
                if problem == "missing":
                    del state["projection.weight"]
                    error, message = ValueError, "missing=.*projection.weight"
                elif problem == "unexpected":
                    state["unknown.weight"] = torch.ones(1)
                    error, message = ValueError, "unexpected=.*unknown.weight"
                else:
                    state["projection.weight"] = torch.ones(3, 2)
                    error, message = RuntimeError, "size mismatch for projection.weight"
                save_file(state, self.checkpoint)
                with self.assertRaisesRegex(error, message):
                    _load_checkpoint(self.model(), self.checkpoint, "cpu")

    def test_float32_matches_standard_load_without_rounding_weights_or_rope(self):
        save_file(self.state(), self.checkpoint)
        reference = TinyModel()
        reference.load_state_dict(self.state())
        model = _load_checkpoint(self.model(), self.checkpoint, "cpu", "float32")
        self.assertEqual(model.projection.weight.dtype, torch.float32)
        self.assertFalse(torch.equal(self.weight, self.weight.bfloat16().float()))
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.rotary.inv_freq, reference.rotary.inv_freq, rtol=0, atol=0)
        sample = torch.tensor([[0.7, -0.1]])
        torch.testing.assert_close(model.projection(sample), reference.projection(sample), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
