import ast
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest

import torch
from transformers import HfArgumentParser

from Bagel.train.freeze_policy import freeze_shared_for_i2i, parameter_receipt


TRAINER = Path(__file__).resolve().parents[1] / "Bagel/train/pretrain_unified_navit.py"


class FrozenParameterTests(unittest.TestCase):
    def test_only_generation_parameters_change_after_optimizer_step(self):
        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.model = torch.nn.Module()
        model.language_model.model.embed_tokens = torch.nn.Embedding(4, 2)
        model.language_model.model.norm = torch.nn.Linear(2, 2)
        model.language_model.model.norm_moe_gen = torch.nn.Linear(2, 2)
        layer = torch.nn.Module()
        layer.q_proj = torch.nn.Linear(2, 2)
        layer.q_proj_moe_gen = torch.nn.Linear(2, 2)
        layer.mlp_moe_gen = torch.nn.Linear(2, 2)
        layer.input_layernorm_moe_gen = torch.nn.Linear(2, 2)
        model.language_model.model.layers = torch.nn.ModuleList([layer])
        model.language_model.lm_head = torch.nn.Linear(2, 4)
        model.vit_model = torch.nn.Linear(2, 2)
        model.connector = torch.nn.Linear(2, 2)
        model.vae2llm = torch.nn.Linear(2, 2)
        model.llm2vae = torch.nn.Linear(2, 2)
        model.time_embedder = torch.nn.Linear(2, 2)
        model.latent_pos_embed = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        model.time_embedder.bias.requires_grad_(False)

        expected = {
            f"{prefix}.{suffix}"
            for prefix in (
                "language_model.model.norm_moe_gen",
                "language_model.model.layers.0.q_proj_moe_gen",
                "language_model.model.layers.0.mlp_moe_gen",
                "language_model.model.layers.0.input_layernorm_moe_gen",
                "vae2llm", "llm2vae", "time_embedder",
            )
            for suffix in ("weight", "bias")
        } - {"time_embedder.bias"}
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        freeze_shared_for_i2i(model)
        receipt = parameter_receipt(model)
        self.assertEqual({row["name"] for row in receipt["trainable"]}, expected)
        self.assertEqual(receipt["frozen_numel"], sum(
            parameter.numel() for name, parameter in model.named_parameters() if name not in expected
        ))

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        inputs = torch.ones(1, 2)
        with torch.enable_grad():
            loss = sum(module(inputs).sum() for module in model.modules()
                       if isinstance(module, torch.nn.Linear))
            loss = loss + model.language_model.model.embed_tokens(torch.tensor([0])).sum()
            loss.backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertEqual(parameter.grad is not None, name in expected)
                self.assertEqual(not torch.equal(parameter, before[name]), name in expected)

    def test_generation_gradients_flow_through_frozen_understanding_operations(self):
        tree = ast.parse(TRAINER.read_text())
        assignment = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.Assign)
                          and ast.unparse(node.targets[0]) == "llm_config.freeze_und")
        for frozen_i2i, detach_understanding in ((False, True), (True, False)):
            llm_config = SimpleNamespace()
            namespace = {"llm_config": llm_config,
                         "training_args": SimpleNamespace(freeze_und=True,
                                                          freeze_shared_for_i2i=frozen_i2i)}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(TRAINER), "exec"), namespace)
            self.assertEqual(llm_config.freeze_und, detach_understanding)

        model = torch.nn.Module()
        model.vae2llm = torch.nn.Linear(2, 2, bias=False)
        model.connector = torch.nn.Linear(2, 2, bias=False)
        model.llm2vae = torch.nn.Linear(2, 2, bias=False)
        for parameter in model.parameters():
            torch.nn.init.ones_(parameter)
        freeze_shared_for_i2i(model)
        with torch.enable_grad():
            features = model.connector(model.vae2llm(torch.ones(1, 2)))
            if llm_config.freeze_und:
                features = features.detach()
            model.llm2vae(features).sum().backward()
        self.assertIsNone(model.connector.weight.grad)
        torch.testing.assert_close(model.vae2llm.weight.grad, torch.full((2, 2), 4.0))

    def test_trainer_rejects_full_llm_freeze_argument(self):
        tree = ast.parse(TRAINER.read_text())
        arguments = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "TrainingArguments")
        namespace = {"dataclass": dataclass, "field": field, "Optional": Optional,
                     "__name__": __name__}
        exec(compile(ast.Module(body=[arguments], type_ignores=[]), str(TRAINER), "exec"), namespace)
        parser = HfArgumentParser(namespace["TrainingArguments"])
        error = StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as failure:
            parser.parse_args_into_dataclasses(["--freeze_llm", "True"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("ambiguous option: --freeze_llm", error.getvalue())
        args, = parser.parse_args_into_dataclasses(["--freeze_shared_for_i2i", "True"])
        self.assertTrue(args.freeze_shared_for_i2i)


if __name__ == "__main__":
    unittest.main()
