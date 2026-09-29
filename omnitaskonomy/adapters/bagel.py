"""BAGEL losses and parameter roles for the UMM runners."""

import json
from pathlib import Path
import random
import re

import numpy as np

from omnitaskonomy.bagel import load_bagel, make_inferencer
from omnitaskonomy.umm import Loss, ParameterSpec


def classify_parameter(name):
    if "_moe_gen" in name:
        return None
    layer_match = re.search(r"layers\.(\d+)\.", name)
    layer = int(layer_match[1]) if layer_match else None
    module = None
    if name.startswith("language_model.") and name.endswith("weight"):
        for fragment, family in (("input_layernorm", "input_ln"), ("post_attention_layernorm", "post_attn_ln"),
                                  ("q_norm", "q_norm"), ("k_norm", "k_norm"), ("embed_tokens", "embed_tokens")):
            if fragment in name:
                module = f"llm.{family}.weight"
                break
        if name == "language_model.model.norm.weight":
            module = "llm.final_norm.weight"
        if module is None:
            match = re.search(r"\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$", name)
            if match:
                module = "llm." + match[1] + ".weight"
    elif name.startswith("vit_model.") and name.endswith("weight"):
        suffix = re.sub(r"^vit_model\.vision_model\.(?:encoder\.layers\.\d+\.|embeddings\.)?", "", name)
        module = "vit." + suffix
    elif name.startswith("connector."):
        module = name
    elif name == "vit_pos_embed.pos_embed":
        module = "connector.vit_pos_embed"
    if module is None:
        return None
    structural_zero = name == "language_model.model.norm.weight" or layer == 27 and module in {
        "llm.post_attn_ln.weight", "llm.q_norm.weight", "llm.self_attn.q_proj.weight",
        "llm.self_attn.o_proj.weight", "llm.mlp.down_proj.weight", "llm.mlp.gate_proj.weight", "llm.mlp.up_proj.weight"}
    return {"name": name, "module": module, "layer": layer, "structural_zero_i2i": structural_zero}


def _seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def objective_loss(runtime, packed_dataset, record, objective, noise_seed):
    import torch
    from data.dataset_base import collate_wrapper

    inner = packed_dataset.grouped_datasets[0]
    if objective == "i2t" and "gradient_mcq" in record:
        sample = inner._init_data()
        for value in record["images"]:
            inner._add_image(sample, inner._image(value), need_loss=False, need_vae=False,
                             need_vit=True, enable_cfg=False)
        inner._add_text(sample, record["gradient_mcq"]["prompt"], need_loss=False, enable_cfg=False)
        inner._add_text(sample, record["gradient_mcq"]["answer"], need_loss=True, enable_cfg=False)
    else:
        sample = inner.parse_row(record)
    token_count = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
    if token_count > packed_dataset.max_num_tokens_per_sample:
        raise ValueError(f"Gradient example exceeds token limit: {record['uid']}/{token_count}")
    packed = packed_dataset.pack_sequence(sample, packed_dataset.set_sequence_status())
    data = packed_dataset.to_tensor(packed)
    data["batch_data_indexes"] = [{"uid": record["uid"]}]
    batch = collate_wrapper()([data]).cuda(runtime.device)
    data = batch.to_dict()
    data.pop("batch_data_indexes")
    data.pop("ce_loss_weights", None)
    _seed(noise_seed)
    with torch.enable_grad(), torch.autocast(runtime.device.type, dtype=torch.bfloat16):
        if "padded_images" in data:
            with torch.no_grad():
                data["padded_latent"] = runtime.vae_model.encode(data.pop("padded_images").to(dtype=torch.bfloat16))
        result = runtime.model(**data)
        count = len(data["mse_loss_indexes" if objective == "i2i" else "ce_loss_indexes"])
        if count == 0:
            raise ValueError("Gradient objective has no supervised loss tokens")
        if "gradient_mcq" in record and count != 2:
            raise ValueError("Benchmark objective must supervise one option label plus EOS")
        loss = result["mse"].mean(dim=-1).sum() / count if objective == "i2i" else result["ce"].sum() / count
    if not bool(torch.isfinite(loss)):
        raise ValueError("Gradient objective produced a non-finite loss")
    return loss, count


class BagelAdapter:
    def __init__(self, model_path, checkpoint, device, options):
        unknown = options.keys() - {"dtype", "recompute_latent_positions"}
        if unknown:
            raise ValueError(f"Unknown BAGEL adapter options: {sorted(unknown)}")
        checkpoint = checkpoint or model_path / "ema.safetensors"
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        self.runtime = load_bagel(model_path, checkpoint, device, dtype=options.get("dtype", "bfloat16"))
        self.model = self.runtime.model
        self.checkpoint_files = (checkpoint, *(model_path / name for name in (
            "llm_config.json", "vit_config.json", "ae.safetensors", "tokenizer.json", "tokenizer_config.json")))
        if options.get("recompute_latent_positions", False):
            from modeling.bagel.modeling_utils import PositionEmbedding
            self.model.latent_pos_embed = PositionEmbedding(self.model.max_latent_size, self.model.hidden_size).to(
                self.runtime.device, next(self.model.parameters()).dtype)
            self.model.to(dtype=next(self.model.parameters()).dtype)
        self._datasets = {}
        self._inferencer = None

    def parameter_specs(self):
        from train.freeze_policy import parameter_role

        for name, parameter in self.model.named_parameters():
            metadata = classify_parameter(name)
            role = parameter_role(name) if parameter.requires_grad else "fixed"
            yield ParameterSpec(name, role, metadata["module"] if metadata else None,
                                metadata["layer"] if metadata else None,
                                trainable=parameter.requires_grad,
                                zero_objectives=("i2i",) if metadata and metadata["structural_zero_i2i"] else ())

    def loss(self, records, objective, context):
        from data.dataset_base import DataConfig, PackedDataset
        from data.transforms import ImageTransform

        # BAGEL dispatches packed loss computation through its training forward.
        self.model.train()
        key = (str(context.manifest), objective, context.condition_dropout, context.target_interpolation,
               json.dumps(context.vit_transform, sort_keys=True), context.max_tokens)
        if key not in self._datasets:
            spec = {"manifest": str(context.manifest), "kind": objective,
                    "transform": self.runtime.vae_transform,
                    "vit_transform": ImageTransform(**context.vit_transform) if context.vit_transform else self.runtime.vit_transform,
                    "target_interpolation": context.target_interpolation, "fixed_batch_size": 1, "weight": 1}
            config = DataConfig({"adapter": spec}, text_cond_dropout_prob=context.condition_dropout,
                                vit_cond_dropout_prob=context.condition_dropout,
                                vae_cond_dropout_prob=context.condition_dropout, max_latent_size=64)
            self._datasets[key] = PackedDataset(config, self.runtime.tokenizer, self.runtime.new_token_ids,
                                                0, 1, 1, expected_num_tokens=1,
                                                max_num_tokens_per_sample=context.max_tokens)
        total, count = 0, 0
        for index, record in enumerate(records):
            mean, tokens = objective_loss(self.runtime, self._datasets[key], record, objective, context.seed + index)
            total = total + mean * tokens
            count += tokens
        return Loss(total, count)

    def generate(self, messages, *, output="text", seed=42, dataset=None, **kwargs):
        import torch
        from PIL import Image

        if output not in {"text", "image"}:
            raise ValueError("BAGEL output must be text or image")
        if self._inferencer is None:
            self._inferencer = make_inferencer(self.runtime)
        inputs = []
        for message in messages:
            if message["type"] == "text":
                inputs.append(message["value"])
            elif message["type"] == "image":
                with Image.open(message["value"]) as image:
                    inputs.append(image.convert("RGB"))
            else:
                raise ValueError(f"Unsupported BAGEL message type: {message['type']}")
        _seed(seed)
        self.model.eval()
        with torch.inference_mode(), torch.autocast(self.runtime.device.type, dtype=torch.bfloat16):
            outputs = self._inferencer.interleave_inference(inputs, understanding_output=output == "text", **kwargs)
        expected = str if output == "text" else Image.Image
        return next(value for value in reversed(outputs) if isinstance(value, expected))

    def save_checkpoint(self, directory):
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items()},
                  str(directory / "model.safetensors"))


def create_adapter(*, model_path, checkpoint=None, device="cuda:0", options=None):
    return BagelAdapter(Path(model_path), Path(checkpoint) if checkpoint else None, device, options or {})
