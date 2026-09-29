"""Load BAGEL weights, tokenizer, image transforms and inference caches."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


@dataclass
class BagelRuntime:
    model: Any
    tokenizer: Any
    vae_model: Any
    vae_transform: Any
    vit_transform: Any
    new_token_ids: dict
    device: Any


def load_bagel(model_path, checkpoint=None, device="cuda:0", *, dtype="bfloat16"):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Bagel"))
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.autoencoder import load_ae
    from modeling.bagel import (Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
                                SiglipVisionConfig, SiglipVisionModel)
    from modeling.qwen2 import Qwen2Tokenizer

    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    parameter_dtype = dtypes[dtype]
    model_path = Path(model_path)
    checkpoint = Path(checkpoint) if checkpoint is not None else model_path / "ema.safetensors"
    llm = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm.qk_norm = True
    llm.tie_word_embeddings = False
    llm.layer_module = "Qwen2MoTDecoderLayer"
    llm.freeze_und = False
    vit = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit.rope = False
    vit.num_hidden_layers -= 1
    vae, vae_config = load_ae(str(model_path / "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm,
        vit_config=vit, vae_config=vae_config, vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh", latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
    weights = {key.removeprefix("_fsdp_wrapped_module.").removeprefix("module."): value
               for key, value in load_file(str(checkpoint), device="cpu").items()}
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    # RoPE frequencies created by the constructor must remain FP32.
    for name, parameter in list(model.named_parameters()):
        set_module_tensor_to_device(model, name, device, value=parameter,
                                    dtype=parameter_dtype, clear_cache=False)
    model.to(device=device).eval()
    vae.to(device=device, dtype=parameter_dtype).eval().requires_grad_(False)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, token_ids, _ = add_special_tokens(tokenizer)
    return BagelRuntime(model, tokenizer, vae, ImageTransform(512, 256, 16),
                        ImageTransform(518, 224, 14), token_ids, torch.device(device))


def make_inferencer(runtime):
    """Bridge BAGEL's custom cache methods, which bypass forward device hooks."""
    import torch
    from inferencer import InterleaveInferencer

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(runtime.device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    for name in ("prepare_prompts", "prepare_vae_images", "prepare_vit_images",
                 "prepare_vae_latent", "prepare_vae_latent_cfg", "prepare_start_tokens"):
        original = getattr(runtime.model, name)

        def prepare(*args, _original=original, **kwargs):
            return move(_original(*args, **kwargs))

        setattr(runtime.model, name, prepare)
    dtype = runtime.model.language_model.model.embed_tokens.weight.dtype
    for name in ("encode", "decode"):
        original = getattr(runtime.vae_model, name)

        def vae_call(tensor, *args, _original=original, **kwargs):
            return _original(tensor.to(runtime.device, dtype=dtype), *args, **kwargs)

        setattr(runtime.vae_model, name, vae_call)
    return InterleaveInferencer(runtime.model, runtime.vae_model, runtime.tokenizer,
        runtime.vae_transform, runtime.vit_transform, runtime.new_token_ids)
