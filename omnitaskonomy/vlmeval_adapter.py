"""BAGEL adapter for the vendored VLMEvalKit runner."""

from pathlib import Path
import sys

from vlmeval.dataset import DATASET_TYPE
from vlmeval.vlm.base import BaseModel


def _load_checkpoint(model, checkpoint, device, parameter_dtype="bfloat16"):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[parameter_dtype]
    weights = load_file(str(checkpoint), device="cpu")
    weights = {key.removeprefix("_fsdp_wrapped_module.").removeprefix("module."): value
               for key, value in weights.items()}
    result = model.load_state_dict(weights, strict=False, assign=True)
    # Understanding inference has no latent-generation input/output layers.
    generation_prefixes = ("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")
    unexpected = [key for key in result.unexpected_keys if not key.startswith(generation_prefixes)]
    if result.missing_keys or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={result.missing_keys}, unexpected={unexpected}")
    del weights

    # Preserve constructor-created FP32 buffers, including nonpersistent RoPE frequencies.
    parameter_names = [name for name, _ in model.named_parameters()]
    for name in parameter_names:
        set_module_tensor_to_device(
            model, name, device, value=model.get_parameter(name),
            dtype=dtype, clear_cache=False,
        )
    return model.to(device=device).eval()


class OmniTaskonomyBAGEL(BaseModel):
    INTERLEAVE = False
    INSTALL_REQ = False

    def __init__(self, model_path, checkpoint, seed=42, parameter_dtype="bfloat16"):
        super().__init__()
        import torch
        from accelerate import init_empty_weights
        from transformers import set_seed

        bagel_root = Path(__file__).resolve().parents[1] / "VLMEvalKit/vlmeval/vlm/bagel"
        sys.path.insert(0, str(bagel_root))
        from vlmeval.vlm.bagel.modeling.bagel import (
            Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from vlmeval.vlm.bagel.modeling.qwen2 import Qwen2Tokenizer
        from vlmeval.vlm.bagel.data.data_utils import add_special_tokens
        from vlmeval.vlm.bagel.data.transforms import ImageTransform

        # run.py limits CUDA_VISIBLE_DEVICES before importing model code on each rank.
        torch.cuda.set_device(0)
        set_seed(seed)
        model_path = Path(model_path)
        llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        config = BagelConfig(
            visual_gen=False, visual_und=True, llm_config=llm_config, vit_config=vit_config,
            vit_max_num_patch_per_side=70, latent_patch_size=2, max_latent_size=64,
            connector_act="gelu_pytorch_tanh",
        )
        with init_empty_weights():
            self.model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        self.model = _load_checkpoint(self.model, checkpoint, "cuda:0", parameter_dtype)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        self.image_transform = ImageTransform(
            max_image_size=980, min_image_size=378, image_stride=14, max_pixels=2007040,
        )

    def generate_inner(self, message, dataset=None):
        import torch
        from PIL import Image

        images = []
        texts = []
        for item in message:
            if item["type"] == "image":
                with Image.open(item["value"]) as image:
                    images.append(image.convert("RGB"))
            elif item["type"] == "text":
                texts.append(item["value"])
            else:
                raise ValueError(f"BAGEL evaluation only supports image/text messages: {item['type']}")
        prompt = "\n".join(texts)
        if dataset and DATASET_TYPE(dataset) == "VQA":
            prompt += " Answer:"
        max_length = 1000 if dataset == "CV-Bench-3D" else 32
        with torch.inference_mode():
            return self.model.chat(
                tokenizer=self.tokenizer, new_token_ids=self.new_token_ids,
                image_transform=self.image_transform, images=images, prompt=prompt,
                max_length=max_length, do_sample=False, temperature=1.0,
            ).strip()
