# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import json
import os
import sys
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, CurriculumPackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.freeze_policy import freeze_shared_for_i2i, parameter_receipt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


class Tee:
    """A class that writes to both a file and stdout/stderr."""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stream = stream
    
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
    
    def flush(self):
        self.file.flush()
        self.stream.flush()
    
    def isatty(self):
        return self.stream.isatty()
    
    def close(self):
        self.file.close()


def _unpatchify_vit(patches, pos_ids, patch_size=14, max_patches_per_side=70):
    """Reconstruct a VIT image from patchified tokens + position IDs.

    Args:
        patches:  [N, patch_size^2 * C] float tensor (one image)
        pos_ids:  [N] int tensor, encoded as h_idx * max_patches_per_side + w_idx
        patch_size: VIT patch size (default 14)
        max_patches_per_side: position embedding width (default 70)
    Returns:
        [C, H, W] float tensor in original normalized range
    """
    h_idxs = pos_ids // max_patches_per_side
    w_idxs = pos_ids % max_patches_per_side
    n_h = int(h_idxs.max().item()) + 1
    n_w = int(w_idxs.max().item()) + 1
    p = patch_size
    c = patches.shape[1] // (p * p)

    canvas = torch.zeros(c, n_h * p, n_w * p, dtype=patches.dtype)
    for patch, hi, wi in zip(patches, h_idxs.tolist(), w_idxs.tolist()):
        # patch: [p*p*c] -> [c, p, p]
        canvas[:, hi * p:(hi + 1) * p, wi * p:(wi + 1) * p] = (
            patch.reshape(p, p, c).permute(2, 0, 1)
        )
    return canvas


def save_sanity_check(data, data_indexes, tokenizer, save_dir, logger):
    """Save the first training batch to disk for sanity checking (rank-0 only)."""
    import json
    from torchvision.utils import save_image

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"[sanity check] saving first-batch data to {save_dir}")

    # 1. data_indexes metadata
    with open(os.path.join(save_dir, "data_indexes.json"), "w") as f:
        json.dump(data_indexes, f, indent=2, default=str)

    # 2. Decoded text + sample boundaries
    if "packed_text_ids" in data:
        ids = data["packed_text_ids"].cpu().tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        with open(os.path.join(save_dir, "packed_text.txt"), "w", encoding="utf-8") as f:
            f.write(f"sequence_length: {data['sequence_length']}\n")
            f.write(f"sample_lens: {data['sample_lens']}\n\n")
            f.write(decoded)

    # 3. VAE images — output images for generation tasks (mean/std=0.5)
    if "padded_images" in data:
        vae_dir = os.path.join(save_dir, "vae_images")
        os.makedirs(vae_dir, exist_ok=True)
        imgs = data["padded_images"].cpu().float()  # [N, C, H, W]
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(vae_dir, f"vae_{i:03d}.png"))
        logger.info(f"[sanity check] saved {len(imgs)} VAE (generation target) images → vae_images/")

    # 4. VIT images — input images for understanding tasks (mean/std=0.5)
    if "packed_vit_tokens" in data and "vit_token_seqlens" in data:
        vit_dir = os.path.join(save_dir, "vit_images")
        os.makedirs(vit_dir, exist_ok=True)
        tokens = data["packed_vit_tokens"].cpu().float()   # [total_patches, p^2*C]
        seqlens = data["vit_token_seqlens"].cpu().tolist() # patches per image
        pos_ids = data["packed_vit_position_ids"].cpu()    # [total_patches]

        offset = 0
        for i, n_patches in enumerate(seqlens):
            n_patches = int(n_patches)
            img_patches = tokens[offset: offset + n_patches]
            img_pos_ids = pos_ids[offset: offset + n_patches]
            offset += n_patches

            img = _unpatchify_vit(img_patches, img_pos_ids)   # [C, H, W] in [-1, 1]
            img = (img * 0.5 + 0.5).clamp(0, 1)
            save_image(img, os.path.join(vit_dir, f"vit_{i:03d}.png"))

        logger.info(f"[sanity check] saved {len(seqlens)} VIT (understanding input) images → vit_images/")

    logger.info(f"[sanity check] done — {save_dir}")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def freeze_layers(module: torch.nn.Module, num_layers_to_freeze: int, layer_attr_name: str = "layers", freeze_from_start: bool = True) -> None:
    """
    Freeze N layers of a module that has a ModuleList of layers.
    
    Args:
        module: The module containing layers (e.g., language_model or vit_model)
        num_layers_to_freeze: Number of layers to freeze
        layer_attr_name: Name of the attribute containing the ModuleList of layers (default: "layers")
        freeze_from_start: If True, freeze the first N layers; if False, freeze the last N layers (default: True)
    """
    if not hasattr(module, layer_attr_name):
        raise ValueError(f"Module does not have attribute '{layer_attr_name}'")
    
    layers = getattr(module, layer_attr_name)
    if not isinstance(layers, torch.nn.ModuleList):
        raise ValueError(f"Attribute '{layer_attr_name}' is not a ModuleList")
    
    total_layers = len(layers)
    if num_layers_to_freeze > total_layers:
        raise ValueError(f"Requested to freeze {num_layers_to_freeze} layers, but model only has {total_layers} layers")
    
    if freeze_from_start:
        # Freeze first N layers: [0, num_layers_to_freeze)
        layer_indices = range(num_layers_to_freeze)
    else:
        # Freeze last N layers: [total_layers - num_layers_to_freeze, total_layers)
        start_idx = total_layers - num_layers_to_freeze
        layer_indices = range(start_idx, total_layers)
    
    for i in layer_indices:
        layers[i].eval()
        for param in layers[i].parameters():
            param.requires_grad = False


def freeze_llm_input_layernorm(model: torch.nn.Module) -> tuple[list[str], int]:
    """Freeze only the understanding-side input RMSNorm in every LLM layer.

    BAGEL's MoT decoder has a separate ``input_layernorm_moe_gen`` for the
    image-generation branch.  This helper deliberately leaves that module,
    post-attention norms, and the final model norm untouched.
    """
    try:
        layers = model.language_model.model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot locate model.language_model.model.layers for input_ln freeze"
        ) from exc

    frozen_names: list[str] = []
    frozen_params = 0
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "input_layernorm"):
            raise RuntimeError(f"LLM layer {layer_idx} has no input_layernorm")
        layer_params = list(layer.input_layernorm.named_parameters())
        if [name for name, _ in layer_params] != ["weight"]:
            raise RuntimeError(
                f"Unexpected input_layernorm parameters in LLM layer {layer_idx}: "
                f"{[name for name, _ in layer_params]} (expected ['weight'])"
            )
        for leaf_name, param in layer_params:
            param.requires_grad = False
            frozen_names.append(
                f"language_model.model.layers.{layer_idx}.input_layernorm.{leaf_name}"
            )
            frozen_params += param.numel()

    if len(frozen_names) != len(layers):
        raise RuntimeError(
            f"Frozen input_ln tensor count mismatch: {len(frozen_names)} vs {len(layers)} layers"
        )
    return frozen_names, frozen_params


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )
    total_data_num: int = field(
        default=0,
        metadata={"help": "Stop training after this many total samples (summed across all GPUs). 0 = disabled (use total_steps)."}
    )
    save_at_data_nums: str = field(
        default="",
        metadata={"help": "Comma-separated cumulative sample counts at which to save extra checkpoints, e.g. '15000' for an early 15-epoch checkpoint on a 1k set."}
    )
    target_dataset_name: str = field(
        default="",
        metadata={"help": "If set, total_data_num counts only samples from this dataset (by group name in the YAML config). If empty, counts all samples."}
    )
    sanity_dump_every_samples: int = field(
        default=0,
        metadata={"help": "If >0, dump a sanity_check/epoch_NNN/ snapshot of the first batch each time cumulative_data_num crosses a multiple of this value (e.g. 1000 for one dump per 1k-sample epoch). 0 = only dump the very first batch."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.993,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    train_with_no_ema: bool = field(
        default=False,
        metadata={"help": "Disable EMA entirely during training without changing other parameters."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    freeze_shared_for_i2i: bool = field(
        default=False,
        metadata={"help": "Freeze understanding and shared parameters; retain generation experts and latent/time projections."}
    )
    freeze_llm_first_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Freeze the first N layers of the language model."}
    )
    freeze_llm_last_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Freeze the last N layers of the language model."}
    )
    freeze_llm_layers_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Freeze the first X fraction of language model layers (e.g., 0.33 for first 1/3)."}
    )
    freeze_llm_layers_ratio_from_end: Optional[float] = field(
        default=None,
        metadata={"help": "Freeze the last X fraction of language model layers (e.g., 0.33 for last 1/3)."}
    )
    freeze_vit_first_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Freeze the first N layers of the ViT. Overrides freeze_vit if set."}
    )
    freeze_vit_last_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Freeze the last N layers of the ViT. Overrides freeze_vit if set."}
    )
    freeze_vit_layers_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Freeze the first X fraction of ViT layers (e.g., 0.33 for first 1/3). Overrides freeze_vit if set."}
    )
    freeze_vit_layers_ratio_from_end: Optional[float] = field(
        default=None,
        metadata={"help": "Freeze the last X fraction of ViT layers (e.g., 0.33 for last 1/3). Overrides freeze_vit if set."}
    )
    train_only_input_ln: bool = field(
        default=False,
        metadata={"help": "Probe setting: freeze EVERYTHING, then unfreeze only the LLM "
                          "input_layernorm weights (train just input_ln). Applied last, "
                          "overriding all other freeze flags. Intended for the i2t stage."}
    )
    freeze_llm_input_ln: bool = field(
        default=False,
        metadata={"help": "Ablation setting: freeze only the understanding-side LLM "
                          "input_layernorm weight in every decoder layer; all other "
                          "parameters follow the ordinary freeze flags."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
    # save ema only
    save_ema_only: bool = field(
        default=True,
        metadata={"help": "Save only the EMA model weights."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.freeze_llm_input_ln and training_args.train_only_input_ln:
        raise ValueError(
            "--freeze_llm_input_ln and --train_only_input_ln have opposite semantics "
            "and cannot be enabled together"
        )
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        
        # Redirect stdout and stderr to log file (captures all print statements)
        stdout_log_path = os.path.join(training_args.results_dir, "stdout_stderr.log")
        tee_stdout = Tee(stdout_log_path, sys.stdout)
        tee_stderr = Tee(stdout_log_path, sys.stderr)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode=("disabled" if os.environ.get("WANDB_MODE") == "disabled" else ("offline" if training_args.wandb_offline else "online")),
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
        logger.info(f"All print outputs will be logged to: {stdout_log_path}")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)
    # Opt-in maximal determinism (DETERMINISTIC=1). Pins cuDNN + uses deterministic algos where
    # available. warn_only=True so ops without a deterministic kernel (flash-attn backward, some
    # FSDP/scatter reductions) WARN instead of erroring. Requires CUBLAS_WORKSPACE_CONFIG=:4096:8
    # in the env. NOTE: flash-attn + different GPU arch are still not bitwise-reproducible.
    if os.environ.get("DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        if dist.get_rank() == 0:
            logger.info("[DETERMINISTIC=1] cudnn.deterministic=True, benchmark=False, use_deterministic_algorithms(warn_only=True)")

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    # Frozen I2I must retain gradients through fixed understanding operations to generation parameters.
    llm_config.freeze_und = training_args.freeze_und and not training_args.freeze_shared_for_i2i
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
            
    ignored_modules = []
    # Freeze language model layers (with support for partial freezing)
    total_llm_layers = len(model.language_model.model.layers)
    if (training_args.freeze_llm_first_n_layers is not None or 
        training_args.freeze_llm_last_n_layers is not None or 
        training_args.freeze_llm_layers_ratio is not None or 
        training_args.freeze_llm_layers_ratio_from_end is not None):
        # Calculate number of layers to freeze
        if training_args.freeze_llm_first_n_layers is not None:
            num_layers_to_freeze = training_args.freeze_llm_first_n_layers
            freeze_from_start = True
        elif training_args.freeze_llm_last_n_layers is not None:
            num_layers_to_freeze = training_args.freeze_llm_last_n_layers
            freeze_from_start = False
        elif training_args.freeze_llm_layers_ratio is not None:
            # Use ratio from start
            num_layers_to_freeze = int(total_llm_layers * training_args.freeze_llm_layers_ratio)
            freeze_from_start = True
        else:  # freeze_llm_layers_ratio_from_end is not None
            # Use ratio from end
            num_layers_to_freeze = int(total_llm_layers * training_args.freeze_llm_layers_ratio_from_end)
            freeze_from_start = False
        
        direction_str = "first" if freeze_from_start else "last"
        logger.info(f"Freezing {direction_str} {num_layers_to_freeze} out of {total_llm_layers} language model layers")
        freeze_layers(model.language_model.model, num_layers_to_freeze, layer_attr_name="layers", freeze_from_start=freeze_from_start)
    
    
    # Freeze ViT layers (with support for partial freezing)
    if training_args.visual_und:
        vit_encoder = model.vit_model.vision_model.encoder
        vit_layers = vit_encoder.layers
        total_vit_layers = len(vit_layers)
        
        if (training_args.freeze_vit_first_n_layers is not None or 
            training_args.freeze_vit_last_n_layers is not None or 
            training_args.freeze_vit_layers_ratio is not None or 
            training_args.freeze_vit_layers_ratio_from_end is not None):
            # Calculate number of layers to freeze
            if training_args.freeze_vit_first_n_layers is not None:
                num_layers_to_freeze = training_args.freeze_vit_first_n_layers
                freeze_from_start = True
            elif training_args.freeze_vit_last_n_layers is not None:
                num_layers_to_freeze = training_args.freeze_vit_last_n_layers
                freeze_from_start = False
            elif training_args.freeze_vit_layers_ratio is not None:
                # Use ratio from start
                num_layers_to_freeze = int(total_vit_layers * training_args.freeze_vit_layers_ratio)
                freeze_from_start = True
            else:  # freeze_vit_layers_ratio_from_end is not None
                # Use ratio from end
                num_layers_to_freeze = int(total_vit_layers * training_args.freeze_vit_layers_ratio_from_end)
                freeze_from_start = False
            
            direction_str = "first" if freeze_from_start else "last"
            logger.info(f"Freezing {direction_str} {num_layers_to_freeze} out of {total_vit_layers} ViT layers")
            freeze_layers(vit_encoder, num_layers_to_freeze, layer_attr_name="layers", freeze_from_start=freeze_from_start)
        elif training_args.freeze_vit:
            # Freeze entire ViT
            model.vit_model.eval()
            for param in model.vit_model.parameters():
                param.requires_grad = False

        # The connector and position embedding live outside the understanding MoT branch.
        if training_args.freeze_und:
            model.connector.eval()
            for module in (model.connector, model.vit_pos_embed):
                for param in module.parameters():
                    param.requires_grad = False

    # Input-LN freeze ablation. Apply after the ordinary component/layer flags so
    # it composes with the current focus R2 setting (freeze the last half of LLM
    # layers) while leaving every non-target parameter exactly as that recipe sets it.
    if training_args.freeze_llm_input_ln:
        frozen_names, frozen_param_count = freeze_llm_input_layernorm(model)
        logger.info(
            f"[freeze_llm_input_ln] froze {len(frozen_names)} understanding-side "
            f"input_layernorm tensors ({frozen_param_count} params); all other "
            "parameters follow the ordinary freeze flags."
        )
        logger.info(f"[freeze_llm_input_ln] examples: {frozen_names[:3]}")

    # Probe setting: train ONLY the LLM input_layernorm weights. Applied LAST so it
    # overrides every other freeze flag: freeze all model params, then unfreeze the
    # input_layernorm weights. (use_orig_params=True in FSDP allows this mix.)
    if training_args.train_only_input_ln:
        n_unfrozen, unfrozen_names = 0, []
        for name, param in model.named_parameters():
            # understanding-branch input_layernorm only (exclude the generation moe_gen variant;
            # it gets no gradient in the i2t stage anyway, but keep the set precise).
            if "input_layernorm" in name and "_moe_gen" not in name:
                param.requires_grad = True
                n_unfrozen += param.numel()
                unfrozen_names.append(name)
            else:
                param.requires_grad = False
        logger.info(f"[train_only_input_ln] unfroze {len(unfrozen_names)} input_layernorm "
                    f"tensors ({n_unfrozen} params); everything else frozen.")
        logger.info(f"[train_only_input_ln] examples: {unfrozen_names[:3]}")

    if training_args.freeze_shared_for_i2i:
        if not training_args.visual_gen or training_args.train_only_input_ln:
            raise ValueError("freeze_shared_for_i2i requires generation and cannot combine with input-LN-only training")
        freeze_shared_for_i2i(model)
    if dist.get_rank() == 0:
        receipt = parameter_receipt(model)
        receipt["freeze_settings"] = {name: value for name, value in vars(training_args).items()
                                      if name.startswith("freeze_") or name == "train_only_input_ln"}
        receipt["policy"] = "generation-only" if training_args.freeze_shared_for_i2i else "configured"
        with open(os.path.join(training_args.results_dir, "trainable_parameters.json"), "w") as handle:
            json.dump(receipt, handle, indent=2)

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard
    )
    ema_model = None if training_args.train_with_no_ema else deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config,  ignored_modules=ignored_modules)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    def _make_packed_dataset(grouped_datasets, data_status_=None,
                             conditioning_dropout_prob=None, phase_name='single'):
        cfg = DataConfig(grouped_datasets=grouped_datasets)
        if training_args.visual_und:
            cfg.vit_patch_size = model_args.vit_patch_size
            cfg.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        cfg.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        cfg.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        cfg.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
        if conditioning_dropout_prob is not None:
            if not 0 <= conditioning_dropout_prob <= 1:
                raise ValueError('conditioning_dropout_prob must be between 0 and 1')
            cfg.text_cond_dropout_prob = conditioning_dropout_prob
            cfg.vae_cond_dropout_prob = conditioning_dropout_prob
            cfg.vit_cond_dropout_prob = conditioning_dropout_prob
        if dist.get_rank() == 0:
            logger.info(f'[conditioning-dropout] stage={phase_name} '
                        f'text={cfg.text_cond_dropout_prob} '
                        f'vae={cfg.vae_cond_dropout_prob} '
                        f'vit={cfg.vit_cond_dropout_prob}')
        if training_args.visual_gen:
            vae_image_downsample_ = model_args.latent_patch_size * vae_config.downsample
            cfg.vae_image_downsample = vae_image_downsample_
            cfg.max_latent_size = model_args.max_latent_size
        return PackedDataset(
            cfg,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_workers=data_args.num_workers,
            expected_num_tokens=training_args.expected_num_tokens,
            max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
            max_num_tokens=data_args.max_num_tokens,
            max_buffer_size=data_args.max_buffer_size,
            prefer_buffer_before=data_args.prefer_buffer_before,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=data_status_,
        )

    if 'curriculum' in dataset_meta:
        # Repeated phases share parsed data to bound host memory. Different dropout
        # settings need separate packers even when they use the same dataset YAML.
        phases = []
        config_dir = os.path.dirname(os.path.abspath(data_args.dataset_config_file))
        _phase_ds_cache = {}
        for phase_index, phase in enumerate(dataset_meta['curriculum']):
            phase_cfg_path = os.path.join(config_dir, phase['dataset_config_file'])
            dropout = phase.get('conditioning_dropout_prob')
            cache_key = (phase_cfg_path, dropout)
            if cache_key not in _phase_ds_cache:
                with open(phase_cfg_path, "r") as f:
                    phase_meta = yaml.safe_load(f)
                _phase_ds_cache[cache_key] = _make_packed_dataset(
                    phase_meta, conditioning_dropout_prob=dropout,
                    phase_name=f'phase{phase_index}',
                )
            phases.append((_phase_ds_cache[cache_key], phase['num_samples'],
                           phase.get('target_dataset_name', '')))
        if dist.get_rank() == 0:
            print(f"[curriculum] {len(phases)} phases share {len(_phase_ds_cache)} unique datasets")
        train_dataset = CurriculumPackedDataset(phases)
    else:
        train_dataset = _make_packed_dataset(dataset_meta, data_status)

    train_dataset.set_epoch(data_args.data_seed)
    # Opt-in (DETERMINISTIC=1): pin the DataLoader worker RNG with a fixed generator + seed
    # numpy/random per worker (PyTorch only seeds torch in workers, not numpy/random).
    _dl_kwargs = {}
    if os.environ.get("DETERMINISTIC") == "1":
        def _seed_worker(worker_id):
            import numpy as _np, random as _rnd
            s = (torch.initial_seed() + worker_id) % (2 ** 32)
            _np.random.seed(s); _rnd.seed(s)
        _dl_gen = torch.Generator(); _dl_gen.manual_seed(training_args.global_seed)
        _dl_kwargs = dict(generator=_dl_gen, worker_init_fn=_seed_worker)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
        **_dl_kwargs,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    cumulative_data_num = 0
    last_sanity_epoch_dumped = -1
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    if training_args.save_at_data_nums.strip():
        save_at_data_nums = sorted({
            int(item.strip())
            for item in training_args.save_at_data_nums.replace(";", ",").split(",")
            if item.strip()
        })
    else:
        save_at_data_nums = []
    next_save_at_idx = 0
    saved_checkpoint_steps = set()

    def _save_training_checkpoint(step, label):
        if step < 0:
            return
        if step in saved_checkpoint_steps:
            logger.info(f"Skipping {label} checkpoint at step {step}: already saved.")
            return

        logger.info(f"Saving {label} checkpoint at step {step}...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during gather_object for {label} checkpoint at step {step}: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir,
            train_steps=step,
            model=fsdp_model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list,
            save_ema_only=training_args.save_ema_only if ema_model is not None else False,
        )
        saved_checkpoint_steps.add(step)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)

        # Sanity check: dump first batch (rank 0 only).
        # - default (sanity_dump_every_samples=0): only at micro_step==0
        # - when set (e.g. 1000): also dump at each epoch boundary into sanity_check/epoch_NNN/
        if dist.get_rank() == 0:
            sanity_save_dir = None
            if training_args.sanity_dump_every_samples > 0:
                epoch_num = cumulative_data_num // training_args.sanity_dump_every_samples
                if epoch_num > last_sanity_epoch_dumped:
                    sanity_save_dir = os.path.join(
                        training_args.checkpoint_dir,
                        "sanity_check",
                        f"epoch_{epoch_num:03d}",
                    )
                    last_sanity_epoch_dumped = epoch_num
            elif micro_step == 0:
                sanity_save_dir = os.path.join(training_args.checkpoint_dir, "sanity_check")
            if sanity_save_dir is not None:
                save_sanity_check(
                    data=data,
                    data_indexes=data_indexes,
                    tokenizer=tokenizer,
                    save_dir=sanity_save_dir,
                    logger=logger,
                )

        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    if 'padded_images' in data:
                        data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                loss_dict = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            # assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            mse_count = len(data['mse_loss_indexes']) if 'mse_loss_indexes' in data else 0
            total_mse_tokens = torch.tensor(mse_count, device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            if mse_count > 0:
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            else:
                mse = mse.sum()  # dummy zero, keeps llm2vae in gradient graph
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        # Exact one-batch GPU audit for the T2I->I2T experiment.  This is a
        # strict opt-in hook: ordinary training does not import the audit code
        # and follows the original optimizer/checkpoint path unchanged.
        if os.environ.get("T2I_I2T_ONE_BATCH_AUDIT_REPORT_DIR"):
            from experiments.t2i_helps_i2t.gpu_preflight import write_rank_evidence

            write_rank_evidence(
                model=fsdp_model,
                batch=data,
                batch_data_indexes=data_indexes,
                loss_dict=loss_dict,
                total_loss=loss,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
            )
            dist.barrier()
            if dist.get_rank() == 0:
                logger.info("T2I_I2T_ONE_BATCH_AUDIT_COMPLETE")
            dist.destroy_process_group()
            return

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()

        # Count samples every step (needed for total_data_num stopping)
        if training_args.target_dataset_name:
            target_samples = sum(
                1 for item in data_indexes if item['dataset_name'] == training_args.target_dataset_name
            )
            total_samples = torch.tensor(target_samples, device=device)
        else:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        cumulative_data_num += total_samples.item()

        if training_args.total_data_num > 0 and cumulative_data_num >= training_args.total_data_num:
            target_info = f" (dataset={training_args.target_dataset_name})" if training_args.target_dataset_name else ""
            logger.info(f"Reached total_data_num={training_args.total_data_num}{target_info} (consumed {cumulative_data_num}), stopping.")
            break

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['cumulative_data_num'] = cumulative_data_num
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            # Per-step per-dataset sample accounting (rank-local view on rank 0 only).
            # This is a cheap debug aid for silent sharding/curriculum-phase bugs: if a
            # curriculum phase is supposed to serve dataset X but rank 0's dataloader
            # keeps yielding dataset Y (or yields 0 samples from X), it will show up
            # here immediately. Aggregation across ranks intentionally skipped to avoid
            # per-step collective overhead; cross-rank checks can be inferred from logs
            # of multiple ranks or from `data_status` at checkpoint time.
            try:
                rank0_dataset_counts = {}
                rank0_dataset_workers = {}
                for _item in (data_indexes or []):
                    _ds = _item.get('dataset_name', '?') if isinstance(_item, dict) else '?'
                    _wid = _item.get('worker_id', -1) if isinstance(_item, dict) else -1
                    rank0_dataset_counts[_ds] = rank0_dataset_counts.get(_ds, 0) + 1
                    rank0_dataset_workers.setdefault(_ds, set()).add(_wid)
                if dist.get_rank() == 0:
                    _parts = [
                        f"{_ds}:n={_n},w={sorted(rank0_dataset_workers[_ds])}"
                        for _ds, _n in sorted(rank0_dataset_counts.items())
                    ]
                    _data_msg = (
                        f"(step={curr_step:07d}) [data-accounting rank0] "
                        f"cumulative={cumulative_data_num} "
                        f"step_samples={total_samples.item()} "
                        f"per_dataset={{{'; '.join(_parts) if _parts else '<empty>'}}}"
                    )
                    logger.info(_data_msg)
                    print(_data_msg, flush=True)
            except Exception as _e:
                if dist.get_rank() == 0:
                    logger.warning(f"data-accounting log failed at step {curr_step}: {_e}")

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        while next_save_at_idx < len(save_at_data_nums) and cumulative_data_num >= save_at_data_nums[next_save_at_idx]:
            target_data_num = save_at_data_nums[next_save_at_idx]
            logger.info(
                f"Reached save_at_data_num={target_data_num} "
                f"(consumed {cumulative_data_num}), saving early checkpoint."
            )
            _save_training_checkpoint(curr_step, f"data{target_data_num}")
            next_save_at_idx += 1

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            _save_training_checkpoint(curr_step, "interval")

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step >= 0:
        _save_training_checkpoint(curr_step, "final")
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    if dist.get_rank() == 0:
        with open(os.path.join(training_args.results_dir, "completion.json"), "w") as handle:
            json.dump({"checkpoint": os.path.abspath(os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}")),
                       "sample_visits": int(cumulative_data_num),
                       "optimizer_updates": (micro_step + 1) // training_args.gradient_accumulation_steps,
                       "pending_microbatches": (micro_step + 1) % training_args.gradient_accumulation_steps}, handle, indent=2)
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
        # Restore stdout/stderr and close log files
        if isinstance(sys.stdout, Tee):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        if isinstance(sys.stderr, Tee):
            sys.stderr.close()
            sys.stderr = sys.__stderr__
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
