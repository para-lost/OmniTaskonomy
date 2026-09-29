"""Generation-only parameter selection for Frozen I2I training."""


def parameter_role(name):
    generation_modules = ("vae2llm.", "llm2vae.", "time_embedder.")
    if name.startswith(generation_modules) or (name.startswith("language_model.") and "_moe_gen" in name):
        return "generation"
    if name.startswith(("vit_model.", "connector.", "vit_pos_embed.", "language_model.lm_head.",
                        "language_model.model.layers.", "language_model.model.norm.")):
        return "understanding"
    return "shared"


def freeze_shared_for_i2i(model):
    """Retain trainable generation experts and latent/time projections only."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(parameter.requires_grad and parameter_role(name) == "generation")


def parameter_receipt(model):
    return {"trainable": [{"name": name, "shape": list(parameter.shape),
                           "numel": parameter.numel()}
                          for name, parameter in model.named_parameters() if parameter.requires_grad],
            "frozen_numel": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
