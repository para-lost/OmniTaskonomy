"""Model adapters share losses and parameter roles across training and analysis."""

from dataclasses import dataclass, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from omnitaskonomy.data.common import sha256

BAGEL_ADAPTER = "omnitaskonomy.adapters.bagel:create_adapter"


@dataclass(frozen=True)
class Loss:
    """Sum over supervised units, with the denominator kept for gradient weighting."""

    total: Any
    count: int

    @property
    def mean(self):
        import torch

        if type(self.count) is not int or self.count <= 0:
            raise ValueError("Loss count must be a positive integer")
        if self.total.ndim != 0 or not bool(torch.isfinite(self.total)):
            raise ValueError("Loss total must be a finite scalar tensor")
        return self.total / self.count


@dataclass(frozen=True)
class LossContext:
    manifest: Path
    seed: int
    condition_dropout: float = 0.0
    target_interpolation: str = "bicubic"
    vit_transform: dict | None = None
    max_tokens: int = 16384


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    role: str
    module: str | None
    layer: int | None = None
    trainable: bool = True
    zero_objectives: tuple[str, ...] = ()


class ParameterInventory:
    def __init__(self, model, specs):
        aliases = dict(model.named_parameters(remove_duplicate=False))
        canonical = {id(parameter): name for name, parameter in model.named_parameters()}
        entries = {}
        for spec in specs:
            if spec.name not in aliases:
                raise ValueError(f"Unknown adapter parameter: {spec.name}")
            if spec.role not in {"generation", "understanding", "shared", "fixed"}:
                raise ValueError(f"Unknown parameter role: {spec.name}/{spec.role}")
            if set(spec.zero_objectives) - {"i2i", "i2t"}:
                raise ValueError(f"Unknown zero-gradient objective: {spec.name}")
            parameter = aliases[spec.name]
            name = canonical[id(parameter)]
            spec = replace(spec, name=name)
            if name in entries and entries[name][1] != spec:
                raise ValueError(f"Conflicting roles or metadata for tied parameter: {name}")
            eligible = spec.trainable and parameter.requires_grad and spec.role != "fixed"
            entries[name] = (parameter, spec, eligible)
        missing = set(canonical.values()) - entries.keys()
        if missing:
            raise ValueError(f"Adapter omitted parameters: {sorted(missing)}")
        self.entries = entries

    def apply(self, policy):
        if policy not in {"all", "generation"}:
            raise ValueError(f"Unknown trainable-parameter policy: {policy}")
        selected = {name for name, (_, spec, eligible) in self.entries.items()
                    if eligible and (policy == "all" or spec.role == "generation")}
        if not selected:
            raise ValueError(f"Model has no eligible parameters for {policy} training")
        for name, (parameter, _, _) in self.entries.items():
            parameter.requires_grad_(name in selected)
            parameter.grad = None
        return {"policy": policy, "trainable": [
            {"name": name, "role": spec.role, "shape": list(parameter.shape), "numel": parameter.numel()}
            for name, (parameter, spec, _) in self.entries.items() if name in selected],
            "frozen_numel": sum(parameter.numel() for name, (parameter, _, _) in self.entries.items()
                                if name not in selected)}

    def parameters(self):
        return [parameter for parameter, _, _ in self.entries.values() if parameter.requires_grad]


def read_options(path):
    options = json.loads(Path(path).read_text()) if path is not None else {}
    if not isinstance(options, dict):
        raise ValueError("Adapter options must be a JSON object")
    return options


def load_adapter(factory, model_path, checkpoint=None, device="cuda:0", options=None):
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("Adapter must identify a factory as package.module:create_adapter")
    create = getattr(importlib.import_module(module), name)
    adapter = create(model_path=Path(model_path), checkpoint=Path(checkpoint) if checkpoint else None,
                     device=device, options=options or {})
    for method in ("loss", "parameter_specs", "generate", "save_checkpoint"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"Adapter must implement {method}()")
    ParameterInventory(adapter.model, adapter.parameter_specs())
    if not adapter.checkpoint_files:
        raise ValueError("Adapter must list the checkpoint and assets used to load the model")
    return adapter


def adapter_provenance(adapter, factory, options=None):
    files = {str(Path(path).resolve()): sha256(path) for path in adapter.checkpoint_files}
    module_name = factory.split(":")[0]
    source = inspect.getsourcefile(importlib.import_module(module_name))
    identity = {"factory": factory, "options": options or {},
                "files": sorted(files.values()), "adapter_sha256": sha256(source) if source else None}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"checkpoint_sha256": digest, "checkpoint_files_sha256": files, **identity}
