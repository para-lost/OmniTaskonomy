"""Per-example UMM gradients with fixed noise and no optimizer updates."""

import hashlib
from pathlib import Path

import numpy as np

from omnitaskonomy.adapters.bagel import classify_parameter, objective_loss, _seed
from omnitaskonomy.data.common import sha256
from omnitaskonomy.gradients.artifacts import stable_seed, write_json
from omnitaskonomy.gradients.manifest import verify_manifest
from omnitaskonomy.umm import BAGEL_ADAPTER, LossContext, ParameterInventory, adapter_provenance, load_adapter


def countsketch(gradient, dimension, seed):
    import torch
    generator = torch.Generator(device=gradient.device).manual_seed(seed)
    n = gradient.numel()
    index = torch.randint(dimension, (n,), generator=generator, device=gradient.device, dtype=torch.int32)
    sign = torch.randint(2, (n,), generator=generator, device=gradient.device, dtype=torch.int8).mul_(2).sub_(1)
    result = torch.zeros(dimension, device=gradient.device, dtype=torch.float32)
    result.scatter_add_(0, index.long(), gradient.detach().float().flatten() * sign)
    positions = torch.arange(min(4096, n), device=gradient.device, dtype=torch.int64) * (n - 1) // (min(4096, n) - 1)
    sentinels = torch.stack((positions, index[positions].long(), sign[positions].long())).cpu().numpy()
    return result, hashlib.sha256(sentinels.tobytes()).hexdigest()


def extract(manifest_path, model_path, output, *, checkpoint=None, state="base", device="cuda:0",
            modules=("llm.input_ln.weight",), sketch_seed=2026082602, adapter=None, adapter_options=None,
            reference_identity=None):
    import torch
    manifest_path, model_path, output = Path(manifest_path).resolve(), Path(model_path).resolve(), Path(output)
    frozen = verify_manifest(manifest_path)
    factory = adapter or BAGEL_ADAPTER
    options = dict(adapter_options or {})
    if adapter is None:
        options["recompute_latent_positions"] = True
    runtime = load_adapter(factory, model_path, checkpoint, device, options)
    identity = adapter_provenance(runtime, factory, options)
    if reference_identity is not None and identity["checkpoint_sha256"] != reference_identity:
        raise ValueError("Matrix model or adapter differs from the reference checkpoint")
    weights_hash = identity["checkpoint_sha256"] if adapter else sha256(runtime.checkpoint_files[0])
    manifest_hash = sha256(manifest_path)
    root = Path(__file__).resolve().parents[2]
    source_files = ["omnitaskonomy/gradients/extract.py", "omnitaskonomy/umm.py"]
    if factory == BAGEL_ADAPTER:
        source_files += ["omnitaskonomy/adapters/bagel.py", "omnitaskonomy/bagel.py", "omnitaskonomy/datasets.py",
                         "Bagel/data/dataset_base.py", "Bagel/data/interleaved_base.py", "Bagel/data/transforms.py",
                         "Bagel/modeling/bagel/bagel.py", "Bagel/modeling/bagel/qwen2_navit.py",
                         "Bagel/modeling/bagel/modeling_utils.py", "Bagel/modeling/autoencoder.py"]
    implementation_hashes = {name: sha256(root / name) for name in source_files}
    parameter_inventory = ParameterInventory(runtime.model, runtime.parameter_specs())
    runtime.model.eval()
    for layer in runtime.model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
    parameters, inventory = {}, []
    for name, (value, spec, _) in parameter_inventory.entries.items():
        selected = spec.module is not None and ("all" in modules or spec.module in modules)
        value.requires_grad_(selected)
        if not selected:
            continue
        parameters[name] = value
        limit = 8192 if spec.module.startswith("vit.") else 16384
        exact = "norm" in name or "layernorm" in name or value.numel() <= limit
        inventory.append({"name": name, "module": spec.module, "layer": spec.layer, "role": spec.role,
                          "zero_objectives": list(spec.zero_objectives),
                          "structural_zero_i2i": "i2i" in spec.zero_objectives,
                          "shape": list(value.shape), "numel": value.numel(),
                          "representation": "raw" if exact else "countsketch",
                          "stored_dim": value.numel() if exact else limit,
                          "sketch_seed": None if exact else stable_seed(sketch_seed, name),
                          "sketch_method": None if exact else "torch_generator_two_randint_streams_v1"})
    if not inventory or ("all" not in modules and set(modules) - {p["module"] for p in inventory}):
        raise ValueError("Requested gradient modules do not match model parameters")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for index, parameter in enumerate(inventory):
        parameter["file"] = f"parameter_{index:04d}.npy"
        arrays[parameter["name"]] = np.lib.format.open_memmap(output / parameter["file"], mode="w+", dtype=np.float32,
                                               shape=(len(frozen["rows"]), parameter["stored_dim"]))
    rows = []
    for index, row in enumerate(frozen["rows"]):
        noise_seed = row["loss_seed"] if row.get("loss_seed") is not None else stable_seed(
            frozen["sample_seed"], f"loss:{row['task']}:{row['objective']}:{row['uid']}") % (2**32 - 1)
        data_seed = row["data_seed"] if row.get("data_seed") is not None else stable_seed(
            frozen["sample_seed"], f"data:{row['task']}:{row['objective']}:{row['uid']}") % (2**32 - 1)
        _seed(data_seed)
        runtime.model.zero_grad(set_to_none=True)
        context = LossContext(manifest_path.parent / row["manifest"], noise_seed,
                              target_interpolation=row["target_interpolation"], vit_transform=row.get("vit_transform"))
        with torch.enable_grad():
            result = runtime.loss([row["record"]], row["objective"], context)
            loss, tokens = result.mean, result.count
            if loss.requires_grad:
                loss.backward()
            elif not all(row["objective"] in parameter["zero_objectives"] for parameter in inventory):
                raise ValueError(f"Detached gradient objective: {row['uid']}/{row['objective']}")
        raw_norms = {}
        for parameter in inventory:
            name = parameter["name"]
            gradient = parameters[name].grad
            if gradient is None:
                if row["objective"] in parameter["zero_objectives"]:
                    encoded = np.zeros(parameter["stored_dim"], dtype=np.float32)
                    raw_norms[name] = 0.
                else:
                    raise ValueError(f"Missing gradient: {row['uid']}/{name}")
            else:
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError(f"Non-finite gradient: {row['uid']}/{name}")
                raw_norms[name] = float(torch.linalg.vector_norm(gradient.detach().double()))
                if parameter["representation"] == "raw":
                    encoded = gradient.detach().float().flatten().cpu().numpy()
                else:
                    projected, fingerprint = countsketch(gradient, parameter["stored_dim"], parameter["sketch_seed"])
                    if parameter.setdefault("sketch_mapping_fingerprint", fingerprint) != fingerprint:
                        raise ValueError("CountSketch mapping changed between samples")
                    encoded = projected.cpu().numpy()
            arrays[name][index] = encoded
        rows.append({key: row[key] for key in ("task", "objective", "uid", "source_uid", "fold", "image_hashes")}
                    | {"loss": float(loss.detach()), "loss_tokens": tokens, "noise_seed": noise_seed, "raw_norms": raw_norms,
                       "sampling_key": row.get("sampling_key", f"{row['task']}.{row['objective']}")})
        if (index + 1) % 25 == 0:
            print(f"Gradient extraction: {index + 1}/{len(frozen['rows'])}", flush=True)
    for parameter in inventory:
        arrays[parameter["name"]].flush()
        parameter["sha256"] = sha256(output / parameter["file"])
    verify_manifest(manifest_path)
    if adapter_provenance(runtime, factory, options) != identity or sha256(manifest_path) != manifest_hash:
        raise ValueError("Checkpoint or manifest changed during extraction")
    write_json(output / "metadata.json", {"schema_version": 1, "status": "complete", "state": state,
               "folds": frozen["folds"], "rows": rows, "parameters": inventory,
               "provenance": {"manifest_sha256": manifest_hash, "checkpoint_sha256": weights_hash,
                              "adapter": identity, "checkpoint": str(runtime.checkpoint_files[0]),
                              "implementation_sha256": implementation_hashes, "conditioning_dropout": 0,
                              "stored_precision": "float32", "sketch_seed": sketch_seed,
                              "torch_version": torch.__version__, "numpy_version": np.__version__}})
    return output
