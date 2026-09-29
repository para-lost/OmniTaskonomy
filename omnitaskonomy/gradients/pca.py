"""Five-fold shared, uncentered PCA in the dual (example Gram) space."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from omnitaskonomy.data.common import sha256
from omnitaskonomy.gradients.artifacts import gram, parameter_blocks, validate_rows, write_json


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Undefined gradient direction: zero or non-finite norm")
    return values / norms[:, None]


def fit_pca(raw_gram, rows, fold, *, energy=.99, rcond=1e-7, tie_rtol=1e-6):
    if not 0 < energy <= 1:
        raise ValueError("PCA retained energy must lie in (0, 1]")
    raw_gram = np.asarray(raw_gram, dtype=np.float64)
    if raw_gram.shape != (len(rows), len(rows)) or not np.isfinite(raw_gram).all():
        raise ValueError("Invalid gradient Gram matrix")
    norms = np.sqrt(np.maximum(np.diag(raw_gram), 0))
    if np.any(norms <= 1e-12):
        raise ValueError("Zero gradient has no PCA direction")
    if not np.allclose(raw_gram, raw_gram.T, rtol=1e-10, atol=1e-10):
        raise ValueError("Asymmetric gradient Gram matrix")
    train = np.array([i for i, row in enumerate(rows) if row["fold"] != fold], dtype=np.int64)
    groups = Counter((rows[i]["task"], rows[i]["objective"]) for i in train)
    expected = {(row["task"], row["objective"]) for row in rows}
    if set(groups) != expected:
        raise ValueError("PCA training fold is missing a task/objective stratum")
    weights = np.array([1 / (len(groups) * groups[rows[i]["task"], rows[i]["objective"]]) for i in train])
    scales = np.sqrt(weights) / norms[train]
    moment = raw_gram[np.ix_(train, train)] * scales[:, None] * scales[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh((moment + moment.T) / 2)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    positive = np.maximum(eigenvalues, 0)
    total = positive.sum()
    if total <= 0 or -np.minimum(eigenvalues, 0).sum() > 1e-10 * total:
        raise ValueError("PCA Gram has an invalid negative spectrum")
    rank = int(np.sum(positive > positive[0] * rcond))
    k = int(np.searchsorted(np.cumsum(positive[:rank]), energy * total)) + 1
    if k > rank:
        raise ValueError("Stable PCA spectrum cannot retain the requested energy")
    tolerance = max(tie_rtol * positive[k - 1], 10 * np.finfo(float).eps * positive[0])
    boundary = positive[k - 1]
    while k < rank and abs(positive[k] - boundary) <= tolerance:
        k += 1
    # Multiplied by raw cross-Grams, these coefficients project raw gradients.
    coefficients = scales[:, None] * eigenvectors[:, :k] / np.sqrt(positive[:k])
    projected_train = raw_gram[np.ix_(train, train)] @ coefficients
    directions = unit_rows(projected_train)
    angular_moment = (directions * weights[:, None]).T @ directions
    d_eff = float(np.trace(angular_moment)**2 / np.sum(angular_moment**2))
    return {"train": train, "coefficients": coefficients, "k": k,
            "eigenvalues": positive[:k], "captured_energy": float(positive[:k].sum() / total),
            "d_eff": d_eff}


def analyze_reference(artifact, output, *, layers=True, energy=.99):
    rows = artifact["metadata"]["rows"]
    nfolds = artifact["metadata"]["folds"]
    validate_rows(rows, paired=True, folds=nfolds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    folds, pairs, undefined = [], [], []
    parameters = {p["name"]: p for p in artifact["metadata"]["parameters"]}
    pair_indexes = defaultdict(dict)
    for i, row in enumerate(rows):
        pair_indexes[row["task"], row["uid"]][row["objective"]] = i
    for block_index, ((module, parameter_name), names) in enumerate(parameter_blocks(artifact, layers).items()):
        layer = parameters[parameter_name]["layer"] if parameter_name else None
        identity = {"module": module, "layer": layer, "parameter": parameter_name}
        raw = gram(artifact, names)
        zeros = np.flatnonzero(np.diag(raw) <= 1e-24)
        if len(zeros):
            declared = {"i2i", "i2t"}
            for name in names:
                parameter = parameters[name]
                legacy_zeros = ["i2i"] if parameter.get("structural_zero_i2i", False) else []
                declared.intersection_update(parameter.get("zero_objectives", legacy_zeros))
            if set(zeros) == {i for i, row in enumerate(rows) if row["objective"] in declared}:
                undefined.append({**identity, "reason": "structural zero gradient", "objectives": sorted(declared)})
                continue
            raise ValueError(f"Unexpected zero gradient in {module}/{parameter_name}")
        for fold in range(nfolds):
            fit = fit_pca(raw, rows, fold, energy=energy)
            coordinates = raw[:, fit["train"]] @ fit["coefficients"]
            directions = unit_rows(coordinates)
            filename = f"basis_{block_index:04d}_fold{fold}.npz"
            np.savez(output / filename, train=fit["train"], coefficients=fit["coefficients"],
                     eigenvalues=fit["eigenvalues"])
            folds.append({**identity, "fold": fold, "k": fit["k"], "energy": energy,
                          "captured_energy": fit["captured_energy"], "d_eff": fit["d_eff"],
                          "parameters": names, "basis_file": filename, "sha256": sha256(output / filename)})
            for (task, uid), indexes in pair_indexes.items():
                left, right = indexes["i2i"], indexes["i2t"]
                if rows[left]["fold"] != fold:
                    continue
                cosine = float(directions[left] @ directions[right])
                pairs.append({**identity, "task": task, "uid": uid, "fold": fold,
                              "cosine": cosine, "normalized_cosine": cosine * np.sqrt(fit["d_eff"]),
                              "no_pca_cosine": float(raw[left, right] / np.sqrt(raw[left, left] * raw[right, right]))})
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["module"], row["layer"], row["parameter"], row["task"]].append(row)
    summary = [{"module": module, "layer": layer, "parameter": parameter, "task": task, "n": len(values),
                "cosine_mean": float(np.mean([v["cosine"] for v in values])),
                "normalized_cosine_mean": float(np.mean([v["normalized_cosine"] for v in values]))}
               for (module, layer, parameter, task), values in grouped.items()]
    report = {"schema_version": 1, "method": "shared_uncentered_pca", "folds": folds, "pairs": pairs,
              "summary": summary,
              "undefined": undefined, "input_audit": artifact["audit"]}
    write_json(output / "report.json", report)
    return report


def load_basis(path, artifact, module="llm.input_ln.weight"):
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    if any(report["input_audit"][key] != artifact["audit"][key] for key in ("metadata_sha256", "arrays")):
        raise ValueError("PCA basis was fitted on a different reference artifact")
    selected = [r for r in report["folds"] if r["module"] == module and r["parameter"] is None]
    if {r["fold"] for r in selected} != set(range(artifact["metadata"]["folds"])):
        raise ValueError(f"Missing concatenated reference PCA for {module}")
    result = {}
    for row in selected:
        filename = path / row["basis_file"]
        if sha256(filename) != row["sha256"]:
            raise ValueError("PCA basis hash differs")
        with np.load(filename, allow_pickle=False) as values:
            result[row["fold"]] = {**row, "train": values["train"], "coefficients": values["coefficients"]}
    return result


def project_artifact(query, reference, fit):
    parameters = {p["name"]: p for p in query["metadata"]["parameters"]}
    refs = {p["name"]: p for p in reference["metadata"]["parameters"]}
    n = len(query["metadata"]["rows"])
    cross = np.zeros((n, len(fit["train"])), dtype=np.float64)
    for name in fit["parameters"]:
        for key in ("shape", "representation", "sketch_seed", "stored_dim"):
            if parameters[name][key] != refs[name][key]:
                raise ValueError(f"Gradient parameter spaces differ: {name}/{key}")
        for key in ("sketch_method", "sketch_mapping_fingerprint"):
            if parameters[name].get(key) != refs[name].get(key):
                raise ValueError(f"Gradient sketch maps differ: {name}/{key}")
        q, r = query["arrays"][name], reference["arrays"][name]
        for start in range(0, q.shape[1], 2048):
            qpart = np.asarray(q[:, start:start + 2048], dtype=np.float64)
            rpart = np.asarray(r[fit["train"], start:start + 2048], dtype=np.float64)
            cross += qpart @ rpart.T
    return cross @ fit["coefficients"]
