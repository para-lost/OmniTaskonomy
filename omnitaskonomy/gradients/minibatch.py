"""Token-weighted raw minibatch gradients in the frozen reference PCA bases."""

from collections import defaultdict
import hashlib

import numpy as np

from omnitaskonomy.gradients.artifacts import stable_seed
from omnitaskonomy.gradients.pca import project_artifact, unit_rows


def sampling_schedule(key, fold, n, repeats=2000, seed=2026091607):
    if n < 1 or repeats < 2:
        raise ValueError("Minibatches require a nonempty fold and at least two draws")
    rng = np.random.default_rng(stable_seed(seed, f"point:{key}:fold:{fold}"))
    # This width preserves the historical streams for all batch-size prefixes.
    return rng.integers(n, size=(repeats, 128), dtype=np.int32)


def minibatch_directions(projected, tokens, indices, batch_size=64):
    if not 1 <= batch_size <= indices.shape[1]:
        raise ValueError("Batch size is outside the stored sampling schedule")
    tokens = np.asarray(tokens, dtype=float)
    if not np.isfinite(tokens).all() or np.any(tokens <= 0):
        raise ValueError("Loss-token weights must be positive and finite")
    counts = np.zeros((len(indices), len(projected)), dtype=float)
    np.add.at(counts, (np.arange(len(indices))[:, None], indices[:, :batch_size]), 1)
    return unit_rows(counts @ (projected * tokens[:, None]))


def analyze_minibatches(artifact, reference, bases, *, repeats=2000, batch_size=64, seed=2026091607):
    if set(bases) != set(range(artifact["metadata"]["folds"])):
        raise ValueError("PCA bases do not cover every query fold")
    if artifact["metadata"]["state"] != "base" or reference["metadata"]["state"] != "base":
        raise ValueError("Cross-task gradient alignment must use base-checkpoint gradients")
    if artifact["metadata"]["provenance"]["checkpoint_sha256"] != reference["metadata"]["provenance"]["checkpoint_sha256"]:
        raise ValueError("Source, target and reference gradients use different checkpoints")
    rows = artifact["metadata"]["rows"]
    known = defaultdict(set)
    for row in reference["metadata"]["rows"]:
        known["source:" + row["source_uid"]].add(row["fold"])
        for digest in row.get("image_hashes", []):
            known["image:" + digest].add(row["fold"])
    for row in rows:
        identities = ["source:" + row["source_uid"]]
        if row["objective"] == "i2t":
            identities += ["image:" + digest for digest in row.get("image_hashes", [])]
        for identity in identities:
            if known[identity] and known[identity] != {row["fold"]}:
                raise ValueError("Query/reference source or decoded image crosses PCA folds")
            known[identity].add(row["fold"])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["task"], row["objective"]].append(i)
    sampling_keys = {}
    for key, indexes in groups.items():
        streams = {rows[i].get("sampling_key", ".".join(key)) for i in indexes}
        if len(streams) != 1:
            raise ValueError("One gradient pool has multiple sampling streams")
        sampling_keys[key] = streams.pop()
    if len(set(sampling_keys.values())) != len(groups):
        raise ValueError("Different gradient pools must use independent sampling streams")
    sources = sorted(key for key in groups if key[1] == "i2i")
    targets = sorted(key for key in groups if key[1] == "i2t")
    if not sources or not targets:
        raise ValueError("Cross-task analysis requires I2I and I2T pools")
    means = np.zeros((len(sources), len(targets)))
    variances = np.zeros_like(means)
    prefix = np.zeros_like(means)
    schedules, fold_rows = [], []
    for fold, fit in sorted(bases.items()):
        projected = project_artifact(artifact, reference, fit)
        directions, weights = {}, {}
        for key, indexes in groups.items():
            selected = [i for i in indexes if rows[i]["fold"] == fold]
            schedule = sampling_schedule(sampling_keys[key], fold, len(selected), repeats, seed)
            tokens = np.array([rows[i]["loss_tokens"] for i in selected])
            directions[key] = minibatch_directions(projected[selected], tokens, schedule, batch_size)
            weights[key] = len(selected) / len(indexes)
            schedules.append({"task": key[0], "objective": key[1], "fold": fold, "n": len(selected),
                              "sampling_key": sampling_keys[key],
                              "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest()})
        left = np.stack([directions[key] for key in sources], axis=1)
        right = np.stack([directions[key] for key in targets], axis=1)
        cosines = np.einsum("rsi,rti->rst", left, right)
        weights_array = np.array([weights[key] for key in targets])[None, :]
        fold_mean = cosines.mean(axis=0)
        means += fold_mean * weights_array
        variances += cosines.var(axis=0, ddof=1) * weights_array**2 / repeats
        prefix += cosines[:repeats // 2].mean(axis=0) * weights_array
        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                fold_rows.append({"source_task_id": source[0], "node_id": target[0], "fold": fold,
                                  "alignment": float(fold_mean[i, j]), "weight": float(weights_array[0, j])})
    cells = []
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            cells.append({"source_task_id": source[0], "node_id": target[0], "alignment": float(means[i, j]),
                          "mcse": float(np.sqrt(variances[i, j])), "prefix_difference": float(means[i, j] - prefix[i, j]),
                          "n_source": len(groups[source]), "n_target": len(groups[target]),
                          "batch_i2i": batch_size, "batch_i2t": batch_size})
    return {"schema_version": 1, "cells": cells, "folds": fold_rows, "schedules": schedules,
            "repeats_per_fold": repeats, "sampling_seed": seed,
            "method": "token_weighted_raw_gradient_then_reference_pca_then_unit_norm",
            "input_audit": {"gradients": artifact["audit"], "reference": reference["audit"],
                            "bases": {str(f): fit["sha256"] for f, fit in bases.items()}}}
