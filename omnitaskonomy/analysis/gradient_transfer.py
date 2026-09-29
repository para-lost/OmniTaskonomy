"""Join measured gradient alignment to the released, seed-paired transfer matrix."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from omnitaskonomy.data.common import sha256
from omnitaskonomy.taxonomy import load_taxonomy

BALANCED7 = ("i2t:CATEGORY_INSTANCE", "i2t:COLOR_MATERIAL", "i2t:DEPTH_DISTANCE",
             "i2t:METRIC_3D_RELATION", "i2t:RELATIVE_3D_RELATION", "i2t:OBJECT_COUNTING",
             "i2t:CORRESPONDENCE_TRACKING")


def _correlations(rows, x="alignment", y="gain_mean_pp"):
    a, b = np.array([r[x] for r in rows], dtype=float), np.array([r[y] for r in rows], dtype=float)
    def correlation(left, right):
        return float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 and left.std() and right.std() else None
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]
    return {"n": len(rows), "pearson": correlation(a, b), "spearman": correlation(ranks(a), ranks(b))}


def import_estimates(path, taxonomy_path):
    """Explicit adapter for the published research matrix_v14/matrix_v15 schema."""
    path = Path(path)
    matrix = json.loads(path.read_text())
    taxonomy = load_taxonomy(taxonomy_path)
    sources = {r["source_id"]: r["id"] for r in taxonomy["tasks"] if r["modality"] == "i2i"}
    targets = {r["id"] for r in taxonomy["tasks"] if r["modality"] == "i2t"}
    cells, seen = [], set()
    for row in matrix["cells"]:
        source, target = sources[row["model_id"]], row["capability_id"]
        if target not in targets or (source, target) in seen or not np.isfinite(row["cosine"]) or not -1 <= row["cosine"] <= 1:
            raise ValueError("Invalid or duplicate imported gradient estimate")
        seen.add((source, target))
        cells.append({"source_task_id": source, "node_id": target, "alignment": row["cosine"],
                      "mcse": row["mcse"], "n_target": row["gradient_n"],
                      "batch_i2i": row["batch_i2i"], "batch_i2t": row["batch_i2t"]})
    expected = {(source, target) for source in sources.values() for target in targets}
    if seen != expected:
        raise ValueError("Research matrix import must contain all 19 × 25 cells")
    return {"schema_version": 1, "cells": cells, "method": "imported_research_estimates",
            "input_audit": {"path": str(path.resolve()), "sha256": sha256(path),
                            "taxonomy_provenance_sha256": sha256(Path(taxonomy_path) / "provenance.json"),
                            "verification_scope": "estimate identities and file hash; raw gradients not revalidated"}}


def associate(gradients, transfer, *, balanced_targets=BALANCED7):
    if gradients["schema_version"] != 1 or transfer["schema_version"] != 1 or not gradients["input_audit"]:
        raise ValueError("Gradient and transfer inputs require schema 1 and provenance")
    estimates = {}
    for row in gradients["cells"]:
        key = row["source_task_id"], row["node_id"]
        if key in estimates or not np.isfinite(row["alignment"]) or not -1 <= row["alignment"] <= 1:
            raise ValueError(f"Invalid or duplicate gradient estimate: {key}")
        estimates[key] = row
    cells, seen = [], set()
    for row in transfer["transfer_matrix"]:
        key = row["source_task_id"], row["node_id"]
        if key in seen or row["gain_mean_pp"] is None:
            raise ValueError(f"Duplicate or empty transfer cell: {key}")
        seen.add(key)
        estimate = estimates.get(key)
        cells.append({**row, "alignment": estimate["alignment"] if estimate else None,
                      "mcse": estimate.get("mcse") if estimate else None})
    if estimates.keys() - seen:
        raise ValueError("Gradient estimates refer to cells absent from the transfer matrix")
    selected = [row for row in cells if row["node_id"] in balanced_targets]
    sources = {row["source_task_id"] for row in cells}
    if {r["node_id"] for r in selected} != set(balanced_targets) or any(r["alignment"] is None for r in selected):
        raise ValueError("Missing gradient estimates for a balanced capability")
    if len(selected) != len(sources) * len(balanced_targets):
        raise ValueError("Balanced gradient-transfer matrix is incomplete")
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["node_id"]].append(row)
    means = [{"node_id": node, "n_sources": len(values),
              "mean_alignment": float(np.mean([r["alignment"] for r in values])),
              "mean_transfer_gain_pp": float(np.mean([r["gain_mean_pp"] for r in values]))}
             for node, values in grouped.items()]
    available = [row for row in cells if row["alignment"] is not None]
    return {"schema_version": 1, "cells": cells, "selected_cells": selected, "capability_means": means,
            "correlations": {"all": _correlations(available), "balanced7": _correlations(selected),
                             "capability_means": _correlations(means, "mean_alignment", "mean_transfer_gain_pp")},
            "input_audit": {"gradients": gradients["input_audit"], "transfer": transfer["input_audit"]}}
