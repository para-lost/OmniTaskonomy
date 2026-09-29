"""Matched per-example raw magnitudes and losses across I2I-only checkpoints."""

from collections import defaultdict

import numpy as np

from omnitaskonomy.gradients.artifacts import parameter_blocks


def summarize_norms(artifacts, *, paper=False):
    samples, layer_rows, summary, identities = [], [], [], {}
    seen = set()
    for artifact in artifacts:
        meta = artifact["metadata"]
        state = meta["state"]
        rows = meta["rows"]
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["task"], row["objective"]].append(i)
            key = row["task"], row["objective"], row["uid"]
            identity = (row["source_uid"], row["fold"], row.get("noise_seed"), tuple(row.get("image_hashes", [])))
            if identities.setdefault(key, identity) != identity:
                raise ValueError("Checkpoint diagnostics changed source, fold, image, or noise draw")
        parameters = {p["name"]: p for p in meta["parameters"]}
        for (module, _), names in parameter_blocks(artifact, layers=False).items():
            if any(parameters[name]["representation"] != "raw" for name in names):
                raise ValueError("Magnitude diagnostics require gradients in original coordinates")
            squared = np.zeros(len(rows))
            per_parameter = {}
            for name in names:
                raw = np.asarray(artifact["arrays"][name], dtype=float)
                norms = np.linalg.norm(raw, axis=1)
                per_parameter[name] = norms
                squared += norms**2
            raw_norms = np.sqrt(squared)
            for (task, objective), indexes in groups.items():
                key = task, objective, module, state
                if key in seen:
                    raise ValueError(f"Duplicate checkpoint diagnostic: {key}")
                seen.add(key)
                selected = raw_norms[indexes]
                losses = [rows[i]["loss"] for i in indexes]
                identity = {"task": task, "objective": objective, "module": module, "state": state}
                summary.append({**identity, "n": len(indexes), "mean_loss": float(np.mean(losses)),
                                "mean_l2": float(selected.mean()), "median_l2": float(np.median(selected)),
                                "rms_l2": float(np.sqrt(np.mean(selected**2)))})
                for i in indexes:
                    samples.append({**identity, "uid": rows[i]["uid"], "fold": rows[i]["fold"],
                                    "loss": rows[i]["loss"], "raw_l2": float(raw_norms[i])})
                for name in names:
                    layer_rows.append({**identity, "parameter": name, "layer": parameters[name]["layer"],
                                       "n": len(indexes), "mean_l2": float(per_parameter[name][indexes].mean())})
    by_group = defaultdict(dict)
    for row in summary:
        by_group[row["task"], row["objective"], row["module"]][row["state"]] = row
    for key, states in by_group.items():
        if "base" not in states:
            raise ValueError(f"Missing base checkpoint for {key}")
        if paper and (set(states) != {"base", "3k", "10k", "30k"} or any(r["n"] != 500 for r in states.values())):
            raise ValueError("Paper norms require the same 500 examples at base/3k/10k/30k")
        expected = {r["uid"] for r in samples if (r["task"], r["objective"], r["module"]) == key and r["state"] == "base"}
        for state, row in states.items():
            actual = {r["uid"] for r in samples if (r["task"], r["objective"], r["module"]) == key and r["state"] == state}
            if actual != expected:
                raise ValueError("Checkpoint diagnostic sample sets differ")
            base = states["base"]["mean_l2"]
            row["mean_l2_ratio_to_base"] = row["mean_l2"] / base if base else None
    return {"schema_version": 1, "summary": summary, "samples": samples, "layers": layer_rows,
            "input_audit": [artifact["audit"] for artifact in artifacts]}
