"""Completion receipts for the experimental VLMEvalKit runner."""

from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path


def _flatten_scores(scores):
    if isinstance(scores, dict):
        rows = [scores]
    else:
        from pandas import DataFrame

        if not isinstance(scores, DataFrame):
            raise TypeError("Evaluation scores must be a DataFrame or metric dictionary")
        rows = scores.to_dict(orient="records")

    metrics = {}
    for row in rows:
        prefix = f"split={row['split']}|" if "split" in row else ""
        for column, value in row.items():
            if column == "split":
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Evaluation metric {column!r} is not numeric: {value!r}")
            value = float(value)
            # A category can be absent from one split in the MMT score table.
            if math.isnan(value):
                continue
            if not math.isfinite(value):
                raise ValueError(f"Evaluation metric {column!r} is not finite")
            key = prefix + str(column)
            if key in metrics:
                raise ValueError(f"Duplicate evaluation metric: {key}")
            metrics[key] = value
    return metrics


def record_evaluation(run_dir, dataset_name, status, *, judge=None, scores=None,
                      skip_reason=None, error_message=None):
    """Called only by rank zero; replace each dataset's receipt on every transition."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "status.json"
    receipt = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "datasets": {}}
    row = {"status": status}
    if judge is not None:
        row["judge_model"] = judge
    if scores is not None:
        row["metrics"] = _flatten_scores(scores)
    if skip_reason is not None:
        row["skip_reason"] = skip_reason
    if error_message is not None:
        row["error_message"] = str(error_message)
    receipt["datasets"][dataset_name] = row
    receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return receipt
