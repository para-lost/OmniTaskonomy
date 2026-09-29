"""Read existing per-question VLMEvalKit scores without re-scoring predictions."""

from pathlib import Path
import re
import string

import pandas as pd

from omnitaskonomy.data.common import sha256


def _normalize_index(value):
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def read_scores(path, expected_sha256=None):
    path = Path(path).resolve()
    readers = {".xlsx": pd.read_excel, ".csv": pd.read_csv}
    if path.suffix.lower() not in readers:
        raise ValueError(f"Unsupported score format {path.suffix!r}: {path}; use .xlsx or .csv")
    before = sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError(f"Score SHA256 mismatch: {path}; expected {expected_sha256}, found {before}")
    required = {"index", "hit", "question", "answer"}
    wanted = required | set(string.ascii_uppercase) | {"log"}
    # Object dtype preserves textual IDs such as "001"; "NA" can be a real option or ID.
    frame = readers[path.suffix.lower()](
        path, usecols=lambda column: column in wanted, dtype=object, keep_default_na=False,
    )
    after = sha256(path)
    if after != before:
        raise ValueError(f"Score file changed while reading: {path}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns {sorted(missing)}: {path}")
    if frame.empty:
        raise ValueError(f"No score rows: {path}")
    options = [column for column in string.ascii_uppercase if column in frame]
    hits = pd.to_numeric(frame["hit"], errors="coerce")
    rows = {}
    for position, source in enumerate(frame.to_dict(orient="records")):
        uid = _normalize_index(source["index"])
        if not uid:
            raise ValueError(f"Empty question index in row {position + 2}: {path}")
        if uid in rows:
            raise ValueError(f"Duplicate question index UID {uid!r}: {path}")
        hit = hits.iloc[position]
        if hit not in (0, 1):
            raise ValueError(f"Nonbinary hit for UID {uid!r}: {source['hit']!r} in {path}")
        question, answer = _normalize_text(source["question"]), _normalize_text(source["answer"])
        if not question or not answer:
            raise ValueError(f"Missing question or answer for UID {uid!r}: {path}")
        row = {"hit": int(hit), "question": question, "answer": answer,
               "options": {column: _normalize_text(source[column]) for column in options}}
        if "log" in source:
            row["log"] = str(source["log"])
        rows[uid] = row
    return {"path": str(path), "sha256": before, "rows": rows}


def compare_questions(reference_rows, candidate_rows):
    missing = reference_rows.keys() - candidate_rows.keys()
    extra = candidate_rows.keys() - reference_rows.keys()
    if missing or extra:
        raise ValueError(f"Question-set drift: missing UIDs {sorted(missing)[:5]}, unexpected UIDs {sorted(extra)[:5]}")
    for uid, reference in reference_rows.items():
        candidate = candidate_rows[uid]
        for field in ("question", "answer", "options"):
            if reference[field] != candidate[field]:
                raise ValueError(f"Question {field} drift for UID {uid!r}: {reference[field]!r} != {candidate[field]!r}")
