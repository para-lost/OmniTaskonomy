"""Exact question-block swaps for one to three seeds of binary correctness.

The integer recurrence is adapted from the research implementation at
gen4und/taxonomy/scripts/unitaskonomy_v15/exact_permutation.py.
"""

from collections import deque
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4096, typed=True)
def exact_weighted_swap(n1, n2, n3, observed):
    """Return exact tail counts for absolute weights 1, 2 and 3.

    The subset-sum polynomial is P(x)=(1+x)^n1 (1+x^2)^n2 (1+x^3)^n3.
    For A=(1+x)(1+x^2)(1+x^3), A P'=B P with B=A*(log P)'.
    Equating coefficients gives a six-term integer recurrence. The inclusive
    lower tail and symmetry give P(|sum(sign_i * weight_i)| >= observed).
    """
    if not all(type(v) is int and v >= 0 for v in (n1, n2, n3, observed)):
        raise ValueError("Counts and absolute observed statistic must be nonnegative integers")
    weight = n1 + 2*n2 + 3*n3
    if observed > weight or (weight - observed) % 2:
        raise ValueError("Observed statistic violates sign-swap bounds or parity")
    denominator = 1 << (n1 + n2 + n3)
    if observed == 0:
        return denominator, denominator
    cutoff = (weight - observed) // 2
    a = (1, 1, 1, 2, 1, 1, 1)
    b = (n1, 2*n2, n1+2*n2+3*n3, n1+3*n3, 2*n2+3*n3, n1+2*n2+3*n3)
    recent = deque([1], maxlen=6)
    tail = 1
    for k in range(1, cutoff + 1):
        numerator = sum((b[j-1] - a[j]*(k-j)) * coefficient
                        for j, coefficient in enumerate(reversed(recent), 1))
        coefficient, remainder = divmod(numerator, k)
        if remainder or coefficient < 0:
            raise ArithmeticError("Polynomial coefficient recurrence lost exactness")
        recent.append(coefficient)
        tail += coefficient
    extreme = 2 * tail
    if not 0 < extreme <= denominator:
        raise ArithmeticError("Invalid exact permutation tail mass")
    return extreme, denominator


def paired_test(source_hits, baseline_hits):
    """Swap all selected seeds together within each aligned question column.

    Both inputs have shape [seed, question] and contain only binary correctness.
    Callers must align seed IDs and question IDs before invoking this function.
    """
    source = np.asarray(source_hits)
    baseline = np.asarray(baseline_hits)
    if source.ndim != 2 or baseline.ndim != 2 or source.shape != baseline.shape:
        raise ValueError("Source and baseline hits must have the same [seed, question] shape")
    n_seeds, n_questions = source.shape
    if not 1 <= n_seeds <= 3:
        raise ValueError("Exact paired tests support one, two or three seeds")
    if n_questions < 1:
        raise ValueError("Exact paired tests require at least one question")
    for name, hits in (("Source", source), ("Baseline", baseline)):
        if hits.dtype.kind not in "buif" or not np.all((hits == 0) | (hits == 1)):
            raise ValueError(f"{name} correctness must contain only numeric 0 or 1")
    # Signed arithmetic preserves negative differences for bool/unsigned inputs.
    differences = (source.astype(np.int8) - baseline.astype(np.int8)).sum(axis=0)
    counts = [int(np.count_nonzero(np.abs(differences) == weight)) for weight in (1, 2, 3)]
    signed = int(differences.sum())
    numerator, denominator = exact_weighted_swap(*counts, abs(signed))
    return {
        "n_questions": n_questions,
        "n_seeds": n_seeds,
        "gain_pp": 100 * signed / (n_seeds * n_questions),
        "p_value": numerator / denominator,
        "p_numerator": str(numerator),
        "p_denominator": str(denominator),
        "significant": 20 * numerator < denominator,
        "direction": "positive" if signed > 0 else "negative" if signed < 0 else "zero",
    }
