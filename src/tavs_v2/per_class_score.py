"""Cohort-relative per-class scoring for noisy-client detection.

Stdlib only (no torch, no flwr) so both the strategy and the standalone
analysis scripts can import it without pulling the training stack in.

Why this exists. Our pooled per-sample loss variance scores AUROC 0.734 on a
Dirichlet(0.3) partition and inverts outright on milder noise (0.436). FedNoRo
(IJCAI 2023, Sec. 1) diagnoses the cause: under class imbalance a CLEAN client
holding rare or hard classes produces a wide loss spread that mimics a noisy
one. Their fix is to stop pooling and score each class separately, normalised
against what the other clients report for that same class -- "though
inter-class samples are non-IID under class imbalance, intra-class samples are
more IID".

Measured on our own logs (non-IID Dirichlet 0.3, CIFAR-10N worst, 5 seeds):
    pooled variance        0.734
    pooled mean            0.682
    per-class mean         0.965   <- FedNoRo's statistic
    per-class variance     0.899   <- their decomposition, our statistic

The scoring is deliberately COHORT-RELATIVE: each class is min-max normalised
across the clients observed this round. That is what neutralises class
imbalance -- the question becomes "is this client's class-3 spread large
compared with other clients' class-3 spread", not "is it large in absolute
terms". The cost is that the score has no meaning for a single client in
isolation, and a cohort of entirely clean clients will still be spread across
the full [0, 1] range, so somebody is always ranked worst. With 40% of clients
noisy and 10 sampled per round, mixed cohorts are near-certain here; a
deployment with very few noisy clients would need an absolute threshold (or
FedNoRo's GMM, which can decide "all one cluster") instead.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def per_class_cohort_scores(
    cohort: Sequence[Tuple[str, Sequence[float]]],
) -> Dict[str, float]:
    """Score each client in a cohort from its per-class statistic vector.

    cohort: (client_key, per_class_vector) pairs. Vectors must all be the
    same length; entries may be None or NaN for classes the client holds no
    samples of.

    Returns {client_key: score in [0, 1]}, where HIGHER means more
    anomalous (larger normalised per-class values). Callers that want a
    trust-like quantity should use 1 - score.

    Implements FedNoRo Sec 4.1 steps 1-3:
      1. fill a missing class with the cohort minimum for that class. Their
         reasoning: a class the client holds no samples of contributes no
         noisy samples, and by the small-loss trick cleaner means smaller;
      2. min-max normalise each class across the cohort, so no single class
         dominates the vector;
      3. take the L2 norm. FedNoRo splits on a GMM whose noisy component is
         the "large norm mean vector", so the norm is the ordering their
         clustering acts on.

    Divided by sqrt(num_classes) at the end so the result lands in [0, 1]
    regardless of class count -- each normalised component is in [0, 1], so
    the raw norm is bounded by sqrt(C). That makes the score directly usable
    as a behaviour score with no hand-tuned rescale window, unlike the pooled
    variance which needed one.
    """
    rows = [(k, list(v)) for k, v in cohort if v is not None and len(v) > 0]
    if not rows:
        return {}
    num_classes = len(rows[0][1])
    if num_classes < 1 or any(len(v) != num_classes for _k, v in rows):
        return {}

    # Step 1: per-class cohort minimum, used to fill absent classes.
    present: List[List[float]] = [[] for _ in range(num_classes)]
    for _k, vec in rows:
        for c, v in enumerate(vec):
            if not _is_missing(v):
                present[c].append(float(v))

    filled: Dict[str, List[float]] = {}
    for k, vec in rows:
        out = []
        for c, v in enumerate(vec):
            if _is_missing(v):
                out.append(min(present[c]) if present[c] else 0.0)
            else:
                out.append(float(v))
        filled[k] = out

    # Step 2: min-max normalise each class across the cohort. A class where
    # every client reports the same value carries no discriminating
    # information this round, so it contributes 0 rather than an arbitrary
    # constant that would inflate every client's norm equally.
    lo = [min(filled[k][c] for k in filled) for c in range(num_classes)]
    hi = [max(filled[k][c] for k in filled) for c in range(num_classes)]

    # Step 3: L2 norm, scaled into [0, 1].
    denom = math.sqrt(num_classes)
    scores: Dict[str, float] = {}
    for k, row in filled.items():
        acc = 0.0
        for c in range(num_classes):
            span = hi[c] - lo[c]
            n = ((row[c] - lo[c]) / span) if span > 0 else 0.0
            acc += n * n
        s = math.sqrt(acc) / denom
        scores[k] = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    return scores


def decode_vector(raw) -> Optional[List[float]]:
    """Recover a per-class vector from a fit-metric value.

    Flower's proto accepts only bool/bytes/float/int/str, so the client ships
    list-valued metrics as JSON strings; a raw list raises at serde time AFTER
    fit() returns and silently drops the entire result. Accept either form so
    this works against live metrics and against already-decoded history.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            import json
            val = json.loads(raw)
        except (TypeError, ValueError):
            return None
    else:
        val = raw
    try:
        return [float(x) if x is not None else float("nan") for x in val]
    except (TypeError, ValueError):
        return None
