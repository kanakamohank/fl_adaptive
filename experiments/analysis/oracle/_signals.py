"""Shared utilities for oracle-signal analysis.

The oracle experiment logs per-(round, cid) diagnostic signals into
`pipeline_results.json` under `final_trust_state.oracle_signal_history`.
These helpers load a run, split clients into clean/noisy using the
ground-truth noisy_client_config_ids, and compute common per-signal
statistics (detrended per-client means, midpoint-threshold classification
accuracy, per-round accuracy sweep, rank-of-noisy, Mann-Whitney U).

The helpers are shared across the plot scripts in this directory so a
change in detrending logic lands in one place.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Scalar oracle signals the client/strategy log. Lists are decoded
# separately (per_class_pretrain_loss, per_sample_pretrain_loss) and are
# not included here.
SCALAR_SIGNALS: Tuple[str, ...] = (
    "pretrain_loss_mean", "pretrain_loss_var",
    "loss_epoch_first", "loss_epoch_last", "memorization_gap",
    "update_norm", "first_batch_grad_norm",
    "cosine_vs_weighted_mean", "cosine_vs_simple_mean",
    "bvd_behavior_score", "bvd_max_z",
    "small_loss_fraction",
)


def load_run(path: str) -> dict:
    """Load a single pipeline_results.json. Does not validate shape;
    callers should check for `final_trust_state.oracle_signal_history`."""
    with open(path) as f:
        return json.load(f)


def oracle_history(run: dict) -> Dict[str, Dict[str, dict]]:
    """Return oracle_signal_history as {round_str: {cid: entry}}.
    Empty dict if the run didn't log oracle signals.
    """
    fts = run.get("final_trust_state") or {}
    return fts.get("oracle_signal_history") or {}


def noisy_predicate(run: dict) -> Callable[[str], bool]:
    """Return a function `is_noisy(cid)` built from the ground-truth
    cid_to_client_config_id map and the recorded noisy_client_config_ids
    list. Clients not in the map (e.g. late joiners) are treated as not
    noisy, which matches the pipeline's own labelling convention."""
    cid_to_cfg = run.get("cid_to_client_config_id") or {}
    noisy_cfgs = set(run.get("noisy_client_config_ids") or [])
    def _is_noisy(cid: str) -> bool:
        return cid_to_cfg.get(cid, cid) in noisy_cfgs
    return _is_noisy


def _finite_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def per_cid_detrended_mean(osh: Dict[str, Dict[str, dict]],
                            signal: str) -> Dict[str, float]:
    """Return {cid: mean(signal_r - round_mean_r) over the rounds this
    cid appeared in}. Reviewer-required: raw per-client means are
    dominated by the across-round trajectory, not by per-client
    variation; detrending by round mean strips that out.

    Rounds where the cid produced no finite value are skipped. Returns
    an empty dict if the signal is absent entirely.
    """
    by_cid: Dict[str, List[float]] = {}
    rounds = sorted(int(r) for r in osh.keys())
    for r in rounds:
        row = []
        for cid, e in osh[str(r)].items():
            fv = _finite_float(e.get(signal))
            if fv is None:
                continue
            row.append((cid, fv))
        if not row:
            continue
        rm = sum(v for _, v in row) / len(row)
        for cid, fv in row:
            by_cid.setdefault(cid, []).append(fv - rm)
    return {cid: sum(s) / len(s) for cid, s in by_cid.items() if s}


def midpoint_threshold_errors(pcm: Dict[str, float],
                               is_noisy: Callable[[str], bool]) -> Optional[dict]:
    """Simple binary classifier: threshold at the midpoint of the
    clean-group median and the noisy-group median. Predict noisy on the
    side of the threshold the noisy-group median sits on. Returns
    counts + accuracy, or None when either group is empty.
    """
    clean = sorted([v for c, v in pcm.items() if not is_noisy(c)])
    noisy = sorted([v for c, v in pcm.items() if is_noisy(c)])
    if not clean or not noisy:
        return None
    c_med = clean[len(clean) // 2]
    n_med = noisy[len(noisy) // 2]
    thr = (c_med + n_med) / 2
    noisy_above = n_med > c_med
    if noisy_above:
        noisy_err = sum(1 for v in noisy if v < thr)
        clean_err = sum(1 for v in clean if v >= thr)
    else:
        noisy_err = sum(1 for v in noisy if v > thr)
        clean_err = sum(1 for v in clean if v <= thr)
    total = len(clean) + len(noisy)
    return {
        "clean_n": len(clean), "noisy_n": len(noisy),
        "clean_errors": clean_err, "noisy_errors": noisy_err,
        "total_errors": clean_err + noisy_err,
        "accuracy": 1.0 - (clean_err + noisy_err) / total,
        "clean_median": c_med, "noisy_median": n_med,
        "threshold": thr,
    }


def per_round_accuracy(osh: Dict[str, Dict[str, dict]], signal: str,
                        is_noisy: Callable[[str], bool]) -> Dict[int, Optional[float]]:
    """For each round, use ONLY values from that round (no averaging)
    with the midpoint-threshold classifier. Rounds with too few
    observations in either group are skipped (None).
    """
    out: Dict[int, Optional[float]] = {}
    for r_str, row in osh.items():
        r = int(r_str)
        clean, noisy = [], []
        for cid, e in row.items():
            fv = _finite_float(e.get(signal))
            if fv is None:
                continue
            (noisy if is_noisy(cid) else clean).append(fv)
        if len(clean) < 2 or len(noisy) < 2:
            out[r] = None
            continue
        c_med = sorted(clean)[len(clean) // 2]
        n_med = sorted(noisy)[len(noisy) // 2]
        thr = (c_med + n_med) / 2
        if n_med > c_med:
            errs = sum(1 for v in noisy if v < thr) + sum(1 for v in clean if v >= thr)
        else:
            errs = sum(1 for v in noisy if v > thr) + sum(1 for v in clean if v <= thr)
        out[r] = 1.0 - errs / (len(clean) + len(noisy))
    return out


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """Mann-Whitney U two-tailed, normal approximation. Handles ties by
    mid-rank. Returns (U_min, z, p). Dependency-free so analysis scripts
    don't need scipy."""
    combined = list(x) + list(y)
    pairs = sorted((v, i) for i, v in enumerate(combined))
    rank_of: Dict[int, float] = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank_of[pairs[k][1]] = avg
        i = j + 1
    R1 = sum(rank_of[i] for i in range(len(x)))
    n1, n2 = len(x), len(y)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    mu = n1 * n2 / 2
    sigma = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    z = (U - mu) / sigma if sigma > 0 else 0.0
    # Two-tailed p via complementary error function.
    p = math.erfc(abs(z) / 2 ** 0.5)
    return U, z, p


def discover_seed_files(results_dir: str, arm: str = "full_verify") -> List[str]:
    """Discover `<arm>_seed*/pipeline_results.json` under `results_dir`.

    Tries three layouts so the caller can point at any convenient level:
      1. `<results_dir>/<arm>_seed*/pipeline_results.json` (results_dir IS
         a skipXX directory)
      2. `<results_dir>/skip*/<arm>_seed*/pipeline_results.json`
         (results_dir is the config-tag parent above skip*)
      3. `<results_dir>/**/<arm>_seed*/pipeline_results.json`
         (any deeper nesting; last-resort recursive)

    Returns a sorted list of absolute paths. Empty list if nothing
    matched; caller decides how to error.
    """
    for pat in (
        os.path.join(results_dir, f"{arm}_seed*", "pipeline_results.json"),
        os.path.join(results_dir, "skip*", f"{arm}_seed*", "pipeline_results.json"),
        os.path.join(results_dir, "**", f"{arm}_seed*", "pipeline_results.json"),
    ):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits
    return []


def seed_number(path: str) -> int:
    """Extract the integer N from `..._seed<N>/pipeline_results.json`.
    Returns -1 if the path does not match that convention."""
    base = os.path.basename(os.path.dirname(path))
    if "_seed" in base:
        try:
            return int(base.rsplit("_seed", 1)[1])
        except ValueError:
            return -1
    return -1
