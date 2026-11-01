#!/usr/bin/env python3
"""How much does the trust signal actually SPREAD clients apart, per partition?

Why this exists
---------------
Trust weighting helps under IID (+0.0125 to +0.0166, t=11-14, 5/5 seeds) and
hurts under every Dirichlet partition tested. Two mechanisms were proposed and
one of them is falsified here.

Hypothesis A (falsified): weighting over-concentrates the aggregate on a few
clients, and under a skewed partition that starves classes. FedLAW (ICLR 2026,
Sec. 1) argues exactly this shape of failure and guards against it by capping
every client's weight at 1/(n - b_f) on a unit-capped simplex; we have a 0.05
FLOOR but no cap, so the concern applied on its face. It does not survive
measurement: concentration is HIGHER under IID (2.4x the unweighted top share)
than under alpha=1.0 (1.1x), i.e. highest exactly where weighting helps.

Hypothesis B (supported): what changes with the partition is how far apart the
signal pushes clients, and the DIRECTION of that effect flips. Under IID a
large, correct spread helps. Under a skewed partition, the larger the correct
spread, the more accuracy is lost.

Two signals are reconstructed exactly as the strategy computed them, so these
are the weights the aggregate actually saw:

  fixed-window   behaviour = 1 - clip((pretrain_loss_var - lo)/(hi - lo), 0, 1)
                 with the run's own --pretrain-loss-var-rescale lo,hi
  per-class      behaviour = 1 - per_class_cohort_scores(...), cohort-relative
                 min-max, so its spread is near-maximal by construction

Reported per run: sd of the behaviour score within a round, the mean
clean-minus-noisy score gap (positive = the signal is pointed the right way),
and what share of clients land pinned at 0 or at 1 (a pinned client is one the
window could not rank).

Usage:
  python3 experiments/analysis/oracle/signal_dispersion.py            # known set
  python3 experiments/analysis/oracle/signal_dispersion.py results/foo ...
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.tavs_v2.per_class_score import per_class_cohort_scores

DEFAULT_LO, DEFAULT_HI = 0.5, 3.0


def window_from_config(cfg, path):
    """The rescale window this run actually used.

    Read from the run's own config when present, else recovered from the
    config-tag directory name (pv-rescale-06-15 -> 0.6, 1.5), else the
    strategy default. Hard-coding one window would silently mis-score any run
    launched with a different --pretrain-loss-var-rescale.
    """
    lo = cfg.get("pretrain_loss_var_rescale_lo")
    hi = cfg.get("pretrain_loss_var_rescale_hi")
    if lo is not None and hi is not None and float(hi) > float(lo):
        return float(lo), float(hi)
    m = re.search(r"pv-rescale-(\d+)-(\d+)", str(path))
    if m:
        # tag digits are the value x10 with the point dropped: 06 -> 0.6
        lo, hi = int(m.group(1)) / 10.0, int(m.group(2)) / 10.0
        if hi > lo:
            return lo, hi
    return DEFAULT_LO, DEFAULT_HI


def _summarise(spread, gap, pinned_lo, pinned_hi, total):
    if not spread:
        return None
    return {
        "rounds": len(spread),
        "sd": statistics.mean(spread),
        "gap": statistics.mean(gap) if gap else float("nan"),
        "pin_lo": 100.0 * pinned_lo / total if total else 0.0,
        "pin_hi": 100.0 * pinned_hi / total if total else 0.0,
    }


def analyse(root: Path):
    """Returns (fixed_window_summary, per_class_summary, concentration)."""
    w_spread, w_gap, w_lo, w_hi, w_tot = [], [], 0, 0, 0
    p_spread, p_gap = [], []
    conc_w, conc_u = [], []

    for p in sorted(glob.glob(str(root / "**" / "tavs_skip_seed*" / "pipeline_results.json"),
                              recursive=True)):
        d = json.loads(Path(p).read_text())
        fts = d["final_trust_state"]
        osh = fts.get("oracle_signal_history") or {}
        cmap = fts.get("cid_to_client_config_id", {})
        noisy = set(d.get("noisy_client_config_ids", []))
        lo, hi = window_from_config(d.get("config") or {}, p)

        for _r, entries in osh.items():
            ver = [(cid, e) for cid, e in entries.items() if e.get("is_verified")]
            if len(ver) < 3:
                continue

            # --- fixed-window signal, and the concentration it produces
            scores, weights, counts = {}, [], []
            for cid, e in ver:
                raw = e.get("pretrain_loss_var")
                n = e.get("num_examples") or 0
                if raw is None or n <= 0:
                    continue
                s = 1.0 - max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))
                scores[cid] = s
                w_tot += 1
                if s <= 0.0:
                    w_lo += 1
                if s >= 1.0:
                    w_hi += 1
                # the 0.05 floor is what aggregate_fit applies to an inlier
                weights.append(max(0.05, s) * n)
                counts.append(float(n))
            if len(scores) >= 3:
                w_spread.append(statistics.stdev(scores.values()))
                cl = [v for k, v in scores.items() if cmap.get(k, k) not in noisy]
                ns = [v for k, v in scores.items() if cmap.get(k, k) in noisy]
                if cl and ns:
                    w_gap.append(statistics.mean(cl) - statistics.mean(ns))
            if len(weights) >= 3 and sum(weights) > 0 and sum(counts) > 0:
                conc_w.append(max(weights) / sum(weights))
                conc_u.append(max(counts) / sum(counts))

            # --- per-class cohort-relative signal
            cohort = [(cid, e["per_class_pretrain_var"]) for cid, e in ver
                      if e.get("per_class_pretrain_var")]
            if len(cohort) >= 3:
                b = {k: 1.0 - v for k, v in per_class_cohort_scores(cohort).items()}
                if len(b) >= 3:
                    p_spread.append(statistics.stdev(b.values()))
                    cl = [v for k, v in b.items() if cmap.get(k, k) not in noisy]
                    ns = [v for k, v in b.items() if cmap.get(k, k) in noisy]
                    if cl and ns:
                        p_gap.append(statistics.mean(cl) - statistics.mean(ns))

    conc = None
    if conc_w:
        mw, mu = statistics.mean(conc_w), statistics.mean(conc_u)
        conc = {"weighted": mw, "unweighted": mu,
                "amplification": (mw / mu) if mu > 0 else float("nan")}
    return (_summarise(w_spread, w_gap, w_lo, w_hi, w_tot),
            _summarise(p_spread, p_gap, 0, 0, 0),
            conc)


DEFAULT_RUNS = [
    ("results/nm_uniform_iid_twagg", "UNIFORM noise, IID"),
    ("results/nm_uniform_a10_twagg", "UNIFORM noise, a=1.0"),
    ("results/tavs_integration_v2_iid_worst_twagg", "C10N worst, IID"),
    ("results/sweep_a30_twagg", "C10N worst, a=3.0"),
    ("results/sweep_a10_twagg", "C10N worst, a=1.0"),
    ("results/tavs_integration_v3_noniid_perclass", "C10N worst, a=0.3 (window signal)"),
    ("results/tavs_integration_v5_noniid_perclass_trust", "C10N worst, a=0.3 (per-class signal)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", type=Path)
    args = ap.parse_args()
    runs = ([(r, r.name) for r in args.roots] if args.roots
            else [(Path(a), b) for a, b in DEFAULT_RUNS])

    print("Behaviour-score dispersion, reconstructed as the strategy computed it.")
    print("gap > 0 means clean clients correctly score above noisy ones.")
    print("'pinned' = share of clients the fixed window could not rank (score 0 or 1).\n")
    hdr = (f"{'run':<38} {'rnds':>5} {'sd':>7} {'gap':>8} "
           f"{'pin@0':>7} {'pin@1':>7} {'pc sd':>7} {'pc gap':>8} {'conc':>7}")
    print(hdr)
    print("-" * len(hdr))
    for root, label in runs:
        if not root.exists():
            print(f"{label:<38} (missing)")
            continue
        win, pc, conc = analyse(root)
        if not win:
            print(f"{label:<38} (no oracle history)")
            continue
        pcs = f"{pc['sd']:7.3f} {pc['gap']:+8.3f}" if pc else f"{'--':>7} {'--':>8}"
        cs = f"{conc['amplification']:6.2f}x" if conc else f"{'--':>7}"
        print(f"{label:<38} {win['rounds']:>5} {win['sd']:7.3f} {win['gap']:+8.3f} "
              f"{win['pin_lo']:6.1f}% {win['pin_hi']:6.1f}% {pcs} {cs}")

    print("\nconc = top client's share of the aggregate under weighting, as a multiple")
    print("of its share without weighting. >1 means weighting concentrated the update.")


if __name__ == "__main__":
    main()
