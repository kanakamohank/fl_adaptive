#!/usr/bin/env python3
"""Mechanism-visualisation plot: per-sample pretrain loss histograms for
a handful of noisy and clean clients at a specific round. Requires that
the run was executed with `--log-per-sample-losses-at-round R`.

What it answers: at round R, do noisy clients show visibly bimodal
per-sample loss (two humps = right-labeled low, wrong-labeled high) or
unimodal with a fatter tail? The answer depends on R:
  - early R (1-5): memorization has not kicked in; bimodal is plausible
  - mid R (15-25): memorization has partially masked the second hump
  - late R: fully memorized; shape may resemble clean

The script does NOT make shape claims; it just plots. Interpretation is
in the write-up.

Usage:
  python experiments/analysis/oracle/plot_mechanism_histograms.py \
      --pipeline-results \
        results/oracle_mech/.../full_verify_seed1/pipeline_results.json \
      --round 20 \
      --out plots/mechanism_r20.png
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from experiments.analysis.oracle import _signals as sig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-results", required=True,
                    help="Path to one seed's pipeline_results.json.")
    ap.add_argument("--round", type=int, required=True,
                    help="Round at which per-sample losses were dumped. "
                         "Must match the pilot's --log-per-sample-losses-at-round value.")
    ap.add_argument("--n-each", type=int, default=3,
                    help="How many representative clean/noisy clients to plot "
                         "(top row = clean, bottom row = noisy). Default 3.")
    ap.add_argument("--clip", type=float, default=8.0,
                    help="Cap per-sample loss at this value for readable histograms. "
                         "A sample with CE loss > 8 is strongly misclassified; "
                         "clipping keeps the x-axis legible without hiding the signal.")
    ap.add_argument("--out", default="plots/mechanism_histograms.png")
    args = ap.parse_args()

    run = sig.load_run(args.pipeline_results)
    osh = sig.oracle_history(run)
    row = osh.get(str(args.round)) or {}
    is_noisy = sig.noisy_predicate(run)

    clean = [(cid, e["per_sample_pretrain_loss"]) for cid, e in row.items()
             if not is_noisy(cid) and e.get("per_sample_pretrain_loss")]
    noisy = [(cid, e["per_sample_pretrain_loss"]) for cid, e in row.items()
             if is_noisy(cid) and e.get("per_sample_pretrain_loss")]
    if not clean or not noisy:
        raise SystemExit(
            f"round {args.round} has {len(clean)} clean / {len(noisy)} noisy clients "
            f"with per_sample_pretrain_loss. Did the pilot run with "
            f"--log-per-sample-losses-at-round {args.round}?"
        )

    n_show = min(args.n_each, len(clean), len(noisy))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7),
                              sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0.0, args.clip, 50)

    for i in range(n_show):
        cid, losses = clean[i]
        ax = axes[0, i]
        clipped = [min(v, args.clip) for v in losses]
        ax.hist(clipped, bins=bins, color="#2b6cb0", alpha=0.75, edgecolor="white")
        ax.set_title(f"CLEAN  cid={cid[:6]}  var={st.pvariance(losses):.2f}",
                      color="#2b6cb0")
        ax.axvline(2.303, color="gray", linestyle=":", alpha=0.5,
                    label="ln(10) random-guess" if i == 0 else None)
        if i == 0:
            ax.set_ylabel("count")
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        cid, losses = noisy[i]
        ax = axes[1, i]
        clipped = [min(v, args.clip) for v in losses]
        ax.hist(clipped, bins=bins, color="#c0392b", alpha=0.75, edgecolor="white")
        ax.set_title(f"NOISY  cid={cid[:6]}  var={st.pvariance(losses):.2f}",
                      color="#c0392b")
        ax.axvline(2.303, color="gray", linestyle=":", alpha=0.5)
        if i == 0:
            ax.set_ylabel("count")
        ax.set_xlabel("per-sample pretrain loss")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Per-sample pretrain loss histograms at round {args.round}\n"
                  f"source: {os.path.basename(os.path.dirname(args.pipeline_results))}",
                  fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"saved: {args.out}")

    # Also print pooled quantiles so the shape can be read without the image.
    all_clean = [v for _, lst in clean for v in lst]
    all_noisy = [v for _, lst in noisy for v in lst]
    def _q(xs, p):
        xs = sorted(xs); return xs[int(p * len(xs))]
    print(f"\n{'group':<6} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} "
          f"{'p90':>6} {'p95':>6} {'p99':>7}")
    for name, xs in [("clean", all_clean), ("noisy", all_noisy)]:
        print(f"{name:<6} " + " ".join(f"{_q(xs, p):>6.2f}" for p in
              (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) + f" {_q(xs, 0.99):>7.2f}")


if __name__ == "__main__":
    main()
