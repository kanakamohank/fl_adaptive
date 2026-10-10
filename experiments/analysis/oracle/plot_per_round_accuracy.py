#!/usr/bin/env python3
"""Per-round signal strength: how early in training does each oracle
signal separate clean from noisy?

For each round r, use ONLY values from round r (no averaging) with the
midpoint-threshold classifier. Plot mean accuracy across seeds + 1σ
band for a configurable set of signals.

Usage:
  python experiments/analysis/oracle/plot_per_round_accuracy.py \
      --results-dir results/oracle/r40/<cfg_tag>/skip00 \
      --arm full_verify \
      --out plots/per_round_accuracy.png
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from experiments.analysis.oracle import _signals as sig  # noqa: E402


DEFAULT_SIGNALS = ("pretrain_loss_var", "pretrain_loss_mean",
                    "memorization_gap", "bvd_behavior_score",
                    "small_loss_fraction")
DEFAULT_COLORS = {
    "pretrain_loss_var":        "#1f77b4",
    "pretrain_loss_mean":       "#2ca02c",
    "memorization_gap":         "#ff7f0e",
    "bvd_behavior_score":       "#d62728",
    "small_loss_fraction":      "#9467bd",
    "cosine_vs_weighted_mean":  "#8c564b",
    "update_norm":              "#17becf",
    "first_batch_grad_norm":    "#e377c2",
    "loss_epoch_last":          "#7f7f7f",
    "loss_epoch_first":         "#bcbd22",
    "bvd_max_z":                "#17202a",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arm", default="full_verify",
                    choices=("full_verify", "tavs_skip", "random_skip"))
    ap.add_argument("--signals", nargs="*", default=list(DEFAULT_SIGNALS),
                    help="Which signals to plot. Default: top-5.")
    ap.add_argument("--chance-baseline", type=float, default=0.60,
                    help="Horizontal line for the always-clean baseline. "
                         "0.60 = 20 noisy of 50; 0.90 = 5 of 50; etc.")
    ap.add_argument("--out", default="plots/per_round_accuracy.png")
    ap.add_argument("--title", default=None,
                    help="Figure title. Default derived from results-dir.")
    args = ap.parse_args()

    paths = sig.discover_seed_files(args.results_dir, args.arm)
    if not paths:
        raise SystemExit(f"no {args.arm}_seed*/pipeline_results.json under {args.results_dir}")

    # Compute per-round-per-signal across all seeds.
    accs = {s: {} for s in args.signals}  # signal -> {round: [acc per seed]}
    rounds_seen = set()
    for p in paths:
        run = sig.load_run(p)
        osh = sig.oracle_history(run)
        if not osh:
            continue
        is_noisy = sig.noisy_predicate(run)
        for s in args.signals:
            row = sig.per_round_accuracy(osh, s, is_noisy)
            for r, acc in row.items():
                if acc is None:
                    continue
                accs[s].setdefault(r, []).append(acc)
                rounds_seen.add(r)

    rounds = sorted(rounds_seen)
    if not rounds:
        raise SystemExit("no per-round accuracies produced; check signals and arm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in args.signals:
        means, stds = [], []
        xs = []
        for r in rounds:
            vals = accs[s].get(r, [])
            if vals:
                means.append(st.mean(vals))
                stds.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
                xs.append(r)
        if not xs:
            continue
        color = DEFAULT_COLORS.get(s, None)
        ax.plot(xs, means, label=s, color=color, linewidth=2)
        ax.fill_between(xs,
                         [m - sd for m, sd in zip(means, stds)],
                         [m + sd for m, sd in zip(means, stds)],
                         color=color, alpha=0.15)
    ax.axhline(args.chance_baseline, color="gray", linestyle=":", linewidth=1,
                label=f"chance baseline ({args.chance_baseline:.0%})")
    ax.set_xlabel("Federation round")
    ax.set_ylabel("Per-round noisy-vs-clean classification accuracy")
    ax.set_title(args.title or f"Signal strength per round — {os.path.basename(args.results_dir.rstrip('/'))} ({args.arm})")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
