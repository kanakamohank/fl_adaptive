#!/usr/bin/env python3
"""
Pilot: trust-based skip vs uniform-random skip, all honest clients.

Answers ONE question before we spend on a full grid:
  With verification cost fixed (same skip rate across policies), does the trust
  signal decide who to skip better than a coin flip?

Three arms at matched skip rate:
  - full_verify: no skips (control, upper bound on cost, upper bound on accuracy)
  - tavs_skip: trust-EMA promotes; skip rate emerges from the trust dynamics
  - random_skip: same expected skip rate as tavs_skip's target, uniform-random selection

Success signal: tavs_skip's accuracy sits at or near full_verify's; random_skip
trails. Failure signal: tavs_skip and random_skip overlap -- trust is not doing
work a coin flip could not.

This is the FAIR baseline the existing tavs_vs_full_seeded harness lacked: the
gap between TAVS and Full mixes "skipping helped" with "trust-based skipping
helped", and only random_skip separates them.

Small by design: 100 clients total, 20 per round, 20 rounds, 2 seeds, honest
only. Runs end-to-end in about an hour on CPU. If the signal is there we scale;
if it is not we redesign, not spend more compute.

Usage:
    python -m experiments.pilot_skip_comparison
    python -m experiments.pilot_skip_comparison --seeds 1,2 --rounds 20 --skip-rate 0.4
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tavs_v2 import PipelineConfig, TavsEspConfig, TAVSESPPipeline
from src.tavs_v2.tavs_esp_strategy import FullVerificationStrategy, RandomSkipStrategy

logger = logging.getLogger(__name__)


def _config_tag(args) -> str:
    """Directory component encoding the DATA-side config, so runs at different
    split / noise settings do not collide.

    Encodes split mode, non-Dirichlet-alpha (only when the split uses it),
    and label-noise knobs. num_classes is folded into the tag only when it
    departs from the CIFAR-10 default of 10 so future CIFAR-100 runs do not
    silently share a directory with CIFAR-10 runs at the same noise config.
    """
    split = getattr(args, "data_split", "dirichlet")
    parts = [f"split-{split}"]
    if split == "dirichlet":
        parts.append(f"a{args.data_alpha:g}")
    if args.noisy_client_fraction > 0 and args.label_noise_rate > 0:
        nf = int(round(args.noisy_client_fraction * 100))
        nr = int(round(args.label_noise_rate * 100))
        parts.append(f"noiseC{nf:02d}_R{nr:02d}")
        # Only tag the class count when it deviates from the default so
        # existing CIFAR-10 runs keep their paths.
        if getattr(args, "label_noise_num_classes", 10) != 10:
            parts.append(f"K{args.label_noise_num_classes}")
    else:
        parts.append("clean")
    return "_".join(parts)


def build_arm(name: str, args, seed: int, skip_rate: float):
    """Return (strategy_class, extra_kwargs) for the requested arm.

    Kept out of a module-level ARMS constant because RandomSkipStrategy takes
    two per-run kwargs (skip_rate, skip_seed) that a frozen dict would freeze
    away. skip_seed is bound to the run's seed here specifically: without it
    RandomSkipStrategy.__init__ defaults skip_seed=0 and every seed of the
    random arm draws the identical Bernoulli sequence, silently killing half
    the intended seed variance.
    """
    if name == "full_verify":
        return FullVerificationStrategy, {}
    if name == "tavs_skip":
        return None, {}   # None -> pipeline uses TavsEspStrategy directly
    if name == "random_skip":
        return RandomSkipStrategy, {"skip_rate": skip_rate, "skip_seed": seed}
    raise ValueError(f"unknown arm: {name}")


ARM_ORDER = ["full_verify", "tavs_skip", "random_skip"]
ARM_COLOURS = {"full_verify": "#d95f02",
               "tavs_skip":   "#1b9e77",
               "random_skip": "#7570b3"}
ARM_NAMES = {"full_verify": "Full verify",
             "tavs_skip":   "TAVS skip",
             "random_skip": "Random skip"}


def run_one(arm: str, seed: int, args, skip_rate: float):
    """One end-to-end pipeline run; returns the fields we plot and score on."""
    strategy_class, strategy_kwargs = build_arm(arm, args, seed, skip_rate)

    tavs_config = TavsEspConfig(
        # Kept identical to tavs_vs_full_seeded.py so results here are
        # comparable to those runs and any regression is attributable to
        # RandomSkip, not to a config drift.
        theta_low=0.3, theta_high=0.7, alpha_trust=0.9, gamma_budget=0.35,
        tau_ramp=5.0, k_trust=3, target_k=150,
        detection_threshold=5.0,
        clip_promoted_updates=True, promoted_clip_factor=2.0,
        cosine_filter_promoted=False,   # off by default here, matching that harness
        enable_outlier_detection=True,
    )

    # PipelineConfig takes a strategy CLASS, not an instance, and the pipeline
    # calls strategy_class(config=..., model_structure=...). RandomSkipStrategy
    # takes an extra skip_rate arg the pipeline doesn't know how to pass. Bind
    # it in a lightweight subclass rather than functools.partial, because the
    # pipeline reads strategy_class.__name__ for its log line -- partials have
    # no __name__ and that path crashes at run start.
    if strategy_kwargs:
        base_cls = strategy_class
        bound_kwargs = dict(strategy_kwargs)

        class _BoundStrategy(base_cls):
            def __init__(self, config, model_structure=None):
                super().__init__(config, model_structure=model_structure,
                                 **bound_kwargs)

        _BoundStrategy.__name__ = f"{base_cls.__name__}_r{int(skip_rate * 100)}"
        strategy_class = _BoundStrategy

    config = PipelineConfig(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        byzantine_fraction=0.0,   # honest only
        tavs_config=tavs_config,
        strategy_class=strategy_class,
        data_split=args.data_split,
        data_alpha=args.data_alpha,
        # Label noise on a subset of clients: the only source of client-level
        # DIFFERENTIATION in this otherwise-honest pilot. Without it TAVS's
        # trust EMA has no signal to converge on (see the r20/skip49 result).
        label_noise_client_fraction=args.noisy_client_fraction,
        label_noise_rate=args.label_noise_rate,
        seed=seed,
        # Encode the label-noise config in the path so IID+noise runs do not
        # overwrite the earlier no-noise r20/skip49 results the analysis
        # already cites.
        output_dir=str(Path(args.results_dir) / f"r{args.rounds}" /
                       _config_tag(args) /
                       f"skip{int(skip_rate * 100):02d}" /
                       f"{arm}_seed{seed}"),
    )

    print(f"\n{'=' * 70}\n{arm}  seed={seed}  "
          f"(rounds={args.rounds}, skip_target={skip_rate:.2f})\n{'=' * 70}")
    started = time.time()
    results = TAVSESPPipeline(config).run_simulation()

    sched = results.scheduling_history
    total_verified = sum(s["num_verified"] for s in sched)
    total_promoted = sum(s["num_promoted"] for s in sched)
    total_cohort = total_verified + total_promoted
    observed_skip = (total_promoted / total_cohort) if total_cohort else 0.0

    return {
        "arm": arm,
        "seed": seed,
        "final_accuracy": results.server_accuracies[-1],
        # Match the pre-specified late-window used in tavs_vs_full_seeded.
        "late_window": max(1, int(round(args.rounds * 0.25))),
        "late_accuracy": statistics.mean(
            results.server_accuracies[-max(1, int(round(args.rounds * 0.25))):]
        ),
        "accuracy_trajectory": results.server_accuracies,
        "total_verified": total_verified,
        "total_promoted": total_promoted,
        "observed_skip_rate": observed_skip,
        "elapsed_seconds": time.time() - started,
    }


def make_plot(rows, args, out_path, matched_rate):
    """Two panels: accuracy trajectory (headline) and observed skip rate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    noise_tag = ("clean" if args.noisy_client_fraction * args.label_noise_rate == 0
                 else f"{int(args.noisy_client_fraction*100)}% clients @ "
                      f"{int(args.label_noise_rate*100)}% label noise")
    fig.suptitle(
        f"Pilot: trust-based vs random skip at matched rate\n"
        f"{len(args.seed_list)} seeds, {args.rounds} rounds, "
        f"CIFAR-10 alpha={args.data_alpha}, {noise_tag}, "
        f"matched skip={matched_rate:.2f}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: accuracy per round.
    ax = axes[0]
    for arm in ARM_ORDER:
        traj = np.array([r["accuracy_trajectory"] for r in rows if r["arm"] == arm])
        if traj.size == 0:
            continue
        x = np.arange(traj.shape[1])
        ax.plot(x, traj.mean(0), color=ARM_COLOURS[arm], lw=2, label=ARM_NAMES[arm])
        ax.fill_between(x, traj.min(0), traj.max(0),
                        color=ARM_COLOURS[arm], alpha=0.15)
    ax.axhline(0.1, ls=":", c="grey", lw=1)
    ax.text(0.5, 0.105, "random guess", fontsize=8, color="grey")
    ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy (shaded = min-max across seeds)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # Panel 2: verifications vs promotions per arm, seed-level dots.
    ax = axes[1]
    for i, arm in enumerate(ARM_ORDER):
        v = [r["total_verified"] for r in rows if r["arm"] == arm]
        p = [r["total_promoted"] for r in rows if r["arm"] == arm]
        ax.bar(i - 0.18, np.mean(v), 0.35, color=ARM_COLOURS[arm],
               edgecolor="black", label="verified" if i == 0 else None)
        ax.bar(i + 0.18, np.mean(p), 0.35, color=ARM_COLOURS[arm],
               edgecolor="black", hatch="///",
               label="promoted (skipped)" if i == 0 else None)
        ax.scatter([i - 0.18] * len(v), v, color="black", zorder=3, s=15)
        ax.scatter([i + 0.18] * len(p), p, color="black", zorder=3, s=15)
    ax.set_xticks(range(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_NAMES[a] for a in ARM_ORDER])
    ax.set_ylabel(f"Client-rounds over {args.rounds} rounds")
    ax.set_title("Where the compute went")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def summarise(rows, arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm]
    return {
        "mean": statistics.mean(vals) if vals else float("nan"),
        "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def paired_delta(rows, seeds, key, a, b):
    """Paired arm_a minus arm_b, matched by seed. Small-n paired means:
    with two seeds the CI is uninformative but the sign is real, so we report
    the per-seed values and the mean, no test statistic."""
    xa = [next(r[key] for r in rows if r["arm"] == a and r["seed"] == s) for s in seeds]
    xb = [next(r[key] for r in rows if r["arm"] == b and r["seed"] == s) for s in seeds]
    diffs = [x - y for x, y in zip(xa, xb)]
    return {
        "per_seed": {str(s): d for s, d in zip(seeds, diffs)},
        "mean": statistics.mean(diffs),
        "n_negative": sum(1 for d in diffs if d < 0),
        "n_positive": sum(1 for d in diffs if d > 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1,2,3",
                        help="Comma-separated seed list. Default 1,2,3: two seeds "
                             "give a 50%% coin flip on paired-delta sign agreement, "
                             "so 2 is the pilot floor -- 3 is the cheap upgrade that "
                             "turns 1/2 same-sign into 2/3 or 3/3.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=100,
                        help="Client pool size (default 100).")
    parser.add_argument("--clients-per-round", type=int, default=20,
                        help="Cohort size per round (default 20 -- 20%% participation).")
    parser.add_argument("--skip-rate", type=float, default=None,
                        help="Target skip rate for the random_skip arm. If unset "
                             "(the recommended path), matched to TAVS's OBSERVED mean "
                             "skip rate across seeds -- a two-pass run: TAVS first, "
                             "read its observed rate, RandomSkip second at that rate. "
                             "The one-pass fixed-rate variant is available for "
                             "sensitivity checks and is not the matched comparison.")
    parser.add_argument("--data-split", default="iid", choices=("iid", "dirichlet"),
                        help="Client split. 'iid' (default) shards CIFAR-10 evenly "
                             "so every client gets len(dataset)/num_clients samples "
                             "with no class skew -- required for the label-noise "
                             "pilot, because label noise is the only client-level "
                             "differentiation signal and Dirichlet at num_clients=100 "
                             "leaves clients with ~50 samples each (some down to a "
                             "1-sample fallback where 20%% noise rounds to zero). "
                             "'dirichlet' uses --data-alpha and is available for "
                             "sensitivity checks.")
    parser.add_argument("--data-alpha", type=float, default=0.3,
                        help="Dirichlet alpha, only used when --data-split=dirichlet. "
                             "Default 0.3 for continuity with tavs_vs_full_seeded.py.")
    parser.add_argument("--noisy-client-fraction", type=float, default=0.2,
                        help="Fraction of the client pool that gets noisy labels "
                             "(default 0.2 -> 20 of 100 clients).")
    parser.add_argument("--label-noise-rate", type=float, default=0.2,
                        help="Fraction of a noisy client's labels flipped to a "
                             "wrong class (default 0.2).")
    parser.add_argument("--results-dir", default="results/pilot_skip_comparison")
    args = parser.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Two-pass to keep the comparison honest: run full_verify + tavs_skip first,
    # measure TAVS's actual observed skip rate, then run random_skip at that rate.
    # The one-pass fixed-rate mode is preserved via --skip-rate for sensitivity
    # checks but is not a matched-rate comparison.
    rows = []
    for seed in args.seed_list:
        for arm in ("full_verify", "tavs_skip"):
            rows.append(run_one(arm, seed, args, skip_rate=0.0))

    tavs_rates = [r["observed_skip_rate"] for r in rows if r["arm"] == "tavs_skip"]
    matched_rate = (args.skip_rate if args.skip_rate is not None
                    else float(np.mean(tavs_rates)))
    print(f"\n[matched-rate] TAVS observed skip = {tavs_rates} "
          f"-> random_skip target = {matched_rate:.3f}")

    for seed in args.seed_list:
        rows.append(run_one("random_skip", seed, args, skip_rate=matched_rate))
    # Path uses the matched rate so different runs do not collide.
    out_dir = (Path(args.results_dir) / f"r{args.rounds}" / _config_tag(args)
               / f"skip{int(matched_rate*100):02d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_results.json").write_text(json.dumps(
        {"config": {k: v for k, v in vars(args).items() if k != "seed_list"},
         "seeds": args.seed_list, "matched_skip_rate": matched_rate,
         "runs": rows}, indent=2))

    plot_path = out_dir / "pilot_skip_comparison.png"
    make_plot(rows, args, plot_path, matched_rate)

    # Compact console report.
    print(f"\n{'=' * 80}")
    print(f"PILOT: trust-based skip vs random skip, no attack, "
          f"{len(args.seed_list)} seeds x {args.rounds} rounds")
    print(f"{'=' * 80}")
    print(f"{'arm':>14} {'late acc (mean+-std)':>26} "
          f"{'verified':>10} {'promoted':>10} {'obs skip':>10}")
    for a in ARM_ORDER:
        la = summarise(rows, a, "late_accuracy")
        v  = summarise(rows, a, "total_verified")
        p  = summarise(rows, a, "total_promoted")
        os_ = summarise(rows, a, "observed_skip_rate")
        print(f"{a:>14} {la['mean']:>17.3f} +-{la['std']:.3f} "
              f"{v['mean']:>10.0f} {p['mean']:>10.0f} {os_['mean']:>10.2%}")

    # Paired deltas -- the actual test the pilot is trying to run.
    for a, b in (("tavs_skip", "full_verify"),
                 ("random_skip", "full_verify"),
                 ("tavs_skip", "random_skip")):
        d = paired_delta(rows, args.seed_list, "late_accuracy", a, b)
        signs = f"{max(d['n_negative'], d['n_positive'])}/{len(args.seed_list)} same-sign"
        print(f"\n  {a} - {b}  late-acc delta")
        print("    per seed: " + "  ".join(
            f"s{k}:{v:+.3f}" for k, v in d["per_seed"].items()))
        print(f"    mean {d['mean']:+.4f}   ({signs})")

    print(f"\nPlot: {plot_path}")
    print(f"JSON: {out_dir / 'pilot_results.json'}")


if __name__ == "__main__":
    main()
