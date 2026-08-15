#!/usr/bin/env python3
"""
TAVS vs full verification across seeds, with no adversary.

Answers one question: what does trust-adaptive scheduling cost when there is
nothing to defend against? With no attacker, TAVS's only effect is to skip
verifying clients it trusts, so any accuracy difference is the price of that
choice and any verification difference is the saving.

Why seeds are required here specifically
----------------------------------------
A single-run difference cannot support this claim. The clip_factor sweep
accidentally measured the noise floor: five arms that turned out to be
functionally identical (clipping never fired) produced final accuracies of
0.342, 0.239, 0.291, 0.404, 0.280 -- mean 0.311, std 0.064, range 0.165. The
TAVS-vs-full gap under discussion is 0.052, comfortably inside that. Without
repeated seeds the sign of the difference is not even established.

The verification counts do not need this treatment: they are near-deterministic
given the schedule, which is why the saving is the more defensible half of the
claim.

Seeds control model init, DataLoader shuffling, local SGD order and the
Dirichlet split. They do NOT change the ESP projection matrices, which derive
from SHA-256(master_key || round || block) and are identical across runs.

Usage:
    python -m experiments.tavs_vs_full_seeded
    python -m experiments.tavs_vs_full_seeded --seeds 1,2,3 --rounds 20
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
from src.tavs_v2.tavs_esp_strategy import FullVerificationStrategy

logger = logging.getLogger(__name__)

ARMS = {"tavs": None, "full_verification": FullVerificationStrategy}


def run_one(arm, strategy_class, seed, args):
    """One pipeline. Identical config across arms except the strategy."""
    tavs_config = TavsEspConfig(
        theta_low=0.3, theta_high=0.7, alpha_trust=0.9, gamma_budget=0.35,
        tau_ramp=5.0, k_trust=3, target_k=args.target_k, detection_threshold=2.0,
        clip_promoted_updates=True, promoted_clip_factor=args.clip_factor,
    )
    config = PipelineConfig(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        byzantine_fraction=0.0,          # no adversary: this is the whole point
        tavs_config=tavs_config,
        strategy_class=strategy_class,
        data_alpha=args.data_alpha,
        seed=seed,
        output_dir=str(Path(args.results_dir) / f"{arm}_seed{seed}"),
    )

    print(f"\n{'=' * 70}\n{arm}  seed={seed}  ({args.rounds} rounds, no attack)\n{'=' * 70}")
    started = time.time()
    results = TAVSESPPipeline(config).run_simulation()

    sched = results.scheduling_history
    return {
        "arm": arm,
        "seed": seed,
        "final_accuracy": results.server_accuracies[-1],
        # Mean of the last five rounds. Far less end-point noise than the final
        # round alone, which in the sweep swung 0.24-0.40 across identical configs.
        "late_accuracy": statistics.mean(results.server_accuracies[-5:]),
        "accuracy_trajectory": results.server_accuracies,
        "total_verified": sum(s["num_verified"] for s in sched),
        "total_promoted": sum(s["num_promoted"] for s in sched),
        "elapsed_seconds": time.time() - started,
    }


def summarise(rows, arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm]
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def make_plot(rows, args, out_path):
    """Two panels: what TAVS costs (accuracy) and what it saves (verifications)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"TAVS vs Full Verification, no attack "
        f"({len(args.seed_list)} seeds, {args.rounds} rounds, CIFAR-10)",
        fontsize=14, fontweight="bold",
    )
    colours = {"tavs": "#1b9e77", "full_verification": "#d95f02"}
    names = {"tavs": "TAVS", "full_verification": "Full Verification"}

    # Panel 1: accuracy over rounds, mean with min-max band across seeds.
    ax = axes[0]
    for arm in ARMS:
        traj = np.array([r["accuracy_trajectory"] for r in rows if r["arm"] == arm])
        x = np.arange(traj.shape[1])
        ax.plot(x, traj.mean(0), color=colours[arm], lw=2, label=f"{names[arm]} (mean)")
        # Band is min-max across seeds, not a confidence interval: with 3 seeds a
        # CI would imply more precision than the data supports.
        ax.fill_between(x, traj.min(0), traj.max(0), color=colours[arm], alpha=0.18)
    ax.axhline(0.1, ls=":", c="grey", lw=1)
    ax.text(0.5, 0.105, "random guess", fontsize=8, color="grey")
    ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy")
    ax.set_title("Model quality (shaded = min-max across seeds)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # Panel 2: the saving, with per-seed points so spread is visible.
    ax = axes[1]
    for i, arm in enumerate(ARMS):
        vals = [r["total_verified"] for r in rows if r["arm"] == arm]
        ax.bar(i, np.mean(vals), 0.5, color=colours[arm], edgecolor="black")
        ax.scatter([i] * len(vals), vals, color="black", zorder=3, s=25)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([names[a] for a in ARMS])
    ax.set_ylabel(f"Total verifications over {args.rounds} rounds")
    ax.set_title("Verification cost (dots = individual seeds)")
    ax.grid(alpha=0.3, axis="y")

    t = np.mean([r["total_verified"] for r in rows if r["arm"] == "tavs"])
    f = np.mean([r["total_verified"] for r in rows if r["arm"] == "full_verification"])
    if f:
        ax.annotate(f"{(1 - t / f) * 100:.0f}% fewer", xy=(0, t), xytext=(0.5, (t + f) / 2),
                    fontsize=13, fontweight="bold", color="#1b9e77", ha="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=8)
    parser.add_argument("--target-k", type=int, default=150)
    parser.add_argument("--clip-factor", type=float, default=2.0)
    parser.add_argument("--data-alpha", type=float, default=0.3)
    parser.add_argument("--results-dir", default="results/tavs_vs_full_seeded")
    args = parser.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    rows = [run_one(arm, cls, seed, args)
            for seed in args.seed_list
            for arm, cls in ARMS.items()]

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seeded_results.json").write_text(json.dumps(
        {"config": {k: v for k, v in vars(args).items() if k != "seed_list"},
         "seeds": args.seed_list, "runs": rows}, indent=2))

    plot_path = out_dir / "tavs_vs_full_no_attack.png"
    make_plot(rows, args, plot_path)

    acc = {a: summarise(rows, a, "late_accuracy") for a in ARMS}
    ver = {a: summarise(rows, a, "total_verified") for a in ARMS}

    print(f"\n{'=' * 76}")
    print(f"TAVS vs FULL VERIFICATION - no attack, {len(args.seed_list)} seeds")
    print(f"{'=' * 76}")
    print(f"{'arm':>20} {'late acc (mean+-std)':>24} {'verifications':>16}")
    for a in ARMS:
        print(f"{a:>20} {acc[a]['mean']:>15.3f} +-{acc[a]['std']:.3f} "
              f"{ver[a]['mean']:>16.0f}")

    d = acc["tavs"]["mean"] - acc["full_verification"]["mean"]
    pooled = max(acc["tavs"]["std"], acc["full_verification"]["std"])
    saving = 1 - ver["tavs"]["mean"] / ver["full_verification"]["mean"]

    print(f"\n  verification saving : {saving * 100:.0f}%")
    print(f"  accuracy difference : {d:+.3f}  (largest within-arm std {pooled:.3f})")
    if abs(d) < pooled:
        print(f"  -> difference is SMALLER than the seed-to-seed spread: with "
              f"{len(args.seed_list)} seeds\n     this is consistent with no accuracy cost, "
              f"but does not prove equivalence.")
    else:
        print(f"  -> difference EXCEEDS the seed spread; likely real, but "
              f"{len(args.seed_list)} seeds\n     is thin evidence -- confirm with more before "
              f"citing the magnitude.")
    print(f"\nPlot: {plot_path}\nJSON: {out_dir / 'seeded_results.json'}")


if __name__ == "__main__":
    main()
