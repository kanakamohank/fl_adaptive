#!/usr/bin/env python3
"""
Sweep the promoted-update clip radius.

Clipping bounds how far an unverified (promoted) update may sit from the verified
consensus:

    g_i <- c + (g_i - c) * min(1, tau / ||g_i - c||)
    tau  = clip_factor * median_j ||g_j - c||   over verified clients j

At clip_factor=1.0 the first clipped run showed containment working (server loss
capped at ~2.5 versus 2049 unclipped) but clipped ~95% of promoted updates --
51 of 54 in the heavy-attack run. At that rate it behaves as near-uniform
magnitude normalisation rather than an outlier clip, shrinking honest promoted
clients alongside attackers. That is the likely cause of the residual 0.052
accuracy gap against full verification with NO attacker present.

This sweep finds the radius that keeps containment while letting honest promoted
clients through: watch for the factor where clip_rate falls well below 1.0 and
final accuracy peaks.

Only the TAVS arm is run. FullVerificationStrategy never promotes, so no promoted
update ever exists for it to clip and its result is independent of clip_factor --
running it per factor would burn compute to reproduce the same number.

Usage:
    python -m experiments.clip_factor_sweep
    python -m experiments.clip_factor_sweep --factors 1,2,3,5 --scenario no_attack
    python -m experiments.clip_factor_sweep --include-unclipped   # adds a control arm
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tavs_v2 import PipelineConfig, TavsEspConfig, TAVSESPPipeline

logger = logging.getLogger(__name__)

# Byzantine fraction per scenario, matching verification_strategy_comparison.
SCENARIO_BYZANTINE = {"no_attack": 0.0, "light_attack": 0.15, "heavy_attack": 0.25}


def run_one(clip_factor, args):
    """
    Run a single TAVS pipeline at one clip radius.

    clip_factor of None disables clipping entirely, giving the unclipped control.
    Every other setting is held fixed so the radius is the only variable: the
    Dirichlet split is seeded at 42 inside create_dirichlet_splits, so all arms
    see identical client data.
    """
    clipping_on = clip_factor is not None
    label = "off" if not clipping_on else f"{clip_factor:g}"

    tavs_config = TavsEspConfig(
        theta_low=0.3,
        theta_high=0.7,
        alpha_trust=0.9,
        gamma_budget=0.35,
        tau_ramp=5.0,
        k_trust=3,
        target_k=args.target_k,
        detection_threshold=2.0,
        clip_promoted_updates=clipping_on,
        promoted_clip_factor=clip_factor if clipping_on else 1.0,
    )

    pipeline_config = PipelineConfig(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        byzantine_fraction=SCENARIO_BYZANTINE[args.scenario],
        tavs_config=tavs_config,
        attack_types=["layerwise", "distributed"],
        attack_intensities=[1.5, 2.0],
        data_alpha=0.3,
        output_dir=str(Path(args.results_dir) / f"clip_{label}"),
    )

    print(f"\n{'=' * 70}\nclip_factor = {label}  ({args.scenario}, {args.rounds} rounds)\n{'=' * 70}")
    started = time.time()
    results = TAVSESPPipeline(pipeline_config).run_simulation()
    elapsed = time.time() - started

    scheduling = results.scheduling_history
    promoted = sum(s["num_promoted"] for s in scheduling)
    clipped = sum(s.get("num_clipped") or 0 for s in scheduling)

    return {
        "clip_factor": clip_factor,
        "clipping_enabled": clipping_on,
        "final_accuracy": results.server_accuracies[-1],
        "peak_accuracy": max(results.server_accuracies),
        "max_loss": max(results.server_losses),
        "total_verified": sum(s["num_verified"] for s in scheduling),
        "total_promoted": promoted,
        "total_clipped": clipped,
        # The diagnostic that matters: 1.0 means every promoted update was
        # clipped, i.e. the radius is acting as normalisation, not as a filter.
        "clip_rate": (clipped / promoted) if promoted else None,
        "accuracy_trajectory": results.server_accuracies,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factors", default="1,2,3",
                        help="Comma-separated clip factors (default: 1,2,3)")
    parser.add_argument("--scenario", default="no_attack", choices=sorted(SCENARIO_BYZANTINE),
                        help="Attack scenario (default: no_attack, which isolates "
                             "how much honest signal the clip suppresses)")
    parser.add_argument("--include-unclipped", action="store_true",
                        help="Also run with clipping disabled, as a control")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=8)
    parser.add_argument("--target-k", type=int, default=150)
    parser.add_argument("--results-dir", default="results/clip_factor_sweep")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    factors = [float(f) for f in args.factors.split(",") if f.strip()]
    if args.include_unclipped:
        factors = [None] + factors

    rows = [run_one(f, args) for f in factors]

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_results.json").write_text(json.dumps(
        {"config": vars(args), "runs": rows}, indent=2
    ))

    print(f"\n{'=' * 78}")
    print(f"CLIP FACTOR SWEEP - {args.scenario}, {args.rounds} rounds")
    print(f"{'=' * 78}")
    print(f"{'factor':>8} {'final acc':>10} {'peak acc':>9} {'max loss':>10} "
          f"{'promoted':>9} {'clipped':>8} {'clip rate':>10}")
    for r in rows:
        label = "off" if not r["clipping_enabled"] else f"{r['clip_factor']:g}"
        rate = "-" if r["clip_rate"] is None else f"{r['clip_rate']:.2f}"
        print(f"{label:>8} {r['final_accuracy']:>10.3f} {r['peak_accuracy']:>9.3f} "
              f"{r['max_loss']:>10.2f} {r['total_promoted']:>9d} "
              f"{r['total_clipped']:>8d} {rate:>10}")

    print(f"\nRead it this way:")
    print(f"  clip rate near 1.00 -> radius too tight; honest promoted clients are")
    print(f"                         being shrunk along with attackers")
    print(f"  max loss above ~10  -> radius too loose; containment is being lost")
    print(f"  pick the largest factor that still holds max loss down")
    print(f"\nJSON: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
