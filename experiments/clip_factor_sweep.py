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

Measured behaviour of the mechanism, to set expectations:
  * clip_rate does fall with the radius, but only once the radius clears the
    cohort's own spread. On a tightly clustered cohort it stayed at 1.00 through
    factor 3 and only broke at 5; on a more heterogeneous one it fell
    1.00 -> 0.80 -> 0.42 -> 0.07 across factors 1, 2, 3, 5.
  * containment degrades very gracefully. Against an attacker three orders of
    magnitude out of scale, the aggregate norm moved only 0.2031 -> 0.2040
    between factor 1 and factor 10, because such an update is clipped hard at
    any of these radii. So a larger factor is close to free against gross
    attacks; the risk is a subtle attacker sitting just inside the ball, which
    is why the chosen factor should be re-checked on heavy_attack.

Only the TAVS arm is run. FullVerificationStrategy never promotes, so no promoted
update ever exists for it to clip and its result is independent of clip_factor --
running it per factor would burn compute to reproduce the same number.

Results are written under <results-dir>/<scenario>/<cosine-setting>/, so sweeps
on different scenarios or different cosine settings do not overwrite each other.

Usage:
    # Step 1: find the radius, with no adversary to confound the measurement
    python -m experiments.clip_factor_sweep --include-unclipped

    # Step 2: confirm the chosen radius still contains a real attack
    python -m experiments.clip_factor_sweep --factors 3 --scenario heavy_attack

    # Cosine false-positive rate: no attacker, so every rejection is honest
    # signal thrown away. Compare the two runs' accuracy and cos rate.
    python -m experiments.clip_factor_sweep --factors 2 --scenario no_attack --cosine on
    python -m experiments.clip_factor_sweep --factors 2 --scenario no_attack --cosine off
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


def _cosine_tag(args):
    """
    Path component identifying the cosine setting, e.g. 'cos_on_min0' / 'cos_off'.

    The threshold is in the tag, not just the on/off state: a sweep over
    --cosine-min would otherwise write every threshold to the same directory and
    leave only the last one on disk.
    """
    if not args.cosine:
        return "cos_off"
    return f"cos_on_min{args.cosine_min:g}"


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
        # tau_z = 5.0, not the 2.0 used previously.
        #
        # 2.0 flagged 39.9% of VERIFIED clients as Byzantine in a run with ZERO
        # attackers, escalating 4.8% -> 73.5% across 60 rounds. The escalation is a
        # feedback loop: sigma^2 is re-estimated from inliers ONLY, so flagging
        # clients shrinks the variance estimate, which raises everyone's z, which
        # flags more. Four rounds ended with every verified client flagged.
        #
        # max_z is a MAX over 10 parameter blocks, so it needs headroom that a
        # single-statistic threshold does not. Measured ROC on honest-vs-attacker
        # cohorts at realistic heterogeneity:
        #
        #     tau_z   honest FP   TP@3x   TP@10x   TP@100x
        #       2.0        66%     100%     100%      100%
        #       5.0         0%     100%     100%      100%
        #      10.0         0%      80%     100%      100%
        #      20.0         0%       3%     100%      100%
        #
        # 5.0 strictly dominates 2.0: no false positives AND full detection of even a
        # modest 3x attacker. It is also already the TavsEspConfig default -- these
        # scripts were overriding a correct default with a broken value.
        detection_threshold=5.0,
        enable_outlier_detection=args.detection,
        clip_promoted_updates=clipping_on,
        promoted_clip_factor=clip_factor if clipping_on else 1.0,
        # Direction gate, independent of the magnitude clip above. Exposed here
        # so the two can be turned on and off separately -- with cosine wired to
        # its default the sweep would silently measure clip+cosine together and
        # attribute the combined effect to the radius alone.
        cosine_filter_promoted=args.cosine,
        promoted_cosine_min=args.cosine_min,
    )

    pipeline_config = PipelineConfig(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        byzantine_fraction=SCENARIO_BYZANTINE[args.scenario],
        tavs_config=tavs_config,
        attack_types=["layerwise", "distributed"],
        attack_intensities=[1.5, 2.0],
        data_alpha=args.data_alpha,
        # Scenario is part of the path: without it a heavy_attack sweep would
        # overwrite a no_attack sweep run earlier at the same factor, silently
        # destroying the comparison the two runs exist to make.
        # The cosine setting is part of the path for the same reason the scenario
        # is: a --cosine off run would otherwise overwrite the --cosine on run at
        # the same factor, destroying the ablation the pair exists to make.
        output_dir=str(Path(args.results_dir) / args.scenario / f"alpha{args.data_alpha:g}" /
                       f"det_{'on' if args.detection else 'off'}" /
                       _cosine_tag(args) / f"clip_{label}"),
    )

    print(f"\n{'=' * 70}\nclip_factor = {label}  ({args.scenario}, {args.rounds} rounds)\n{'=' * 70}")
    started = time.time()
    results = TAVSESPPipeline(pipeline_config).run_simulation()
    elapsed = time.time() - started

    scheduling = results.scheduling_history
    promoted = sum(s["num_promoted"] for s in scheduling)
    clipped = sum(s.get("num_clipped") or 0 for s in scheduling)
    rejected = sum(s.get("num_cosine_rejected") or 0 for s in scheduling)

    return {
        "clip_factor": clip_factor,
        "clipping_enabled": clipping_on,
        "data_alpha": args.data_alpha,
        "cosine_enabled": args.cosine,
        "cosine_min": args.cosine_min,
        "total_cosine_rejected": rejected,
        # Under --scenario no_attack every promoted client is honest, so this
        # rate IS the false-positive rate of the gate. Anything much above zero
        # there means the gate is discarding honest signal.
        "cosine_rejection_rate": (rejected / promoted) if promoted else None,
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
    # Range widened after measuring how clip_rate responds to the radius. Under a
    # tightly clustered cohort the rate stays pinned at 1.00 all the way to
    # factor 3, so 1,2,3 alone can look flat and miss the knee entirely.
    parser.add_argument("--factors", default="1,2,3,5,10",
                        help="Comma-separated clip factors (default: 1,2,3,5,10)")
    parser.add_argument("--scenario", default="no_attack", choices=sorted(SCENARIO_BYZANTINE),
                        help="Attack scenario (default: no_attack, which isolates "
                             "how much honest signal the clip suppresses)")
    parser.add_argument("--include-unclipped", action="store_true",
                        help="Also run with clipping disabled, as a control")
    # Cosine gate: rejects a promoted update whose movement direction disagrees
    # with the verified cohort's mean movement. Orthogonal to the clip, which
    # only bounds magnitude -- an attacker inside the clip radius but pointing
    # backwards is invisible to the clip and caught here.
    parser.add_argument("--cosine", default="off", choices=("on", "off"),
                        # Defaults to off, matching TavsEspConfig. These
                        # scripts defaulted to "on" and so silently kept the
                        # gate active after the config default was flipped,
                        # discarding 30.9% of promoted updates in a run with
                        # no attacker.
                        help="Direction gate on promoted updates (default: on). "
                             "Run 'off' to attribute an effect to the clip alone.")
    parser.add_argument("--cosine-min", type=float, default=0.0,
                        help="Reject a promoted update below this cosine "
                             "(default 0.0 = reject only actively opposing "
                             "directions). Negative values tolerate the "
                             "directional spread that non-IID data creates "
                             "among honest clients.")
    # Dirichlet concentration for the client split. 0.3 is strongly non-IID;
    # a large value (100+) is effectively IID.
    #
    # Worth running as a control, not just a variation. Federated accuracy sat
    # at 0.343 against a centralised ceiling of 0.743, and that gap has two very
    # different possible causes: the genuine cost of heterogeneity, or a defect
    # in the pipeline. An IID run separates them -- if the gap closes, what
    # remains is non-IID cost and the pipeline is sound; if it does not, there is
    # a second problem that has nothing to do with heterogeneity.
    # Detection off gives a true centralised-vs-federated comparison: with it on
    # the federated arm discards updates, so the comparison measures the detector
    # as much as it measures federation. It also disables the Byzantine defence,
    # so it is only valid on no_attack.
    parser.add_argument("--detection", default="on", choices=("on", "off"),
                        help="BVD outlier detection (default on). 'off' removes "
                             "the Byzantine defence -- no_attack runs only.")
    parser.add_argument("--data-alpha", type=float, default=0.3,
                        help="Dirichlet alpha (default 0.3 = strongly non-IID; "
                             "use 100 for effectively IID)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=8)
    parser.add_argument("--target-k", type=int, default=150)
    parser.add_argument("--results-dir", default="results/clip_factor_sweep")
    args = parser.parse_args()
    args.cosine = args.cosine == "on"
    args.detection = args.detection == "on"   # config field is a bool; the flag is a word

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    factors = [float(f) for f in args.factors.split(",") if f.strip()]
    if args.include_unclipped:
        factors = [None] + factors

    rows = [run_one(f, args) for f in factors]

    out_dir = (Path(args.results_dir) / args.scenario /
               f"alpha{args.data_alpha:g}" /
               f"det_{'on' if args.detection else 'off'}" / _cosine_tag(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_results.json").write_text(json.dumps(
        {"config": vars(args), "runs": rows}, indent=2
    ))

    print(f"\n{'=' * 78}")
    print(f"CLIP FACTOR SWEEP - {args.scenario}, {args.rounds} rounds")
    print(f"{'=' * 78}")
    print(f"cosine gate: {'on' if args.cosine else 'off'} (min={args.cosine_min:g})")
    print(f"{'factor':>8} {'final acc':>10} {'peak acc':>9} {'max loss':>10} "
          f"{'promoted':>9} {'clipped':>8} {'clip rate':>10} {'cos rej':>8} {'cos rate':>9}")
    for r in rows:
        label = "off" if not r["clipping_enabled"] else f"{r['clip_factor']:g}"
        rate = "-" if r["clip_rate"] is None else f"{r['clip_rate']:.2f}"
        cos_rate = "-" if r["cosine_rejection_rate"] is None else f"{r['cosine_rejection_rate']:.2f}"
        print(f"{label:>8} {r['final_accuracy']:>10.3f} {r['peak_accuracy']:>9.3f} "
              f"{r['max_loss']:>10.2f} {r['total_promoted']:>9d} "
              f"{r['total_clipped']:>8d} {rate:>10} "
              f"{r['total_cosine_rejected']:>8d} {cos_rate:>9}")

    print(f"\nRead it this way:")
    print(f"  clip rate near 1.00 -> radius too tight; honest promoted clients are")
    print(f"                         being shrunk along with the attackers. This is")
    print(f"                         usually the binding constraint.")
    print(f"  max loss climbing   -> radius too loose; containment slipping. Note")
    print(f"                         that a grossly out-of-scale attacker stays")
    print(f"                         contained even at large factors, so this may")
    print(f"                         not move at all -- confirm on heavy_attack")
    print(f"                         before trusting a large factor.")
    print(f"  Pick the smallest factor whose clip rate is well below 1.00, then")
    print(f"  re-run that factor on heavy_attack to confirm containment holds.")
    print(f"\nJSON: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
