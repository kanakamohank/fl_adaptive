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
    split / cohort / noise settings do not collide.

    Encodes: split mode; non-Dirichlet-alpha when the split uses it; client
    pool size and per-round cohort when they differ from the historical
    100/20 defaults; label-noise fraction/rate and type when non-clean;
    num_classes when it departs from CIFAR-10's 10.

    Every non-default segment is added only when the arg is set away from the
    default so existing run paths (100 clients, 20/round, uniform noise, 10
    classes) remain byte-identical -- ONLY those runs share directories with
    prior identical runs. Any change to any encoded knob produces a fresh
    directory, matching the caller's "do not override unless strictly same
    config" invariant.
    """
    split = getattr(args, "data_split", "dirichlet")
    parts = [f"split-{split}"]
    if split == "dirichlet":
        parts.append(f"a{args.data_alpha:g}")

    # Client-count and cohort-size segments. Kept out of the tag when they
    # match the historical defaults 100 pool / 20 per-round so existing
    # results at r20/split-iid_noiseC40_R30/ remain reachable without
    # migration. Changing either bumps the tag and produces a fresh dir --
    # e.g. Priority 2's 50-client run lands in a new subtree, not on top of
    # the 100-client baseline.
    num_clients = getattr(args, "num_clients", 100)
    cpr = getattr(args, "clients_per_round", 20)
    if num_clients != 100 or cpr != 20:
        parts.append(f"N{num_clients}_C{cpr}")
    # Dataset in the tag so CIFAR-100 runs do not clobber CIFAR-10 paths.
    _ds = getattr(args, "dataset", "cifar10")
    if _ds != "cifar10":
        parts.append(_ds)

    # Noise is active only when BOTH knobs are > 0 -- exactly one of them
    # being zero produces zero flips per client, so it is a "clean" run and
    # we drop into the clean branch. But the previous version was permissive
    # -- ANY zero produced the "clean" tag -- which meant
    # `--noisy-client-fraction 0.4 --label-noise-rate 0` collided with a
    # genuinely clean run AND with any other config sharing a zero knob.
    # Guard by requiring both to be strictly zero for a clean run; a partial
    # zero raises because it is not a config anyone should be running (the
    # noise is dead but the path would advertise otherwise).
    nf_gt = args.noisy_client_fraction > 0
    nr_gt = args.label_noise_rate > 0
    if nf_gt != nr_gt:
        raise ValueError(
            f"noise knobs disagree: --noisy-client-fraction "
            f"{args.noisy_client_fraction} and --label-noise-rate "
            f"{args.label_noise_rate}. Set BOTH to 0 for a clean run, or "
            f"both to positive for noise. A single zero silently produces "
            f"no flips and would collide with existing paths."
        )
    if nf_gt and nr_gt:
        nf = int(round(args.noisy_client_fraction * 100))
        nr = int(round(args.label_noise_rate * 100))
        parts.append(f"noiseC{nf:02d}_R{nr:02d}")
        noise_type = getattr(args, "label_noise_type", "uniform")
        if noise_type != "uniform":
            # For cifar10n, encode the chosen annotator stream too so runs at
            # different noise characteristics (worst vs aggre vs random1 ...)
            # do not collide at the same path. For pairflip/uniform the type
            # alone is sufficient.
            if noise_type == "cifar10n":
                parts.append(
                    f"type-cifar10n-{getattr(args, 'cifar10n_label_set', 'random1')}"
                )
            else:
                parts.append(f"type-{noise_type}")
        if getattr(args, "label_noise_num_classes", 10) != 10:
            parts.append(f"K{args.label_noise_num_classes}")
    else:
        parts.append("clean")
    # Non-default trust signal lands in its own directory so a bvd baseline
    # and a small-loss-fraction run never collide on disk even at identical
    # noise config.
    trust_signal = getattr(args, "trust_signal", "bvd")
    if trust_signal != "bvd":
        parts.append(f"trust-{trust_signal.replace('_', '-')}")
    # Encode a non-default small-loss rescale so SLF runs at different
    # rescale windows do not collide on disk.
    rescale = getattr(args, "small_loss_rescale", None)
    if rescale and trust_signal == "small_loss_fraction":
        parts_r = [p.strip() for p in str(rescale).split(",") if p.strip()]
        if len(parts_r) == 2:
            lo = int(round(float(parts_r[0]) * 100))
            hi = int(round(float(parts_r[1]) * 100))
            parts.append(f"rescale-{lo:02d}-{hi:02d}")
    # Encode non-default local_epochs so oracle runs do not clobber pilot
    # runs at the same noise config but different epoch counts.
    _le = getattr(args, "local_epochs", None)
    if _le is not None:
        parts.append(f"ep{int(_le)}")
    if getattr(args, "log_oracle_signals", False):
        parts.append("oracle")
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

    # tavs_skip inherits the dataclass defaults from TavsEspConfig for the
    # knobs that used to be per-arm overrides:
    #   * disable_trust_weighted_aggregation: default True (num_examples only).
    #     At n=10 pair-flip 6% pilots the trust-weighted variant was mildly
    #     WORSE than num_examples-only on late accuracy (delta +0.0021,
    #     uncorrected p=0.027, does NOT survive multi-test Bonferroni). Trust
    #     EMA also fails to separate noisy from clean in this regime (Cohen's
    #     d ~= 0 in every arm including full_verify), so trust-weighted
    #     aggregation has no signal to leverage. Occam's razor: drop the knob.
    #     The tavs_skip_noweight ablation is retired because tavs_skip now
    #     runs the exact same aggregation.
    #   * initial_trust=0.25, tau_ramp=5.0, bootstrap_verify_new_clients=True:
    #     the "Tier-1 floor" defaults. The tavs_skip_nofloor ablation that
    #     tried to remove them is retired: raising initial_trust to 0.5 to
    #     bypass Tier 1 pushed the bayesian_posterior_weight per client from
    #     ~0.2 to ~0.5, which slammed the gamma_budget=0.35 phase-2 constraint
    #     and demoted almost everyone BACK to V. Empirically nofloor verified
    #     MORE than tavs (skip 0.368 vs 0.461), opposite of the ablation's
    #     stated intent. Cleanly isolating the floor requires co-moving
    #     gamma_budget/c_lambda too, which is a separate design pass.
    rescale = getattr(args, "small_loss_rescale", None)
    rescale_lo, rescale_hi = 0.0, 1.0
    if rescale:
        parts_r = [p.strip() for p in rescale.split(",")]
        if len(parts_r) != 2:
            raise ValueError(f"--small-loss-rescale must be 'lo,hi'; got {rescale!r}")
        rescale_lo = float(parts_r[0])
        rescale_hi = float(parts_r[1])
    tavs_config = TavsEspConfig(
        theta_low=0.3, theta_high=0.7, alpha_trust=0.9, gamma_budget=0.35,
        tau_ramp=5.0,
        k_trust=3, target_k=150,
        detection_threshold=5.0,
        clip_promoted_updates=True, promoted_clip_factor=2.0,
        cosine_filter_promoted=False,
        enable_outlier_detection=True,
        trust_signal=getattr(args, "trust_signal", "bvd"),
        small_loss_rescale_lo=rescale_lo,
        small_loss_rescale_hi=rescale_hi,
        log_oracle_signals=bool(getattr(args, "log_oracle_signals", False)),
        log_per_sample_losses_at_round=int(getattr(args, "log_per_sample_losses_at_round", 0)),
    )

    # EMA-convergence preflight for non-BVD trust signals. BVD's inlier
    # default (~0.9) trivially clears theta_low; non-BVD signals (e.g.
    # small-loss-fraction, val-accuracy) live in different numerical
    # ranges and can leave trust pinned below theta_low for the whole
    # pilot, which collapses the skip rate (what we saw at SLF seed=1).
    # Project the midpoint of the rescale window and refuse to launch
    # if even that fails to clear theta_low + 0.02 margin.
    if tavs_config.trust_signal != "bvd":
        from src.tavs_v2 import TavsEspStrategy as _S
        _probe = _S(config=tavs_config)
        raw_mid = (rescale_lo + rescale_hi) / 2.0
        rescaled_mid = ((raw_mid - rescale_lo) / (rescale_hi - rescale_lo)
                        if rescale_hi > rescale_lo else 0.5)
        projected = _probe.scheduler.expected_trust_from_participation(
            expected_raw=rescaled_mid,
            total_rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            num_clients=args.num_clients,
            initial_trust=tavs_config.initial_trust,
        )
        margin = 0.02
        print(f"[preflight] trust_signal={tavs_config.trust_signal} "
              f"rescale=[{rescale_lo:.2f},{rescale_hi:.2f}] "
              f"midpoint_rescaled={rescaled_mid:.2f} "
              f"projected_trust_after_pilot={projected:.3f} "
              f"theta_low={tavs_config.theta_low:.2f}")
        if projected < tavs_config.theta_low + margin:
            raise RuntimeError(
                f"preflight: midpoint signal {rescaled_mid:.2f} projects "
                f"trust={projected:.3f} < theta_low+margin="
                f"{tavs_config.theta_low + margin:.3f} after "
                f"{args.rounds} rounds with ~"
                f"{args.rounds * args.clients_per_round / max(1, args.num_clients):.1f} "
                f"verifications per client. Expected skip collapse. "
                f"Fix: pass --small-loss-rescale lo,hi to stretch the "
                f"signal range, or lower alpha_trust / initial_trust."
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

    _client_epochs_override = getattr(args, "local_epochs", None)
    _pipeline_kwargs = {}
    if _client_epochs_override is not None:
        _pipeline_kwargs["client_epochs"] = int(_client_epochs_override)
    config = PipelineConfig(
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        byzantine_fraction=0.0,   # honest only
        tavs_config=tavs_config,
        strategy_class=strategy_class,
        dataset=getattr(args, "dataset", "cifar10"),
        label_noise_num_classes=int(getattr(args, "label_noise_num_classes", 10)),
        **_pipeline_kwargs,
        data_split=args.data_split,
        data_alpha=args.data_alpha,
        # Label noise on a subset of clients: the only source of client-level
        # DIFFERENTIATION in this otherwise-honest pilot. Without it TAVS's
        # trust EMA has no signal to converge on (see the r20/skip49 result).
        label_noise_client_fraction=args.noisy_client_fraction,
        label_noise_rate=args.label_noise_rate,
        label_noise_type=args.label_noise_type,
        cifar10n_path=getattr(args, "cifar10n_path", None),
        cifar10n_label_set=getattr(args, "cifar10n_label_set", "random1"),
        seed=seed,
        # Encode the label-noise config in the path so IID+noise runs do not
        # overwrite the earlier no-noise r20/skip49 results the analysis
        # already cites.
        output_dir=str(Path(args.results_dir) / f"r{args.rounds}" /
                       _config_tag(args) /
                       f"skip{int(skip_rate * 100):02d}" /
                       f"{arm}_seed{seed}"),
    )

    # Skip re-execution if the pipeline already dropped a completed
    # pipeline_results.json into this arm's output_dir. Used for --skip-completed
    # when resuming from an interrupted run. The row we return is reconstructed
    # from the cached JSON so downstream analysis is oblivious to the resume.
    #
    # A cached run is treated as valid iff pipeline_results.json exists AND has
    # a non-empty server_accuracies list -- a half-written file from a crash
    # (empty accuracies, or missing keys) is re-run rather than silently
    # accepted as complete.
    output_dir = Path(config.output_dir)
    cached_path = output_dir / "pipeline_results.json"
    if getattr(args, "skip_completed", False) and cached_path.exists():
        try:
            cached = json.loads(cached_path.read_text())
            server_accuracies = cached.get("server_accuracies") or []
            sched = cached.get("scheduling_history") or []
        except (OSError, json.JSONDecodeError):
            server_accuracies, sched = [], []
        if server_accuracies:
            total_verified = sum(s.get("num_verified", 0) for s in sched)
            total_promoted = sum(s.get("num_promoted", 0) for s in sched)
            total_cohort = total_verified + total_promoted
            observed_skip = (total_promoted / total_cohort) if total_cohort else 0.0
            late_window = max(1, int(round(args.rounds * 0.25)))
            diag = {}
            try:
                s = json.loads((output_dir / "experiment_summary.json").read_text())
                diag = s.get("noise_diagnostic", {}) or {}
            except (OSError, json.JSONDecodeError):
                pass
            print(f"\n{'=' * 70}\n{arm}  seed={seed}   [CACHED, skipping re-run]"
                  f"\n{'=' * 70}")
            return {
                "arm": arm, "seed": seed,
                "final_accuracy": server_accuracies[-1],
                "late_window": late_window,
                "late_accuracy": statistics.mean(server_accuracies[-late_window:]),
                "accuracy_trajectory": server_accuracies,
                "total_verified": total_verified,
                "total_promoted": total_promoted,
                "observed_skip_rate": observed_skip,
                "elapsed_seconds": 0.0,   # cached: no measurable execution cost
                "noise_diagnostic": diag,
                "cached": True,
            }

    print(f"\n{'=' * 70}\n{arm}  seed={seed}  "
          f"(rounds={args.rounds}, skip_target={skip_rate:.2f})\n{'=' * 70}")
    started = time.time()
    results = TAVSESPPipeline(config).run_simulation()

    sched = results.scheduling_history
    total_verified = sum(s["num_verified"] for s in sched)
    total_promoted = sum(s["num_promoted"] for s in sched)
    total_cohort = total_verified + total_promoted
    observed_skip = (total_promoted / total_cohort) if total_cohort else 0.0

    # Slurp the diagnostic that the pipeline dropped into experiment_summary.json.
    # Reading it here (rather than recomputing) keeps the pilot's report in
    # sync with what the pipeline actually persisted -- if the two ever
    # diverge, that is a bug worth catching, not a difference to paper over.
    diag = {}
    try:
        with open(Path(config.output_dir) / "experiment_summary.json") as f:
            diag = json.load(f).get("noise_diagnostic", {}) or {}
    except FileNotFoundError:
        pass

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
        # Noise diagnostic -- see PipelineResults._noise_diagnostic docstring.
        # Absent (None) fields signal "no noise injected" or "no clients seen".
        "noise_diagnostic": diag,
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
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated seed list. Default is 10 seeds. "
                             "Pilot 3 at 3 seeds showed a consistent +0.3pp TAVS "
                             "advantage over random skip on late accuracy but a "
                             "seed-3 inversion in the trust-vs-noise mechanism "
                             "diagnostic (rank-biserial correlation flipped sign). "
                             "10 seeds resolves whether that inversion is a tail "
                             "event or ~1-in-3 mechanism instability -- a stability "
                             "question, not a power question. Reduce only for "
                             "quick smoke tests; do not report on <5 seeds.")
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
    parser.add_argument("--noisy-client-fraction", type=float, default=0.4,
                        help="Fraction of the client pool that gets noisy labels. "
                             "Default 0.4 (=40 of 100 clients) -- STRONGER than the "
                             "0.2 pilot 2 setting, which produced only a +0.17pp "
                             "TAVS-vs-random gap on late accuracy. This pilot tests "
                             "whether the mechanism scales with the differentiation "
                             "signal or hits a ceiling.")
    parser.add_argument("--label-noise-rate", type=float, default=0.3,
                        help="Fraction of a noisy client's labels flipped to a "
                             "wrong class (default 0.3). Combined with "
                             "--noisy-client-fraction=0.4 that is 12%% of total labels "
                             "corrupted, up from 4%% in pilot 2.")
    parser.add_argument("--label-noise-type", default="uniform",
                        choices=("uniform", "pairflip", "cifar10n"),
                        help="Noise TYPE. 'uniform': wrong label drawn uniformly "
                             "at random. 'pairflip': noisy sample swapped with its "
                             "confusable partner (CIFAR-10 default map). 'cifar10n': "
                             "use real human-annotator labels from Wei et al. ICLR "
                             "2022; noise is NON-collusive because different "
                             "annotators make different mistakes, which is the "
                             "regime pair-flip at 30%% broke BVD in. Requires "
                             "--cifar10n-path and --cifar10n-label-set.")
    parser.add_argument("--cifar10n-path", default=None,
                        help="Path to CIFAR-10_human.pt (download from "
                             "github.com/UCSC-REAL/cifar-10-100n). Required when "
                             "--label-noise-type=cifar10n.")
    parser.add_argument("--cifar10n-label-set", default="random1",
                        choices=("clean", "aggre", "worst",
                                 "random1", "random2", "random3"),
                        help="Which CIFAR-10N annotator stream to use. Approximate "
                             "noise rates against clean labels: aggre ~9%% (3-worker "
                             "majority), random1/2/3 ~17-18%% (single worker), worst "
                             "~40%%. Ignored unless --label-noise-type=cifar10n.")
    parser.add_argument("--trust-signal", default="bvd",
                        choices=("bvd", "small_loss_fraction"),
                        help="Which signal drives the trust EMA. 'bvd' (default) "
                             "uses the BVD outlier Z-score. 'small_loss_fraction' "
                             "swaps in a per-client scalar: the fraction of a "
                             "local 10%% held-out val split that the INCOMING "
                             "global model predicts correctly against the "
                             "client's own labels. Co-teaching-style "
                             "(Han et al. 2018), adapted for FL as a drop-in "
                             "replacement for BVD when BVD does not separate "
                             "noisy from clean clients.")
    parser.add_argument("--small-loss-rescale", default=None,
                        help="Optional 'lo,hi' linear rescale of the raw "
                             "small_loss_fraction value before it enters the "
                             "trust EMA. e.g. '0.3,0.7' stretches val_acc "
                             "0.3-0.7 across trust [0, 1]. Needed because "
                             "raw val-accuracy on a mid-training model "
                             "(~0.3-0.6) never climbs above theta_low=0.3 under "
                             "the default EMA; without rescaling the Tier-1 "
                             "floor clamps everyone to Verified and collapses "
                             "the skip rate. Ignored unless --trust-signal="
                             "small_loss_fraction. Default: no rescale.")
    parser.add_argument("--dataset", default="cifar10",
                        choices=("cifar10", "cifar100"),
                        help="Which dataset to run. cifar10 (default, 10 classes) "
                             "or cifar100 (100 classes). CIFAR-100 uses the same "
                             "small cifar_cnn with its final FC resized to 100.")
    parser.add_argument("--label-noise-num-classes", type=int, default=10,
                        help="Number of classes the dataset has. Required to be "
                             "set to 100 when --dataset=cifar100. Also used by "
                             "the uniform-noise flipper so a flipped label lands "
                             "on one of the (K-1) wrong classes. Default: 10.")
    parser.add_argument("--log-per-sample-losses-at-round", type=int, default=0,
                        help="One-off diagnostic for the mechanism "
                             "visualisation. When > 0, each verified client "
                             "additionally logs its full per-sample pretrain "
                             "loss list at THAT specific round, dumped into "
                             "oracle_signal_history. Needed to plot the "
                             "bimodal-vs-unimodal loss histogram directly. "
                             "Default 0 = disabled.")
    parser.add_argument("--log-oracle-signals", action="store_true",
                        help="Opt-in diagnostic logging for the oracle "
                             "noise-detection experiment. Each client logs "
                             "pre-training per-class loss / variance, "
                             "memorization gap (epoch_1 loss - epoch_last), "
                             "update norm, first-batch grad norm; the strategy "
                             "logs cosine vs weighted/simple mean per client "
                             "per round. All dumped into "
                             "pipeline_results['oracle_signal_history']. "
                             "Zero cost when off. Does not change scheduling.")
    parser.add_argument("--local-epochs", type=int, default=None,
                        help="Override local_epochs for the oracle experiment. "
                             "Default pulls from PipelineConfig (currently 2).")
    parser.add_argument("--results-dir", default="results/pilot_skip_comparison")
    parser.add_argument("--skip-completed", action="store_true",
                        help="For each (arm, seed), skip re-execution if the arm's "
                             "output_dir already contains a valid pipeline_results.json "
                             "(non-empty server_accuracies). Reconstructs the row from "
                             "the cached JSON so downstream stats and plots match a "
                             "fresh run. Use to resume from an interrupted run without "
                             "redoing completed work. A half-written cache from a "
                             "crash is treated as invalid and re-run.")
    args = parser.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    # Guard: cifar100 has 100 classes and the noise flipper + model head both
    # consult --label-noise-num-classes. A mismatch silently produces a 10-way
    # head on 100-class data (training degenerates) and a noise map that only
    # flips among the first 10 labels. Refuse at parse time.
    _expected_k = {"cifar10": 10, "cifar100": 100}
    _exp = _expected_k.get(getattr(args, "dataset", "cifar10"))
    if _exp is not None and int(getattr(args, "label_noise_num_classes", 10)) != _exp:
        raise SystemExit(
            f"--dataset={args.dataset} requires --label-noise-num-classes={_exp}; "
            f"got {args.label_noise_num_classes}."
        )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Two-pass to keep the comparison honest. First pass: every arm that
    # produces its OWN skip rate (full_verify skips 0%; tavs_skip's skip rate
    # emerges from the scheduler). Read TAVS's observed rate, then run
    # random_skip at that rate in pass two.
    #
    # The one-pass fixed-rate mode is preserved via --skip-rate for
    # sensitivity checks but is not a matched-rate comparison.
    rows = []
    for seed in args.seed_list:
        for arm in ("full_verify", "tavs_skip"):
            rows.append(run_one(arm, seed, args, skip_rate=0.0))

    tavs_rates = [r["observed_skip_rate"] for r in rows if r["arm"] == "tavs_skip"]
    matched_rate = (args.skip_rate if args.skip_rate is not None
                    else float(np.mean(tavs_rates)))
    print(f"\n[matched-rate] tavs_skip     observed skip = {tavs_rates}")
    print(f"[matched-rate] -> random_skip target = {matched_rate:.3f}  "
          f"(matched to tavs_skip mean)")

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

    # Noise diagnostic: did TAVS put lower trust on the actually-noisy clients?
    # Only meaningful for tavs_skip (full_verify has no scheduler decision and
    # random_skip's trust EMA is not used by the policy). Reported per seed
    # from experiment_summary.json's noise_diagnostic block.
    tavs_rows = [r for r in rows if r["arm"] == "tavs_skip"]
    if tavs_rows and any((r["noise_diagnostic"] or {}).get("noise_injected") for r in tavs_rows):
        print("\n  MECHANISM DIAGNOSTIC (tavs_skip): did trust find the noisy clients?")
        print(f"    {'seed':>4} {'n_noisy':>8} {'mean_trust_noisy':>18} "
              f"{'mean_trust_clean':>18} {'gap(C-N)':>10}    {'bottom-k overlap'}")
        for r in tavs_rows:
            d = r["noise_diagnostic"] or {}
            if not d.get("noise_injected"):
                continue
            mn = d.get("mean_trust_noisy", float("nan")) or float("nan")
            mc = d.get("mean_trust_clean", float("nan")) or float("nan")
            gap = d.get("trust_gap_clean_minus_noisy")
            gap_s = f"{gap:+.4f}" if gap is not None else "   n/a"
            ov = d.get("bottom_k_overlap_with_noisy", 0) or 0
            ovp = d.get("bottom_k_overlap_pct", 0.0) or 0.0
            print(f"    {r['seed']:>4} {d['n_noisy']:>8} "
                  f"{mn:>18.4f} {mc:>18.4f} {gap_s:>10}    "
                  f"{ov}/{d['n_noisy']} ({ovp:.0f}%)")

    print(f"\nPlot: {plot_path}")
    print(f"JSON: {out_dir / 'pilot_results.json'}")


if __name__ == "__main__":
    main()
