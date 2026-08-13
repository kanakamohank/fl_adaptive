#!/usr/bin/env python3
"""
Wall-clock benchmark for TAVS-ESP server-side cost.

Unlike `quick_timing_test.py`, this benchmark:
  * times the real Algorithm 1/2/3 classes, not reimplementations of them
  * builds every fixture *outside* the timed region
  * runs warmup iterations and synchronises the accelerator before stopping a timer
  * repeats each measurement and reports median with a bootstrap confidence
    interval instead of a single sample
  * interleaves the arms so thermal drift cancels rather than biasing one arm
  * expresses overhead against a measured client-training reference, because
    "percent overhead" is only meaningful relative to the cost of a real round

Three arms are compared on identical fixtures:
  fedavg        weighted mean over full-dimensional updates (no defence)
  full_verify   project + detect every client every round (traditional defence)
  tavs_esp      trust-adaptive scheduling, project + detect the scheduled subset
"""

import argparse
import json
import platform
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from src.core.models import CIFARCNN, ModelStructure
from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler
from src.tavs_v2.algo2_esp_projection import EphemeralStructuredProjection
from src.tavs_v2.algo3_bvd_aggregation import (
    BlockVarianceDetector,
    UnifiedBayesianAggregator,
)

ClientUpdate = Dict[str, torch.Tensor]
Fixture = Dict[str, ClientUpdate]


# --------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------


def resolve_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued work finishes; without this, GPU timings are fiction."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_reps(
    fn: Callable[[int], None],
    *,
    warmup: int,
    reps: int,
    device: torch.device,
) -> List[float]:
    """Return one wall-clock sample per repetition, in milliseconds."""
    for i in range(warmup):
        fn(-(i + 1))
    synchronize(device)

    samples = []
    for rep in range(reps):
        synchronize(device)
        start = time.perf_counter()
        fn(rep)
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def bootstrap_median_ci(
    samples: List[float], resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the median. Robust to the long right tail
    that scheduler jitter puts on every wall-clock sample."""
    if len(samples) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples)))
        for _ in range(resamples)
    )
    lo = medians[int(alpha / 2 * resamples)]
    hi = medians[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


@dataclass
class Measurement:
    name: str
    samples: List[float] = field(repr=False, default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples)

    @property
    def ci95(self) -> Tuple[float, float]:
        return bootstrap_median_ci(self.samples)

    def as_dict(self) -> Dict:
        lo, hi = self.ci95
        return {
            "name": self.name,
            "median_ms": self.median_ms,
            "ci95_low_ms": lo,
            "ci95_high_ms": hi,
            "n": len(self.samples),
            "min_ms": min(self.samples),
            "max_ms": max(self.samples),
        }


# --------------------------------------------------------------------------
# Fixtures — built once, outside every timer
# --------------------------------------------------------------------------


def build_fixture(
    blocks: Dict[str, int],
    num_clients: int,
    device: torch.device,
    byzantine_fraction: float,
    seed: int,
) -> Tuple[Fixture, Dict[str, int], List[str]]:
    """Pre-materialise client updates. Byzantine clients get a scaled update so
    the detector has something real to separate; honest clients are drawn from a
    common mean plus noise, mimicking post-local-training agreement."""
    gen = torch.Generator().manual_seed(seed)
    consensus = {
        name: torch.randn(dim, generator=gen) * 0.01 for name, dim in blocks.items()
    }

    num_byz = int(round(num_clients * byzantine_fraction))
    byzantine = {f"client_{i}" for i in range(num_byz)}

    fixture: Fixture = {}
    sample_counts: Dict[str, int] = {}
    for i in range(num_clients):
        cid = f"client_{i}"
        scale = 12.0 if cid in byzantine else 1.0
        fixture[cid] = {
            name: ((consensus[name] + torch.randn(dim, generator=gen) * 0.002) * scale).to(
                device
            )
            for name, dim in blocks.items()
        }
        sample_counts[cid] = 100
    return fixture, sample_counts, sorted(byzantine)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def warm_trust_state(
    scheduler: TavsScheduler,
    all_clients: List[str],
    byzantine: List[str],
    rounds: int,
) -> None:
    """Drive the trust EMA forward `rounds` rounds before measuring.

    TAVS only diverges from full verification once honest clients have climbed
    past theta_high and accumulated a k_trust streak. Benchmarking at round 1 --
    where every client sits at the 0.25 cold-start trust and lands in Tier 1 --
    measures a configuration TAVS is never in during steady state.
    """
    byz = set(byzantine)
    for r in range(rounds):
        verified, promoted, _ = scheduler.schedule_verifications(all_clients, r)
        for cid in verified:
            scheduler.update_trust(cid, 0.2 if cid in byz else 0.98, was_verified=True)
        for cid in promoted:
            scheduler.update_trust(cid, 0.0, was_verified=False)


def fedavg_round(fixture: Fixture, sample_counts: Dict[str, int]) -> ClientUpdate:
    """Baseline: sample-weighted mean over full-dimensional updates."""
    total = float(sum(sample_counts[cid] for cid in fixture))
    out: ClientUpdate = {}
    for cid, update in fixture.items():
        w = sample_counts[cid] / total
        for name, tensor in update.items():
            if name in out:
                out[name] = out[name] + tensor * w
            else:
                out[name] = tensor * w
    return out


def defended_round(
    fixture: Fixture,
    verified: set,
    promoted: set,
    projector: EphemeralStructuredProjection,
    detector: BlockVarianceDetector,
    scheduler: TavsScheduler,
    round_num: int,
) -> Tuple[ClientUpdate, Dict[str, float]]:
    """Project the verified set, detect outliers, aggregate verified + promoted."""
    projected = {
        cid: projector.project_client_update(fixture[cid], round_num) for cid in verified
    }
    inliers, _outliers, behavior = detector.detect_outliers(projected, verified)

    verified_updates = {cid: fixture[cid] for cid in inliers}
    verified_weights = {cid: 1.0 for cid in inliers}

    promoted_updates = {cid: fixture[cid] for cid in promoted}
    promoted_weights = {
        cid: scheduler.bayesian_posterior_weight(
            scheduler.get_effective_trust(cid, round_num)
        )
        for cid in promoted
    }

    aggregate = UnifiedBayesianAggregator.aggregate(
        verified_updates, verified_weights, promoted_updates, promoted_weights
    )
    return aggregate, behavior


# --------------------------------------------------------------------------
# Client-training reference cost
# --------------------------------------------------------------------------


def measure_local_training_ms(
    device: torch.device,
    samples_per_client: int,
    batch_size: int,
    local_epochs: int,
    timed_batches: int = 8,
) -> Tuple[float, float]:
    """Time real forward+backward+step on CIFAR-shaped batches, then extrapolate
    to a full local epoch. Returns (per_batch_ms, per_client_round_ms)."""
    model = CIFARCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    def step(_rep: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optimizer.step()

    samples = time_reps(step, warmup=3, reps=timed_batches, device=device)
    per_batch = statistics.median(samples)

    batches_per_epoch = max(1, samples_per_client // batch_size)
    per_client_round = per_batch * batches_per_epoch * local_epochs
    return per_batch, per_client_round


# --------------------------------------------------------------------------
# Benchmark driver
# --------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> Dict:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    structure = ModelStructure.from_model(CIFARCNN())
    blocks = {b["name"]: b["num_params"] for b in structure.blocks}
    total_params = structure.total_params

    fixture, sample_counts, byzantine = build_fixture(
        blocks, args.num_clients, device, args.byzantine_fraction, args.seed
    )
    all_clients = sorted(fixture.keys())

    # --- Scheduling decision is deterministic given trust state; measure the
    # --- verification *count* separately from the wall-clock arms.
    scheduler = TavsScheduler(
        gamma_budget=0.35,
        theta_low=0.3,
        theta_high=0.8,
        alpha_trust=0.9,
        tau_ramp=30.0,
        k_trust=3,
    )
    warm_trust_state(scheduler, all_clients, byzantine, args.warm_rounds)
    measure_round = max(args.round_num, args.warm_rounds)
    verified, promoted, _ = scheduler.schedule_verifications(all_clients, measure_round)
    args.round_num = measure_round

    projector = EphemeralStructuredProjection(target_k=args.target_k, model_blocks=blocks)
    detector = BlockVarianceDetector(tau_z=args.tau_z)

    # Warm the detector's variance state so the bootstrap path is not timed.
    warm_projected = {
        cid: projector.project_client_update(fixture[cid], 0) for cid in verified
    }
    detector.detect_outliers(warm_projected, set(verified))

    measurements: Dict[str, Measurement] = {}

    # --- Arm 1: FedAvg baseline -------------------------------------------
    def arm_fedavg(_rep: int) -> None:
        fedavg_round(fixture, sample_counts)

    # --- Arm 2: full verification (every client, every round) --------------
    full_scheduler = TavsScheduler(
        gamma_budget=1.0,
        theta_low=1.0,  # everyone lands in Tier 1 -> always verified
        theta_high=1.0,
        alpha_trust=0.9,
        tau_ramp=30.0,
        k_trust=3,
    )
    full_scheduler.schedule_verifications(all_clients, args.round_num)

    # Each arm needs its own round stream in fresh-projection mode, otherwise the
    # second arm to run in a rep reuses the first arm's cached JL matrices and
    # its generation cost silently vanishes.
    def round_for(rep: int, arm_index: int) -> int:
        if not args.fresh_projection:
            return args.round_num
        return args.round_num + rep * 8 + arm_index

    def arm_full_verify(rep: int) -> None:
        defended_round(
            fixture,
            set(all_clients),
            set(),
            projector,
            detector,
            full_scheduler,
            round_for(rep, 1),
        )

    # --- Arm 3: TAVS-ESP ---------------------------------------------------
    def arm_tavs(rep: int) -> None:
        defended_round(
            fixture,
            set(verified),
            set(promoted),
            projector,
            detector,
            scheduler,
            round_for(rep, 2),
        )

    arms = [
        ("fedavg", arm_fedavg),
        ("full_verify", arm_full_verify),
        ("tavs_esp", arm_tavs),
    ]
    for name, _fn in arms:
        measurements[name] = Measurement(name)

    # Interleave arms so any drift in machine speed hits all arms equally.
    for name, fn in arms:
        for _ in range(args.warmup):
            fn(-1)
    synchronize(device)

    for rep in range(args.reps):
        for name, fn in arms:
            synchronize(device)
            start = time.perf_counter()
            fn(rep)
            synchronize(device)
            measurements[name].samples.append((time.perf_counter() - start) * 1000.0)

    # --- Stage breakdown for the TAVS arm ---------------------------------
    def stage_project(rep: int) -> None:
        for cid in verified:
            projector.project_client_update(fixture[cid], round_for(rep, 3))

    projected_cache = {
        cid: projector.project_client_update(fixture[cid], args.round_num)
        for cid in verified
    }

    def stage_detect(_rep: int) -> None:
        detector.detect_outliers(projected_cache, set(verified))

    def stage_schedule(rep: int) -> None:
        probe = TavsScheduler(
            gamma_budget=0.35,
            theta_low=0.3,
            theta_high=0.8,
            alpha_trust=0.9,
            tau_ramp=30.0,
            k_trust=3,
        )
        probe.schedule_verifications(all_clients, args.round_num + rep)

    stages = {
        "stage_schedule": stage_schedule,
        "stage_project": stage_project,
        "stage_detect": stage_detect,
    }
    for name, fn in stages.items():
        measurements[name] = Measurement(
            name, time_reps(fn, warmup=args.warmup, reps=args.reps, device=device)
        )

    # --- Client-training reference ----------------------------------------
    per_batch_ms, per_client_round_ms = measure_local_training_ms(
        device,
        samples_per_client=args.samples_per_client,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )

    fedavg_ms = measurements["fedavg"].median_ms
    tavs_ms = measurements["tavs_esp"].median_ms
    full_ms = measurements["full_verify"].median_ms

    # Clients train in parallel in a real deployment, so the round's client
    # phase costs one client's time, not the sum.
    round_client_ms = per_client_round_ms

    report = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "config": {
            "num_clients": args.num_clients,
            "byzantine_fraction": args.byzantine_fraction,
            "byzantine_clients": byzantine,
            "model_parameters": total_params,
            "num_blocks": len(blocks),
            "target_k": args.target_k,
            "tau_z": args.tau_z,
            "reps": args.reps,
            "warmup": args.warmup,
            "fresh_projection_per_rep": args.fresh_projection,
            "seed": args.seed,
            "warm_rounds": args.warm_rounds,
            "measured_at_round": measure_round,
        },
        "scheduling": {
            "trust_scores": {c: scheduler.trust_scores[c] for c in all_clients},
            "verified": sorted(verified),
            "promoted": sorted(promoted),
            "num_verified": len(verified),
            "num_promoted": len(promoted),
            "verification_reduction_vs_full": 1.0 - len(verified) / len(all_clients),
        },
        "measurements": {k: m.as_dict() for k, m in measurements.items()},
        "client_reference": {
            "per_batch_ms": per_batch_ms,
            "per_client_round_ms": per_client_round_ms,
            "samples_per_client": args.samples_per_client,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
        },
        "analysis": {
            "tavs_minus_fedavg_ms": tavs_ms - fedavg_ms,
            "full_minus_fedavg_ms": full_ms - fedavg_ms,
            "tavs_speedup_vs_full": full_ms / tavs_ms if tavs_ms > 0 else float("nan"),
            "server_overhead_pct_of_round_tavs": (tavs_ms - fedavg_ms)
            / (round_client_ms + fedavg_ms)
            * 100.0,
            "server_overhead_pct_of_round_full": (full_ms - fedavg_ms)
            / (round_client_ms + fedavg_ms)
            * 100.0,
        },
    }
    return report


def print_report(report: Dict) -> None:
    env, cfg = report["environment"], report["config"]
    sched, meas = report["scheduling"], report["measurements"]
    ref, ana = report["client_reference"], report["analysis"]

    print("=" * 74)
    print("TAVS-ESP SERVER-SIDE WALL-CLOCK BENCHMARK")
    print("=" * 74)
    print(f"device={env['device']}  torch={env['torch']}  threads={env['threads']}")
    print(
        f"model={cfg['model_parameters']:,} params in {cfg['num_blocks']} blocks   "
        f"target_k={cfg['target_k']}"
    )
    print(
        f"clients={cfg['num_clients']}  byzantine={cfg['byzantine_fraction']:.0%}  "
        f"reps={cfg['reps']} (warmup={cfg['warmup']})"
    )
    print()

    print("-- Scheduling decision ----------------------------------------------")
    print(
        f"verified={sched['num_verified']}  promoted={sched['num_promoted']}  "
        f"reduction vs full verification={sched['verification_reduction_vs_full']:.0%}"
    )
    print()

    print("-- Per-round server cost (median [95% CI]) --------------------------")
    for key in ("fedavg", "full_verify", "tavs_esp"):
        m = meas[key]
        print(
            f"  {key:<14} {m['median_ms']:9.3f} ms  "
            f"[{m['ci95_low_ms']:.3f}, {m['ci95_high_ms']:.3f}]"
        )
    print()

    print("-- TAVS stage breakdown ---------------------------------------------")
    for key in ("stage_schedule", "stage_project", "stage_detect"):
        m = meas[key]
        print(
            f"  {key:<14} {m['median_ms']:9.3f} ms  "
            f"[{m['ci95_low_ms']:.3f}, {m['ci95_high_ms']:.3f}]"
        )
    print()

    print("-- Client-training reference (measured, extrapolated to a round) -----")
    print(f"  per batch          {ref['per_batch_ms']:9.3f} ms")
    print(
        f"  per client/round   {ref['per_client_round_ms']:9.1f} ms   "
        f"({ref['samples_per_client']} samples, bs={ref['batch_size']}, "
        f"{ref['local_epochs']} epoch(s))"
    )
    print()

    print("-- Analysis ----------------------------------------------------------")
    print(f"  TAVS defence cost over FedAvg      {ana['tavs_minus_fedavg_ms']:9.3f} ms")
    print(f"  Full-verify cost over FedAvg       {ana['full_minus_fedavg_ms']:9.3f} ms")
    print(f"  TAVS speedup vs full verification  {ana['tavs_speedup_vs_full']:9.2f}x")
    print(
        f"  Defence as % of real round (TAVS)  "
        f"{ana['server_overhead_pct_of_round_tavs']:9.3f} %"
    )
    print(
        f"  Defence as % of real round (full)  "
        f"{ana['server_overhead_pct_of_round_full']:9.3f} %"
    )
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--byzantine-fraction", type=float, default=0.25)
    parser.add_argument("--target-k", type=int, default=150)
    parser.add_argument("--tau-z", type=float, default=2.0)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--round-num", type=int, default=1)
    parser.add_argument(
        "--warm-rounds",
        type=int,
        default=60,
        help="Rounds of trust dynamics to run before measuring, so the scheduler "
        "is benchmarked in steady state rather than at cold start",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    parser.add_argument("--samples-per-client", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--fresh-projection",
        action="store_true",
        help="Advance the round each rep so JL matrix generation is timed too "
        "(this is the honest per-round cost; without it you measure the cached path)",
    )
    parser.add_argument("--output", default="", help="Write the JSON report here")
    args = parser.parse_args()

    report = run_benchmark(args)
    print_report(report)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {path}")


if __name__ == "__main__":
    main()
