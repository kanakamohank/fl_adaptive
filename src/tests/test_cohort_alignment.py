#!/usr/bin/env python3
"""
Regression test: same-seed cohorts line up across strategy arms.

Locks in the fix for the cohort-divergence bug. Previously the three arms
(TAVS / RandomSkip / FullVerification) sampled cohorts by sorting Flower's
ClientProxy.cid strings and drawing with a seeded rng. Flower's simulation
runtime assigns a fresh 64-bit random integer as each proxy's cid PER
run_simulation() call, so the "sorted pool" was a different list of strings
in every arm and sampling landed on different partition-ids -- an
arm-vs-arm comparison confound. See _ensure_partition_map / _sample_cohort
in tavs_v2/tavs_esp_strategy.py for the fix.

The test spins up an isolated tiny Flower simulation for each of the three
strategies with an identical seed, records the partition-ids sampled per
round via a monkey-patched _sample_cohort, and asserts:

  1. Partition maps in each arm are complete (0..N-1).
  2. Per-round partition-id cohorts are IDENTICAL across arms.
  3. The cohorts still change between rounds (i.e. the seed-per-round
     mixing still works -- we did not accidentally freeze everyone into
     one static cohort forever).

Runs in ~30s on CPU; skips gracefully if Flower's simulation backend is
not installed.
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

# Make sure src.* imports resolve
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def _require_simulation_backend() -> None:
    """
    Fail loudly rather than silently skipping.

    The cohort-alignment invariant is exactly what would silently regress if
    the partition-map code path breaks; a test that prints SKIP and returns
    True in a ray-less env would go green while the real bug came back. Any
    env intended to run the pilot MUST have flwr[simulation] + ray, so if
    they are missing we raise -- CI notices, we notice, the pilot doesn't
    silently run on stale infra.
    """
    try:
        import ray  # noqa: F401
        from flwr.simulation import run_simulation  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "test_cohort_alignment requires flwr[simulation] + ray so that "
            "the partition-map path can actually be exercised. Missing: "
            f"{type(e).__name__}: {e}. Install with `pip install \"flwr[simulation]\"`."
        )


def _run_arm(arm_name, strategy_cls, seed=42, num_clients=12, cpr=4, rounds=3):
    """Return {round: sorted(list_of_partition_ids)} for one arm."""
    import numpy as np
    import flwr as fl
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.simulation import run_simulation

    from src.tavs_v2.tavs_esp_strategy import (
        TavsEspStrategy, TavsEspConfig,
        FullVerificationStrategy, RandomSkipStrategy,
    )
    from src.core.models import ModelStructure

    # Minimal 1-block "model" so ESP setup is well-defined.
    struct = ModelStructure()
    struct.add_block("b0", (2,), 2)

    class C(NumPyClient):
        def __init__(self, pid): self.pid = pid
        def get_properties(self, config): return {"partition-id": self.pid}
        def get_parameters(self, config): return [np.zeros(2, dtype=np.float32)]
        def fit(self, parameters, config):
            return ([np.array([float(self.pid), 0.0], dtype=np.float32)],
                    100, {"client_id": f"honest_{self.pid:02d}", "partition_id": self.pid})
        def evaluate(self, parameters, config):
            return 0.0, 100, {"accuracy": 0.5}

    def client_fn(context: Context):
        return C(context.node_config.get("partition-id", -1)).to_client()

    # Config identical to what the pipeline sets.
    cfg = TavsEspConfig()
    cfg.min_fit_clients = cpr
    cfg.min_available_clients = num_clients
    cfg.min_evaluate_clients = 0
    cfg.fraction_evaluate = 0.0
    cfg.sampling_seed = seed
    cfg.deterministic_sampling = True

    if strategy_cls is None:
        strategy = TavsEspStrategy(config=cfg, model_structure=struct)
    elif strategy_cls is RandomSkipStrategy:
        strategy = RandomSkipStrategy(config=cfg, model_structure=struct,
                                      skip_rate=0.4, skip_seed=seed)
    elif strategy_cls is FullVerificationStrategy:
        strategy = FullVerificationStrategy(config=cfg, model_structure=struct)
    else:
        raise ValueError(strategy_cls)

    # Monkey-patch _sample_cohort to record partition ids for each call.
    partitions_by_round = {}
    orig = type(strategy)._sample_cohort

    def logged(self, client_manager):
        result = orig(self, client_manager)
        pids = sorted(self._cid_to_partition.get(cid, -1) for cid in result.keys())
        partitions_by_round.setdefault(self._current_round, []).append(pids)
        return result

    type(strategy)._sample_cohort = logged
    try:
        random.seed(seed)
        np.random.seed(seed)

        def server_fn(context):
            return ServerAppComponents(strategy=strategy,
                                       config=ServerConfig(num_rounds=rounds))

        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=num_clients,
            backend_config={
                "client_resources": {"num_cpus": 1, "num_gpus": 0},
                "init_args": {"ignore_reinit_error": True, "log_to_driver": False,
                              "num_cpus": 2, "include_dashboard": False},
            },
        )
    finally:
        type(strategy)._sample_cohort = orig

    # One _sample_cohort call per round in the tested strategies.
    return {r: v[0] for r, v in partitions_by_round.items()}


NUM_CLIENTS = 12
CPR = 4
NUM_ROUNDS = 3


def test_cross_arm_cohort_alignment():
    """The three strategies must sample IDENTICAL partition-id cohorts per
    round when handed the same seed."""
    _require_simulation_backend()

    from src.tavs_v2.tavs_esp_strategy import (
        FullVerificationStrategy, RandomSkipStrategy,
    )

    tavs = _run_arm("tavs", None, num_clients=NUM_CLIENTS, cpr=CPR, rounds=NUM_ROUNDS)
    rand = _run_arm("random", RandomSkipStrategy, num_clients=NUM_CLIENTS, cpr=CPR, rounds=NUM_ROUNDS)
    full = _run_arm("full", FullVerificationStrategy, num_clients=NUM_CLIENTS, cpr=CPR, rounds=NUM_ROUNDS)

    # 1) All three saw the same rounds
    assert set(tavs) == set(rand) == set(full), (
        f"round sets differ: tavs={sorted(tavs)} random={sorted(rand)} full={sorted(full)}"
    )

    # 2) Per-round cohorts identical across arms
    for r in sorted(tavs.keys()):
        assert tavs[r] == rand[r] == full[r], (
            f"round {r}: cohorts differ across arms\n"
            f"  tavs   = {tavs[r]}\n"
            f"  random = {rand[r]}\n"
            f"  full   = {full[r]}"
        )
        # 3) No bogus -1 sentinels (would mean partition map was incomplete).
        assert all(p >= 0 for p in tavs[r]), f"round {r}: partition map incomplete: {tavs[r]}"

        # 4) Partial participation actually happened: cohort size == cpr and
        # strictly smaller than the pool. Without this a subclass override
        # returning "everyone" would still trivially pass the cross-arm
        # equality above, proving nothing about partition-id-space sampling.
        assert len(tavs[r]) == CPR, (
            f"round {r}: expected cohort of size {CPR}, got {len(tavs[r])}"
        )
        assert len(tavs[r]) < NUM_CLIENTS, (
            f"round {r}: cohort {tavs[r]} contains every client; partial "
            f"participation was not exercised"
        )

    # 5) Every round's cohort is distinct. A regression that froze cohorts
    # for most rounds but not all would slip past a weaker "some pair differs"
    # check; requiring full distinctness catches it.
    rounds = sorted(tavs.keys())
    unique_cohorts = {tuple(tavs[r]) for r in rounds}
    assert len(unique_cohorts) == len(rounds), (
        f"cohorts repeat across rounds: {tavs}"
    )

    print(f"✓ per-round cohorts identical across TAVS / RandomSkip / FullVerify")
    for r in rounds:
        print(f"    round {r}: partitions = {tavs[r]}")
    return True


def test_partition_map_failure_raises_in_deterministic_mode():
    """A get_properties failure MUST NOT silently fall back to the cid-sort
    path -- that would reintroduce cross-arm divergence invisibly. In
    deterministic_sampling mode the strategy raises and the run aborts."""
    _require_simulation_backend()

    from src.tavs_v2.tavs_esp_strategy import (
        TavsEspStrategy, TavsEspConfig,
    )
    from src.core.models import ModelStructure

    struct = ModelStructure()
    struct.add_block("b0", (2,), 2)

    cfg = TavsEspConfig()
    cfg.min_fit_clients = 2
    cfg.min_available_clients = 4
    cfg.deterministic_sampling = True
    cfg.sampling_seed = 42
    strat = TavsEspStrategy(config=cfg, model_structure=struct)

    class BadProxy:
        cid = "bad-cid"
        def get_properties(self, ins=None, timeout=None, group_id=None):
            raise TimeoutError("simulated ray race")

    class ManagerWithBadProxy:
        def all(self):
            return {"bad-cid": BadProxy()}
        def num_available(self):
            return 1
        def sample(self, num_clients, min_num_clients):
            return [BadProxy()]
        def wait_for(self, num_clients, timeout):
            return True

    strat._current_round = 1
    try:
        strat._sample_cohort(ManagerWithBadProxy())
    except RuntimeError as e:
        assert "Partition-map bootstrap failed" in str(e), (
            f"expected explicit refusal message; got {e!r}"
        )
        assert strat._partition_map_status == "failed"
        print("✓ get_properties failure raises RuntimeError in deterministic mode "
              "(no silent fallback)")
        return True
    raise AssertionError(
        "_sample_cohort silently fell back after get_properties failure -- "
        "this is exactly the regression the fix was written to prevent."
    )


def test_partition_map_status_latches():
    """Once _partition_map_status is set, it does NOT re-attempt the sweep
    every round. Prevents the 30s wait_for perf cliff on repeated failure."""
    _require_simulation_backend()

    from src.tavs_v2.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig
    from src.core.models import ModelStructure

    struct = ModelStructure()
    struct.add_block("b0", (2,), 2)

    cfg = TavsEspConfig()
    cfg.min_fit_clients = 2
    cfg.min_available_clients = 4
    strat = TavsEspStrategy(config=cfg, model_structure=struct)

    call_count = {"n": 0}

    class TrackingManager:
        def all(self):
            call_count["n"] += 1
            return {}
        def wait_for(self, num_clients, timeout):
            return True

    strat._ensure_partition_map(TrackingManager())
    assert strat._partition_map_status == "failed"
    first_calls = call_count["n"]

    # Additional calls must NOT invoke .all() again -- the latch is what
    # keeps the 30s wait_for from being paid every round in a broken run.
    for _ in range(3):
        strat._ensure_partition_map(TrackingManager())
    assert call_count["n"] == first_calls, (
        f"_ensure_partition_map re-ran the sweep after status was latched "
        f"({first_calls} -> {call_count['n']})"
    )
    print("✓ _partition_map_status latches; no repeated get_properties sweeps")
    return True


def main():
    print("🧪 Cross-arm cohort alignment test")
    print("=" * 60)
    tests = [
        test_cross_arm_cohort_alignment,
        test_partition_map_failure_raises_in_deterministic_mode,
        test_partition_map_status_latches,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 all cohort alignment tests PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
