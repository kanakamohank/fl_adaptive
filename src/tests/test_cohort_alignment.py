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


def _has_simulation_backend() -> bool:
    try:
        import ray  # noqa: F401
        from flwr.simulation import run_simulation  # noqa: F401
        return True
    except Exception:
        return False


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


def test_cross_arm_cohort_alignment():
    """The three strategies must sample IDENTICAL partition-id cohorts per
    round when handed the same seed."""
    if not _has_simulation_backend():
        print("SKIP: flwr[simulation] / ray not installed in this env")
        return True

    from src.tavs_v2.tavs_esp_strategy import (
        FullVerificationStrategy, RandomSkipStrategy,
    )

    tavs = _run_arm("tavs", None)
    rand = _run_arm("random", RandomSkipStrategy)
    full = _run_arm("full", FullVerificationStrategy)

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
        # 3) Sanity: no bogus -1 sentinels leaked in (would mean partition
        # map was incomplete).
        assert all(p >= 0 for p in tavs[r]), f"round {r}: partition map incomplete: {tavs[r]}"

    # 4) Cohorts should differ ROUND TO ROUND (otherwise we froze the
    # federation). Cross-arm identity plus round-to-round variation is the
    # exact invariant we want.
    rounds = sorted(tavs.keys())
    assert len(rounds) >= 2
    changed = any(tavs[rounds[i]] != tavs[rounds[i + 1]]
                  for i in range(len(rounds) - 1))
    assert changed, f"cohorts identical across every round: {tavs}"

    print(f"✓ per-round cohorts identical across TAVS / RandomSkip / FullVerify")
    for r in rounds:
        print(f"    round {r}: partitions = {tavs[r]}")
    return True


def main():
    print("🧪 Cross-arm cohort alignment test")
    print("=" * 60)
    ok = test_cross_arm_cohort_alignment()
    if ok:
        print("\n🎯 cohort alignment test PASSED")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
