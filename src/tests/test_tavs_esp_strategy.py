#!/usr/bin/env python3
"""
Test V2 TAVS-ESP Strategy Implementation (The Bridge)

Tests the complete federated learning strategy including:
1. Flower integration (configure_fit, aggregate_fit)
2. Delegation to V2 Layer 1 (TAVS) + Layer 2 (ESP) coordination
3. Trust dynamics update verification
4. End-to-end federated learning simulation loop
"""

import sys
import numpy as np
from typing import List, Dict
from unittest.mock import MagicMock
import io


# Mock Flower imports for testing
class MockClientProxy:
    def __init__(self, cid: str):
        self.cid = cid

class MockFitRes:
    def __init__(self, parameters, metrics: Dict):
        self.parameters = parameters
        self.metrics = metrics
        self.num_examples = 100

class MockEvaluateRes:
    def __init__(self, loss: float, num_examples: int, metrics: Dict):
        self.loss = loss
        self.num_examples = num_examples
        self.metrics = metrics

class MockParameters:
    def __init__(self, tensors: List[np.ndarray]):
        byte_tensors = []
        for t in tensors:
            # Serialize ndarrays into a valid .npy byte stream
            if isinstance(t, np.ndarray):
                b = io.BytesIO()
                np.save(b, t, allow_pickle=False)
                byte_tensors.append(b.getvalue())
            else:
                byte_tensors.append(t)
        self.tensors = byte_tensors

# Mock Flower functions
def mock_parameters_to_ndarrays(params):
    if hasattr(params, 'tensors'):
        res = []
        for t in params.tensors:
            # Deserialize .npy bytes back to ndarray
            if isinstance(t, bytes):
                res.append(np.load(io.BytesIO(t), allow_pickle=False))
            else:
                res.append(t)
        return res
    return [np.random.randn(150000)] # Match the default 'full_model' fallback dim

class MockClientManager:
    def __init__(self, num_clients: int):
        self.clients = [MockClientProxy(f"client_{i}") for i in range(num_clients)]

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int):
        return self.clients[:min(num_clients, len(self.clients))]

    def all(self) -> dict:
        """Return all available clients (required by Flower API)."""
        return {c.cid: c for c in self.clients}

def mock_ndarrays_to_parameters(arrays):
    return MockParameters(arrays)

# Create a mock Strategy base class
class MockStrategy:
    def __init__(self):
        pass

# Mock Flower classes
class MockFitIns:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config

class MockEvaluateIns:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config

# Monkey patch for testing
sys.modules['flwr'] = MagicMock()
sys.modules['flwr.server'] = MagicMock()
sys.modules['flwr.server.strategy'] = MagicMock()
sys.modules['flwr.server.strategy'].Strategy = MockStrategy
sys.modules['flwr.server.client_proxy'] = MagicMock()
sys.modules['flwr.server.client_manager'] = MagicMock()

class MockCommon:
    def __init__(self):
        self.parameters_to_ndarrays = mock_parameters_to_ndarrays
        self.ndarrays_to_parameters = mock_ndarrays_to_parameters
        self.FitIns = MockFitIns
        self.EvaluateIns = MockEvaluateIns
        self.FitRes = MockFitRes
        self.EvaluateRes = MockEvaluateRes
        self.Parameters = MockParameters
        self.Scalar = float  
        self.NDArrays = list  

sys.modules['flwr.common'] = MockCommon()

# Import the actual Strategy AFTER patching flwr
from src.tavs_v2 import TavsEspStrategy

# Dummy Config to mimic PipelineConfig
class DummyConfig:
    def __init__(self):
        self.theta_low = 0.3
        self.theta_high = 0.7
        self.target_k = 150
        self.gamma_budget = 0.35
        self.alpha_trust = 0.9
        self.tau_ramp = 30.0
        self.k_trust = 10
        self.p_decoy = 0.15
        self.detection_threshold = 5.0
        self.master_key = b'test_bridge_key'

def test_tavs_esp_strategy_initialization():
    """Test V2 TAVS-ESP bridge initialization."""
    print("Testing V2 TAVS-ESP Strategy initialization...")

    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)

    # Test 1: Component integration
    assert strategy.scheduler is not None, "V2 Scheduler not initialized"
    assert strategy.projector is not None, "V2 Projector not initialized"
    assert strategy.detector is not None, "V2 Detector not initialized"
    
    # Test 2: Fallback parameters
    assert "full_model" in strategy.model_blocks, "Model blocks fallback failed"
    print("✓ Strategy initialization successful and V2 components linked.")
    return True

def test_configure_fit_scheduling():
    """Test TAVS Layer 1 scheduling bridge in configure_fit."""
    print("\nTesting TAVS Layer 1 scheduling bridge...")

    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)
    client_manager = MockClientManager(num_clients=8)
    initial_params = MockParameters([np.random.randn(150000)])

    # Run scheduling
    fit_configs = strategy.configure_fit(
        server_round=1,
        parameters=initial_params,
        client_manager=client_manager
    )

    assert len(fit_configs) == 8, "Did not configure all available clients"
    
    verified_count = 0
    promoted_count = 0

    for i, (proxy, fit_ins) in enumerate(fit_configs):
        config_dict = fit_ins.config
        assert "is_verified" in config_dict, "Missing V2 is_verified flag"
        if config_dict["is_verified"]:
            verified_count += 1
        else:
            promoted_count += 1

    print(f"✓ V2 Assignments: {verified_count} verified, {promoted_count} promoted")
    return True

def test_aggregate_fit_esp_layer():
    """Test ESP Layer 2 processing and Unified Aggregation bridge."""
    print("\nTesting ESP Layer 2 aggregation bridge...")

    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)
    client_proxies = [MockClientProxy(f"client_{i}") for i in range(6)]
    
    # Bypass Mech 3 so they aren't stuck in Tier 1 for the test
    for c in client_proxies:
        strategy.scheduler.join_rounds[c.cid] = -100

    client_results = []
    for i, proxy in enumerate(client_proxies):
        # Clients return ndarrays; mock_parameters_to_ndarrays will unpack them
        # Let's make client_0 and client_1 massive outliers to test the BVD bridge
        if i < 2:  
            client_params = [np.random.randn(150000) * 50.0]
        else:
            client_params = [np.random.randn(150000) * 0.1]

        fit_res = MockFitRes(
            parameters=MockParameters(client_params),
            metrics={"is_verified": True} # Assume all were verified for this test
        )
        client_results.append((proxy, fit_res))

    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=client_results,
        failures=[]
    )

    assert aggregated_params is not None
    assert "inliers" in metrics, "Missing inliers count from V2 bridge"
    assert "outliers" in metrics, "Missing outliers count from V2 bridge"

    print(f"✓ Aggregation successful: {metrics['inliers']} Inliers, {metrics['outliers']} Outliers")
    return True

def test_trust_dynamics_integration():
    """Test trust score evolution through the strategy bridge."""
    print("\nTesting trust dynamics integration...")

    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)
    client_manager = MockClientManager(num_clients=8)
    initial_params = MockParameters([np.random.randn(150000)])

    # Simulate 3 rounds
    for round_num in range(1, 4):
        fit_configs = strategy.configure_fit(round_num, initial_params, client_manager)

        client_results = []
        for i, (proxy, fit_ins) in enumerate(fit_configs):
            is_verified = fit_ins.config.get("is_verified", True)
            
            # Inject noise for client_0 to test penalization
            noise = 50.0 if proxy.cid == "client_0" else 0.1
            
            fit_res = MockFitRes(
                parameters=MockParameters([np.random.randn(150000) * noise]),
                metrics={"is_verified": is_verified}
            )
            client_results.append((proxy, fit_res))

        strategy.aggregate_fit(round_num, client_results, [])

    # Check trust state via the V2 Scheduler
    trust_scores = strategy.scheduler.trust_scores
    
    assert trust_scores["client_0"] < 0.5, "Attacker was not penalized"
    assert trust_scores["client_1"] > trust_scores["client_0"], "Honest client trust should exceed attacker trust"
    
    print("✓ Trust dynamically tracks via the V2 scheduler.")
    return True

def test_end_to_end_fl_simulation():
    """Test complete federated learning simulation loop."""
    print("\nTesting end-to-end FL simulation loop...")

    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)
    client_manager = MockClientManager(num_clients=5)
    initial_params = MockParameters([np.random.randn(150000)])

    # Run FL simulation for 3 rounds
    for round_num in range(1, 4):
        fit_configs = strategy.configure_fit(round_num, initial_params, client_manager)

        client_results = []
        for i, (proxy, fit_ins) in enumerate(fit_configs):
            client_params = [np.random.randn(150000) * 0.05]
            fit_res = MockFitRes(
                parameters=MockParameters(client_params),
                metrics={"is_verified": fit_ins.config["is_verified"]}
            )
            client_results.append((proxy, fit_res))

        aggregated_params, metrics = strategy.aggregate_fit(round_num, client_results, [])
        initial_params = aggregated_params
        
        print(f"  Round {round_num}: Inliers={metrics.get('inliers')}, Outliers={metrics.get('outliers')}")

    assert len(strategy.scheduler.trust_scores) == 5, "Not all clients tracked"
    print("✓ End-to-end FL simulation bridge successful")
    return True

# ---------------------------------------------------------------------------
# RandomSkipStrategy tests
#
# RandomSkipStrategy is the fair baseline for TAVS: same aggregation pipeline,
# skip decisions replaced by a per-client coin. These tests fence off the ways
# it could silently stop being that baseline: the coin biased, the coin
# non-reproducible, promoted clients bypassing the pipeline safety net.
# ---------------------------------------------------------------------------
from src.tavs_v2.tavs_esp_strategy import RandomSkipStrategy


def test_random_skip_rate_validation():
    """Constructor rejects rates that would degenerate the baseline.

    skip_rate = 1.0 verifies no client, starves cosine/clip of a reference
    cohort, and quietly turns the arm into "unbounded promoted updates". A
    negative rate is a caller bug. Both are refused up-front rather than
    caught mid-round.
    """
    print("\nTesting RandomSkip rate validation...")
    config = DummyConfig()
    RandomSkipStrategy(config=config, skip_rate=0.0)   # boundary: no skip
    RandomSkipStrategy(config=config, skip_rate=0.99)

    for bad in (1.0, -0.01, 1.5):
        try:
            RandomSkipStrategy(config=config, skip_rate=bad)
        except ValueError:
            continue
        raise AssertionError(f"skip_rate={bad} should have raised")
    print("✓ rate validation rejects [1.0, ∞) and negatives")
    return True


def test_random_skip_split_is_partition():
    """V and P cover the sampled cohort exactly and never overlap.

    aggregate_fit routes by V/P membership, so a client missing from both
    would be dropped from aggregation. A client in both would be counted
    twice. Either would silently corrupt the arm's aggregate.
    """
    print("\nTesting RandomSkip V/P partition...")
    config = DummyConfig()
    strategy = RandomSkipStrategy(config=config, skip_rate=0.5, skip_seed=42)
    client_manager = MockClientManager(num_clients=20)
    params = MockParameters([np.random.randn(150000)])

    fit_configs = strategy.configure_fit(1, params, client_manager)
    assert len(fit_configs) == 20

    assn = strategy._round_assignments[1]
    v, p = assn["verified"], assn["promoted"]
    assert v.isdisjoint(p), f"V and P overlap: {v & p}"
    cohort = {proxy.cid for proxy, _ in fit_configs}
    assert v | p == cohort, f"V ∪ P != cohort; missing {cohort - (v | p)}"
    print(f"✓ partition holds: |V|={len(v)}, |P|={len(p)}, cohort={len(cohort)}")
    return True


def test_random_skip_rate_matches_target():
    """Observed skip fraction tracks target rate at cohort-scale N.

    The pipeline is Bernoulli per client, so we expect the observed rate to
    concentrate at the target as clients accumulate. At n=200 and p=0.4 the
    sd of the fraction is sqrt(0.24/200) ≈ 0.035; we allow ±0.10 so the test
    is not flaky on any single seed but still catches a 2σ bias.
    """
    print("\nTesting RandomSkip target-rate calibration...")
    config = DummyConfig()
    strategy = RandomSkipStrategy(config=config, skip_rate=0.4, skip_seed=7)
    client_manager = MockClientManager(num_clients=200)
    params = MockParameters([np.random.randn(150000)])

    strategy.configure_fit(1, params, client_manager)
    assn = strategy._round_assignments[1]
    observed = len(assn["promoted"]) / (len(assn["verified"]) + len(assn["promoted"]))
    assert 0.30 <= observed <= 0.50, f"observed skip {observed:.3f} off target 0.40"
    print(f"✓ observed skip rate {observed:.3f} within ±0.10 of 0.40")
    return True


def test_random_skip_never_leaves_v_empty():
    """The safety guard promotes at least one client back to V.

    Without a verified cohort the cosine gate and magnitude clip have no
    reference and short-circuit with skipped_reason=no_verified_clients_*,
    which lets every promoted update through unbounded. The strategy forces
    one client back to V rather than let the arm silently stop being a
    same-pipeline comparison.
    """
    print("\nTesting RandomSkip empty-V safety guard...")
    config = DummyConfig()
    # skip_rate = 0.99 makes every-promoted the modal outcome for small cohorts.
    strategy = RandomSkipStrategy(config=config, skip_rate=0.99, skip_seed=0)
    client_manager = MockClientManager(num_clients=5)
    params = MockParameters([np.random.randn(150000)])

    for round_num in range(1, 11):
        strategy.configure_fit(round_num, params, client_manager)
        v = strategy._round_assignments[round_num]["verified"]
        assert len(v) >= 1, f"round {round_num}: V empty despite guard"
    print("✓ V never empty across 10 near-total-skip rounds")
    return True


def test_random_skip_seed_reproducibility_and_independence():
    """Same skip_seed reproduces the same split; different seeds diverge.

    The pilot binds skip_seed to the pipeline seed so that multiple runs of
    the random arm do NOT share the class default skip_seed=0 (which would
    give identical Bernoulli sequences across seeds and quietly kill half
    the intended seed variance).
    """
    print("\nTesting RandomSkip seed reproducibility & independence...")
    config = DummyConfig()
    client_manager = MockClientManager(num_clients=30)
    params = MockParameters([np.random.randn(150000)])

    def one(seed: int):
        s = RandomSkipStrategy(config=config, skip_rate=0.4, skip_seed=seed)
        s.configure_fit(1, params, client_manager)
        return s._round_assignments[1]["promoted"]

    p_a1 = one(11); p_a2 = one(11); p_b = one(12)
    assert p_a1 == p_a2, "same seed produced different splits"
    assert p_a1 != p_b, "different seeds produced identical splits (silent seed leak)"
    print(f"✓ deterministic per seed, independent across seeds "
          f"(|A|={len(p_a1)}, |B|={len(p_b)}, symdiff={len(p_a1 ^ p_b)})")
    return True


def test_disable_trust_weighted_aggregation_flag_changes_aggregate():
    """
    The `disable_trust_weighted_aggregation` config flag must actually change
    the aggregation. Without a functional check, a rename or a subtle
    condition typo could turn the ablation into a silent no-op and every
    conclusion drawn from a 4-arm pilot (which comparison arm was which)
    would be wrong in the same direction.

    We build two identical strategies -- same clients, same updates, same
    scheduler state -- and flip only the flag. If the resulting aggregate
    is byte-for-byte identical, the flag doesn't do anything. It must
    differ for the ablation to mean what the pilot script says it means.
    """
    print("\nTesting disable_trust_weighted_aggregation flag actually changes aggregate...")

    def one_run(disable_flag: bool):
        cfg = DummyConfig()
        cfg.disable_trust_weighted_aggregation = disable_flag
        # gamma_budget=0.35 in DummyConfig is fine, but the client trust
        # scores below force a mix of V and P. is_stale checks
        # last_verified_round; without one set the scheduler routes every
        # client into V as "never verified" (fresh client can't skip on no
        # evidence) and both branches collapse to the same identical-weight
        # case. Setting last_verified_round to a recent past round lets
        # theta_low / theta_high do their job.
        strategy = TavsEspStrategy(config=cfg)
        proxies = [MockClientProxy(f"c{i}") for i in range(6)]
        for i, p in enumerate(proxies):
            strategy.scheduler.join_rounds[p.cid] = -100
            strategy.scheduler.trust_scores[p.cid] = 0.25 + 0.1 * i
            # Non-trivial clean streak so Tier-2/3 gating is meaningful.
            strategy.scheduler.clean_streaks[p.cid] = 3
            # Recent enough that neither staleness cap fires.
            strategy.scheduler.last_verified_round[p.cid] = 0
            strategy.scheduler.appearances_since_verified[p.cid] = 0

        # Heterogeneous updates so BVD produces non-uniform behaviour_scores
        # in the "flag on" branch. Without this, all clients score 1.0 and
        # both branches collapse to identical num_examples weights.
        rng = np.random.RandomState(0)
        results = []
        for i, p in enumerate(proxies):
            # Two clients at 50x scale trigger BVD's outlier branch and
            # receive lower behaviour scores when trust weighting is on;
            # the same clients receive num_examples-only weight when off.
            scale = 50.0 if i < 2 else 0.1
            arr = (rng.randn(150000) * scale).astype(np.float32)
            results.append((p, MockFitRes(
                parameters=MockParameters([arr]),
                metrics={"is_verified": True, "client_id": f"honest_{i:02d}"},
            )))

        params, _metrics = strategy.aggregate_fit(1, results, [])
        return mock_parameters_to_ndarrays(params)

    a = one_run(disable_flag=False)[0].ravel()
    b = one_run(disable_flag=True)[0].ravel()
    diff = float(np.abs(a - b).sum())
    assert diff > 1e-4, (
        f"aggregate is unchanged when the flag flips (L1 diff={diff:.2e}) -- "
        f"the ablation is a no-op and the pilot would compare an arm to itself"
    )
    print(f"✓ flag changes the aggregate (L1 diff = {diff:.4f})")

    # Default TavsEspConfig has the flag ON (num_examples-only aggregation).
    # The default was flipped from False to True after the n=6 ablation
    # showed no cost from removing trust weighting. Guard the current
    # default so a silent flip back to False stays visible.
    from src.tavs_v2 import TavsEspConfig
    default = TavsEspConfig()
    assert getattr(default, "disable_trust_weighted_aggregation", None) is True, (
        "TavsEspConfig() default has silently changed -- experiments that "
        "assume num_examples-only aggregation would revert to trust weighting"
    )
    print("✓ TavsEspConfig() default has trust-weighting OFF (num_examples only)")
    return True


def test_cid_to_client_config_id_populated_and_stable():
    """`aggregate_fit` records proxy.cid -> "honest_XX" from FitRes metrics.

    The noise diagnostic joins on this map. If the map is stale or
    incomplete, the pilot's "did TAVS trust the noisy clients less"
    plot reads the wrong trust score for each config id and the
    mechanism proof is silently corrupted. This test locks the contract:
    every seen client is recorded on the first FitRes it produces, and
    subsequent aggregate_fit calls with disjoint clients accumulate --
    they do not drop or corrupt the earlier entries.
    """
    print("\nTesting cid_to_client_config_id accumulation across rounds...")
    config = DummyConfig()
    strategy = TavsEspStrategy(config=config)

    # First round: clients 0-3.
    proxies_r1 = [MockClientProxy(f"cid_{i}") for i in range(4)]
    for c in proxies_r1:
        strategy.scheduler.join_rounds[c.cid] = -100  # bypass ramp cap
    results_r1 = [
        (p, MockFitRes(
            parameters=MockParameters([np.random.randn(150000) * 0.05]),
            metrics={"client_id": f"honest_{i:02d}"},
        ))
        for i, p in enumerate(proxies_r1)
    ]
    strategy.aggregate_fit(1, results_r1, [])
    assert len(strategy.cid_to_client_config_id) == 4
    for i, p in enumerate(proxies_r1):
        assert strategy.cid_to_client_config_id[p.cid] == f"honest_{i:02d}"

    # Second round: clients 4-7 (disjoint from round 1). The round-1 mapping
    # must still be present.
    proxies_r2 = [MockClientProxy(f"cid_{i}") for i in range(4, 8)]
    for c in proxies_r2:
        strategy.scheduler.join_rounds[c.cid] = -100
    results_r2 = [
        (p, MockFitRes(
            parameters=MockParameters([np.random.randn(150000) * 0.05]),
            metrics={"client_id": f"honest_{i:02d}"},
        ))
        for i, p in zip((4, 5, 6, 7), proxies_r2)
    ]
    strategy.aggregate_fit(2, results_r2, [])
    assert len(strategy.cid_to_client_config_id) == 8
    for i, p in enumerate(proxies_r1):
        assert strategy.cid_to_client_config_id[p.cid] == f"honest_{i:02d}", \
            f"round-1 entry for {p.cid} lost after round 2"
    for i, p in zip((4, 5, 6, 7), proxies_r2):
        assert strategy.cid_to_client_config_id[p.cid] == f"honest_{i:02d}"

    # Export includes the map.
    state = strategy.export_complete_state()
    assert "cid_to_client_config_id" in state
    assert state["cid_to_client_config_id"] == strategy.cid_to_client_config_id
    print(f"✓ map accumulated across rounds ({len(strategy.cid_to_client_config_id)} "
          f"clients) and exported via export_complete_state")
    return True


def test_random_skip_is_verified_flag_matches_split():
    """The is_verified flag sent to each client matches its V/P assignment.

    aggregate_fit routes by _round_assignments, not the client-echoed flag,
    so a mismatch here would not corrupt aggregation -- but it would put an
    honest client into unexpected training mode (some clients gate local
    behaviour on this flag). Guard the contract explicitly.
    """
    print("\nTesting RandomSkip is_verified flag routing...")
    config = DummyConfig()
    strategy = RandomSkipStrategy(config=config, skip_rate=0.5, skip_seed=3)
    client_manager = MockClientManager(num_clients=12)
    params = MockParameters([np.random.randn(150000)])

    fit_configs = strategy.configure_fit(1, params, client_manager)
    v = strategy._round_assignments[1]["verified"]
    for proxy, fit_ins in fit_configs:
        told = fit_ins.config["is_verified"]
        in_v = proxy.cid in v
        assert told == in_v, f"{proxy.cid}: told={told} but in_v={in_v}"
    print("✓ every client's is_verified flag matches its V-membership")
    return True


def main():
    """Run all TAVS-ESP strategy tests."""
    print("🧪 TAVS-ESP Strategy Test Suite")
    print("=" * 50)

    try:
        success1 = test_tavs_esp_strategy_initialization()
        success2 = test_configure_fit_scheduling()
        success3 = test_aggregate_fit_esp_layer()
        success4 = test_trust_dynamics_integration()
        success5 = test_end_to_end_fl_simulation()

        rs1 = test_random_skip_rate_validation()
        rs2 = test_random_skip_split_is_partition()
        rs3 = test_random_skip_rate_matches_target()
        rs4 = test_random_skip_never_leaves_v_empty()
        rs5 = test_random_skip_seed_reproducibility_and_independence()
        rs6 = test_cid_to_client_config_id_populated_and_stable()
        rs7 = test_random_skip_is_verified_flag_matches_split()
        rs8 = test_disable_trust_weighted_aggregation_flag_changes_aggregate()

        if all([success1, success2, success3, success4, success5,
                rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8]):
            print(f"\n🎯 All TAVS-ESP Strategy tests PASSED!")
            return True
        else:
            print(f"\n❌ Some TAVS-ESP Strategy tests FAILED")
            return False

    except Exception as e:
        print(f"\n❌ TAVS-ESP Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)