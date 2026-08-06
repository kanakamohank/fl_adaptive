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
import torch
from typing import List, Dict, Tuple
from unittest.mock import MagicMock
import tempfile

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
        self.tensors = tensors

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

# Mock Flower functions
def mock_parameters_to_ndarrays(params):
    if hasattr(params, 'tensors'):
        return params.tensors
    return [np.random.randn(150000)] # Match the default 'full_model' fallback dim

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
from src.tavs.tavs_esp_strategy import TavsEspStrategy

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
    assert trust_scores["client_1"] >= 0.5, "Honest client trust incorrectly dropped"
    
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

        if all([success1, success2, success3, success4, success5]):
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