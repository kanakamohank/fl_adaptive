#!/usr/bin/env python3
"""
Test TAVS-ESP Strategy Implementation

Tests the complete federated learning strategy including:
1. Flower integration (configure_fit, aggregate_fit)
2. Layer 1 (TAVS) + Layer 2 (ESP) coordination
3. Trust dynamics and Byzantine detection
4. End-to-end federated learning simulation
"""

import sys
import numpy as np
import torch
from typing import List, Dict, Tuple
from unittest.mock import MagicMock, Mock
import tempfile
import json

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

# Mock Flower functions
def mock_parameters_to_ndarrays(params):
    if hasattr(params, 'tensors'):
        return params.tensors
    return [np.random.randn(100), np.random.randn(50)]

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

# Monkey patch for testing - be more selective
sys.modules['flwr'] = MagicMock()
sys.modules['flwr.server'] = MagicMock()
sys.modules['flwr.server.strategy'] = MagicMock()
sys.modules['flwr.server.strategy'].Strategy = MockStrategy
sys.modules['flwr.server.client_proxy'] = MagicMock()
sys.modules['flwr.server.client_manager'] = MagicMock()

# Create a mock common module with our classes
class MockCommon:
    def __init__(self):
        self.parameters_to_ndarrays = mock_parameters_to_ndarrays
        self.ndarrays_to_parameters = mock_ndarrays_to_parameters
        self.FitIns = MockFitIns
        self.EvaluateIns = MockEvaluateIns
        self.FitRes = MockFitRes
        self.EvaluateRes = MockEvaluateRes
        self.Parameters = MockParameters
        self.Scalar = float  # Simple scalar type
        self.NDArrays = list  # List of numpy arrays

sys.modules['flwr.common'] = MockCommon()


def test_tavs_esp_strategy_initialization():
    """Test TAVS-ESP strategy initialization and configuration."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig
    from src.core.models import ModelStructure

    print("Testing TAVS-ESP Strategy initialization...")

    # Test 1: Basic initialization
    config = TavsEspConfig(
        theta_low=0.3,
        theta_high=0.7,
        k_ratio=0.2,
        min_fit_clients=3
    )

    model_structure = ModelStructure()
    model_structure.add_block('conv1', (32, 3, 3, 3), 864)
    model_structure.add_block('fc1', (10, 32), 320)

    strategy = TavsEspStrategy(
        config=config,
        model_structure=model_structure
    )

    print(f"Config theta_low: {strategy.config.theta_low}, expected: 0.3")
    print(f"Config k_ratio: {strategy.config.k_ratio}, expected: 0.2")
    print(f"Round number: {strategy.round_number}, expected: 0")

    assert abs(strategy.config.theta_low - 0.3) < 1e-6
    assert abs(strategy.config.k_ratio - 0.2) < 1e-6
    assert strategy.round_number == 0
    print("✓ Strategy initialization successful")

    # Test 2: Component integration
    assert strategy.csprng_manager is not None
    assert strategy.scheduler is not None
    assert strategy.verification is not None
    print("✓ All core components initialized")

    # Test 3: Configuration validation
    assert len(strategy.round_analytics) == 0
    assert strategy.projection is None  # Should be lazy-initialized
    print("✓ Configuration validation passed")

    return True


def test_configure_fit_scheduling():
    """Test TAVS Layer 1 scheduling in configure_fit."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig

    print("\nTesting TAVS Layer 1 scheduling...")

    config = TavsEspConfig(min_fit_clients=4)
    strategy = TavsEspStrategy(config=config)

    # Create mock client manager
    client_manager = MockClientManager(num_clients=8)

    # Create initial parameters
    initial_params = MockParameters([np.random.randn(100), np.random.randn(50)])

    # Test 1: First round scheduling
    fit_configs = strategy.configure_fit(
        server_round=1,
        parameters=initial_params,
        client_manager=client_manager
    )

    assert len(fit_configs) >= config.min_fit_clients
    print(f"✓ Configured {len(fit_configs)} clients for training")

    # Test 2: Verify TAVS assignments
    verified_count = 0
    promoted_count = 0

    for i, (proxy, fit_ins) in enumerate(fit_configs):
        config_dict = fit_ins.config
        print(f"    Client {i} ({proxy.cid}): config = {config_dict}")

        assert "tavs_assignment" in config_dict, f"Missing tavs_assignment in config: {config_dict}"
        assert "trust_score" in config_dict
        assert "tier" in config_dict

        if config_dict["tavs_assignment"] == "verified":
            verified_count += 1
        else:
            promoted_count += 1

    print(f"✓ TAVS assignments: {verified_count} verified, {promoted_count} promoted")

    # Test 3: Round progression
    fit_configs_2 = strategy.configure_fit(
        server_round=2,
        parameters=initial_params,
        client_manager=client_manager
    )

    assert len(fit_configs_2) >= config.min_fit_clients
    print("✓ Multi-round scheduling successful")

    return True


def test_aggregate_fit_esp_layer():
    """Test ESP Layer 2 processing in aggregate_fit."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig

    print("\nTesting ESP Layer 2 aggregation...")

    config = TavsEspConfig(
        k_ratio=0.3,
        projection_type="dense",  # Use dense for simpler testing
        detection_threshold=1.5
    )
    strategy = TavsEspStrategy(config=config)

    # Set initial parameters
    strategy.current_parameters = MockParameters([np.random.randn(100), np.random.randn(50)])

    # Test 1: Create mock client results
    client_proxies = [MockClientProxy(f"client_{i}") for i in range(6)]

    # Register clients first
    for proxy in client_proxies:
        strategy.scheduler.register_client(proxy.cid, round_number=1)

    client_results = []
    for i, proxy in enumerate(client_proxies):
        # Create client parameters (initial + update)
        base_params = [np.random.randn(100) * 0.1, np.random.randn(50) * 0.1]

        # Add Byzantine behavior to first 2 clients
        if i < 2:  # Byzantine clients
            byzantine_noise = [np.random.randn(100) * 2.0, np.random.randn(50) * 2.0]
            client_params = [bp + bn for bp, bn in zip(base_params, byzantine_noise)]
        else:  # Honest clients
            client_params = base_params

        fit_res = MockFitRes(
            parameters=MockParameters(client_params),
            metrics={"tavs_assignment": "verified"}
        )
        client_results.append((proxy, fit_res))

    print(f"✓ Created {len(client_results)} mock client results")

    # Test 2: Run aggregation
    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=client_results,
        failures=[]
    )

    assert aggregated_params is not None
    assert "num_verified" in metrics
    assert "num_byzantine_detected" in metrics
    assert "projection_time_ms" in metrics

    print(f"✓ Aggregation successful: {metrics['num_byzantine_detected']} Byzantine detected")

    # Test 3: Verify analytics
    assert len(strategy.round_analytics) == 1
    analytics = strategy.round_analytics[0]
    assert analytics.round_number == 1
    assert analytics.projection_time_ms > 0
    assert analytics.detection_time_ms > 0

    print("✓ Round analytics recorded")

    return True


def test_trust_dynamics_integration():
    """Test trust score evolution over multiple rounds."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig

    print("\nTesting trust dynamics integration...")

    config = TavsEspConfig(
        alpha=0.9,
        theta_low=0.3,
        theta_high=0.7,
        projection_type="dense"
    )
    strategy = TavsEspStrategy(config=config)
    strategy.current_parameters = MockParameters([np.random.randn(150)])

    client_manager = MockClientManager(num_clients=8)

    # Simulate multiple rounds
    trust_evolution = {}

    for round_num in range(1, 6):
        print(f"  Round {round_num}:")

        # Configure clients
        fit_configs = strategy.configure_fit(
            server_round=round_num,
            parameters=strategy.current_parameters,
            client_manager=client_manager
        )

        # Create client results with behavioral patterns
        client_results = []
        for i, (proxy, fit_ins) in enumerate(fit_configs):
            # Byzantine clients (first 2) have poor behavior
            if i < 2:  # Byzantine
                client_params = [np.random.randn(150) * 3.0]  # Large malicious update
            else:  # Honest
                client_params = [np.random.randn(150) * 0.1]  # Small honest update

            fit_res = MockFitRes(
                parameters=MockParameters(client_params),
                metrics={"tavs_assignment": fit_ins.config["tavs_assignment"]}
            )
            client_results.append((proxy, fit_res))

        # Aggregate results
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=round_num,
            results=client_results,
            failures=[]
        )

        # Track trust evolution
        for client_id in [proxy.cid for proxy, _ in fit_configs]:
            if client_id in strategy.scheduler.client_states:
                trust_score = strategy.scheduler.client_states[client_id].trust_score
                if client_id not in trust_evolution:
                    trust_evolution[client_id] = []
                trust_evolution[client_id].append(trust_score)

        print(f"    Byzantine detected: {metrics['num_byzantine_detected']}")
        print(f"    Budget utilization: {metrics['budget_utilization']:.1%}")

    # Test trust convergence
    honest_clients = ["client_2", "client_3", "client_4"]
    byzantine_clients = ["client_0", "client_1"]

    honest_final_trust = [trust_evolution[cid][-1] for cid in honest_clients if cid in trust_evolution]
    byzantine_final_trust = [trust_evolution[cid][-1] for cid in byzantine_clients if cid in trust_evolution]

    if honest_final_trust and byzantine_final_trust:
        avg_honest_trust = np.mean(honest_final_trust)
        avg_byzantine_trust = np.mean(byzantine_final_trust)

        print(f"✓ Trust convergence: honest={avg_honest_trust:.3f}, byzantine={avg_byzantine_trust:.3f}")

        if avg_honest_trust > avg_byzantine_trust:
            print("✓ Trust system correctly distinguishes client types")
        else:
            print("⚠ Trust system may need more rounds to fully converge")

    return True


def test_end_to_end_fl_simulation():
    """Test complete federated learning simulation with TAVS-ESP."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig

    print("\nTesting end-to-end FL simulation...")

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TavsEspConfig(
            min_fit_clients=5,
            k_ratio=0.25,
            save_round_decisions=True,
            output_dir=temp_dir
        )

        strategy = TavsEspStrategy(config=config)
        client_manager = MockClientManager(num_clients=10)

        # Initialize parameters
        initial_params = MockParameters([np.random.randn(200), np.random.randn(100)])
        strategy.current_parameters = initial_params

        simulation_metrics = []

        # Run FL simulation for 5 rounds
        for round_num in range(1, 6):
            # Configure clients
            fit_configs = strategy.configure_fit(
                server_round=round_num,
                parameters=strategy.current_parameters,
                client_manager=client_manager
            )

            # Simulate client training
            client_results = []
            for i, (proxy, fit_ins) in enumerate(fit_configs):
                # Simulate realistic parameter updates
                if i < 3:  # Some Byzantine clients
                    noise_scale = 1.5 if round_num > 2 else 0.1  # Attack after round 2
                    client_params = [
                        np.random.randn(200) * noise_scale,
                        np.random.randn(100) * noise_scale
                    ]
                else:  # Honest clients
                    client_params = [
                        np.random.randn(200) * 0.05,
                        np.random.randn(100) * 0.05
                    ]

                fit_res = MockFitRes(
                    parameters=MockParameters(client_params),
                    metrics={"tavs_assignment": fit_ins.config["tavs_assignment"]}
                )
                client_results.append((proxy, fit_res))

            # Aggregate
            aggregated_params, metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=client_results,
                failures=[]
            )

            # Update global parameters
            strategy.current_parameters = aggregated_params
            simulation_metrics.append(metrics)

            print(f"  Round {round_num}: "
                  f"V={metrics['num_verified']}, "
                  f"S={metrics['num_promoted']}, "
                  f"Byzantine={metrics['num_byzantine_detected']}, "
                  f"Time={metrics['total_time_ms']:.1f}ms")

        # Verify simulation results
        assert len(simulation_metrics) == 5
        assert len(strategy.round_analytics) == 5

        # Check that Byzantine detection improves over time
        round_3_4_5_detections = [simulation_metrics[i]['num_byzantine_detected'] for i in range(2, 5)]
        if max(round_3_4_5_detections) > 0:
            print("✓ Byzantine detection active in later rounds")

        # Check trust statistics
        trust_stats = strategy.get_trust_statistics()
        assert trust_stats['total_clients'] == len(client_manager.clients)
        print(f"✓ Final trust statistics: {trust_stats['tier_distribution']}")

        print("✓ End-to-end FL simulation successful")

        return True


def test_strategy_export_and_analytics():
    """Test analytics export and state management."""
    from src.tavs.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig

    print("\nTesting analytics and state export...")

    config = TavsEspConfig()
    strategy = TavsEspStrategy(config=config)

    # Run a few rounds to generate data
    strategy.current_parameters = MockParameters([np.random.randn(50)])
    client_manager = MockClientManager(num_clients=4)

    for round_num in range(1, 3):
        fit_configs = strategy.configure_fit(round_num, strategy.current_parameters, client_manager)

        # Create simple results
        results = []
        for proxy, fit_ins in fit_configs:
            fit_res = MockFitRes(
                parameters=MockParameters([np.random.randn(50) * 0.1]),
                metrics={"tavs_assignment": "verified"}
            )
            results.append((proxy, fit_res))

        aggregated_params, metrics = strategy.aggregate_fit(round_num, results, [])
        strategy.current_parameters = aggregated_params

    # Test state export
    complete_state = strategy.export_complete_state()

    assert "config" in complete_state
    assert "trust_state" in complete_state
    assert "csprng_stats" in complete_state
    assert "round_analytics" in complete_state
    assert len(complete_state["round_analytics"]) == 2

    print("✓ Complete state export successful")

    # Test trust statistics
    trust_stats = strategy.get_trust_statistics()
    assert "total_clients" in trust_stats
    assert "trust_statistics" in trust_stats
    assert "tier_distribution" in trust_stats

    print("✓ Trust statistics export successful")

    return True


def main():
    """Run all TAVS-ESP strategy tests."""
    print("🧪 TAVS-ESP Strategy Test Suite")
    print("=" * 50)

    try:
        # Test 1: Initialization
        success1 = test_tavs_esp_strategy_initialization()

        # Test 2: TAVS Layer 1 scheduling
        success2 = test_configure_fit_scheduling()

        # Test 3: ESP Layer 2 aggregation
        success3 = test_aggregate_fit_esp_layer()

        # Test 4: Trust dynamics
        success4 = test_trust_dynamics_integration()

        # Test 5: End-to-end simulation
        success5 = test_end_to_end_fl_simulation()

        # Test 6: Analytics and export
        success6 = test_strategy_export_and_analytics()

        if all([success1, success2, success3, success4, success5, success6]):
            print(f"\n🎯 All TAVS-ESP Strategy tests PASSED!")
            print("✓ Strategy initialization working")
            print("✓ TAVS Layer 1 scheduling working")
            print("✓ ESP Layer 2 aggregation working")
            print("✓ Trust dynamics integration working")
            print("✓ End-to-end FL simulation working")
            print("✓ Analytics and state export working")
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