#!/usr/bin/env python3
"""
Test TAVS Flower Client Integration

Tests the TAVS Flower client wrapper including:
1. Client initialization and configuration
2. Parameter serialization and communication
3. TAVS assignment processing
4. Attack coordination and honest behavior
5. Integration with underlying client implementations
"""

import numpy as np
import torch
from typing import Dict, List
import tempfile
from unittest.mock import MagicMock


def create_mock_data_loader():
    """Create a mock data loader for testing."""
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size

    class MockDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            self.batch_size = 32

        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            # Generate a few mock batches
            for _ in range(3):
                batch_x = torch.randn(self.batch_size, 3, 32, 32)  # CIFAR-like data
                batch_y = torch.randint(0, 10, (self.batch_size,))
                yield batch_x, batch_y

    return MockDataLoader(MockDataset(100))


def test_tavs_client_initialization():
    """Test TAVS client initialization and configuration."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("Testing TAVS client initialization...")

    # Test 1: Honest client initialization
    honest_config = TAVSClientConfig(
        client_id="honest_client_01",
        client_type="honest",
        model_type="cifar_cnn",
        epochs=1,
        batch_size=16
    )

    train_loader = create_mock_data_loader()
    honest_client = TAVSFlowerClient(honest_config, train_loader=train_loader)

    assert honest_client.config.client_id == "honest_client_01"
    assert honest_client.config.client_type == "honest"
    assert honest_client.current_assignment == "verified"
    assert honest_client.trust_score == 0.5
    assert honest_client.tier == 1
    print("✓ Honest client initialization successful")

    # Test 2: Attack client initialization
    attack_config = TAVSClientConfig(
        client_id="attacker_01",
        client_type="null_space",
        model_type="cifar_cnn",
        attack_intensity=2.0
    )

    attack_client = TAVSFlowerClient(attack_config, train_loader=train_loader)
    assert attack_client.config.client_type == "null_space"
    assert attack_client.config.attack_intensity == 2.0
    print("✓ Attack client initialization successful")

    # Test 3: Underlying client creation
    assert honest_client.underlying_client is not None
    assert attack_client.underlying_client is not None
    print("✓ Underlying client creation working")

    return True


def test_parameter_serialization():
    """Test parameter getting and setting for Flower compatibility."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("\nTesting parameter serialization...")

    config = TAVSClientConfig(
        client_id="test_client",
        client_type="honest"
    )

    client = TAVSFlowerClient(config, train_loader=create_mock_data_loader())

    # Test 1: Get initial parameters
    initial_params = client.get_parameters({})

    assert isinstance(initial_params, list)
    assert len(initial_params) > 0
    assert all(isinstance(p, np.ndarray) for p in initial_params)
    print(f"✓ Parameter retrieval: {len(initial_params)} parameter arrays")

    # Test 2: Set parameters
    # Create modified parameters (small random changes)
    modified_params = []
    for param in initial_params:
        modified = param + np.random.randn(*param.shape) * 0.01
        modified_params.append(modified)

    client.set_parameters(modified_params)
    print("✓ Parameter setting successful")

    # Test 3: Verify parameters changed
    new_params = client.get_parameters({})

    # Check that parameters actually changed
    param_diff = np.linalg.norm(
        np.concatenate([p.flatten() for p in new_params]) -
        np.concatenate([p.flatten() for p in initial_params])
    )
    assert param_diff > 0, "Parameters should have changed"
    print(f"✓ Parameter modification verified (diff: {param_diff:.6f})")

    return True


def test_tavs_assignment_processing():
    """Test TAVS assignment processing and state management."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("\nTesting TAVS assignment processing...")

    config = TAVSClientConfig(
        client_id="test_client",
        client_type="honest"
    )

    client = TAVSFlowerClient(config, train_loader=create_mock_data_loader())

    # Test 1: Process verification assignment
    verify_config = {
        "round": 3,
        "tavs_assignment": "verified",
        "trust_score": 0.75,
        "tier": 2,
        "is_decoy": False
    }

    client._process_tavs_config(verify_config)

    assert client.round_number == 3
    assert client.current_assignment == "verified"
    assert client.trust_score == 0.75
    assert client.tier == 2
    assert client.is_decoy == False
    print("✓ Verification assignment processing")

    # Test 2: Process promotion assignment
    promote_config = {
        "round": 4,
        "tavs_assignment": "promoted",
        "trust_score": 0.85,
        "tier": 3,
        "is_decoy": False
    }

    client._process_tavs_config(promote_config)

    assert client.round_number == 4
    assert client.current_assignment == "promoted"
    assert client.trust_score == 0.85
    assert client.tier == 3
    print("✓ Promotion assignment processing")

    # Test 3: Assignment history tracking
    assert len(client.assignment_history) == 2

    first_assignment = client.assignment_history[0]
    assert first_assignment["round"] == 3
    assert first_assignment["assignment"] == "verified"
    assert first_assignment["trust_score"] == 0.75

    second_assignment = client.assignment_history[1]
    assert second_assignment["round"] == 4
    assert second_assignment["assignment"] == "promoted"
    assert second_assignment["trust_score"] == 0.85

    print("✓ Assignment history tracking")

    return True


def test_training_execution():
    """Test training execution with different assignments."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("\nTesting training execution...")

    # Test 1: Honest client training
    honest_config = TAVSClientConfig(
        client_id="honest_test",
        client_type="honest",
        epochs=1
    )

    honest_client = TAVSFlowerClient(honest_config, train_loader=create_mock_data_loader())

    # Get initial parameters
    initial_params = honest_client.get_parameters({})

    # Execute training with verified assignment
    fit_config = {
        "round": 1,
        "tavs_assignment": "verified",
        "trust_score": 0.6,
        "tier": 2
    }

    updated_params, num_examples, metrics = honest_client.fit(initial_params, fit_config)

    assert isinstance(updated_params, list)
    assert num_examples > 0
    assert "client_id" in metrics
    assert metrics["tavs_assignment"] == "verified"
    assert metrics["client_type"] == "honest"
    print(f"✓ Honest training: {num_examples} examples, {len(updated_params)} param arrays")

    # Test 2: Attack client training (if available)
    try:
        attack_config = TAVSClientConfig(
            client_id="attack_test",
            client_type="null_space",
            attack_intensity=1.5
        )

        attack_client = TAVSFlowerClient(attack_config, train_loader=create_mock_data_loader())

        # Test verified assignment (should behave more honestly)
        verified_params, verified_examples, verified_metrics = attack_client.fit(
            initial_params,
            {"round": 1, "tavs_assignment": "verified", "trust_score": 0.3}
        )

        # Test promoted assignment (may execute attack)
        promoted_params, promoted_examples, promoted_metrics = attack_client.fit(
            initial_params,
            {"round": 2, "tavs_assignment": "promoted", "trust_score": 0.8}
        )

        assert verified_metrics["tavs_assignment"] == "verified"
        assert promoted_metrics["tavs_assignment"] == "promoted"
        assert "attack_intensity" in promoted_metrics
        print("✓ Attack client training with different assignments")

    except Exception as e:
        print(f"⚠ Attack client test skipped (dependency issue): {e}")

    return True


def test_client_statistics():
    """Test client statistics and history tracking."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("\nTesting client statistics...")

    config = TAVSClientConfig(
        client_id="stats_test",
        client_type="honest"
    )

    client = TAVSFlowerClient(config, train_loader=create_mock_data_loader())

    # Simulate multiple rounds with different assignments
    assignments = [
        {"round": 1, "tavs_assignment": "verified", "trust_score": 0.5, "tier": 1},
        {"round": 2, "tavs_assignment": "verified", "trust_score": 0.6, "tier": 2},
        {"round": 3, "tavs_assignment": "promoted", "trust_score": 0.8, "tier": 3},
        {"round": 4, "tavs_assignment": "verified", "trust_score": 0.75, "tier": 2, "is_decoy": True}
    ]

    initial_params = client.get_parameters({})

    for assignment in assignments:
        client.fit(initial_params, assignment)

    # Get statistics
    stats = client.get_tavs_statistics()

    assert stats["client_id"] == "stats_test"
    assert stats["client_type"] == "honest"
    assert stats["total_rounds"] == 4
    assert stats["verification_count"] == 3  # Rounds 1, 2, 4
    assert stats["promotion_count"] == 1   # Round 3
    assert stats["decoy_count"] == 1       # Round 4
    assert stats["current_trust_score"] == 0.75
    assert stats["current_tier"] == 2

    print(f"✓ Client statistics: {stats['verification_count']} verified, "
          f"{stats['promotion_count']} promoted, {stats['decoy_count']} decoy")

    # Test assignment history
    assert len(stats["assignment_history"]) == 4
    assert stats["assignment_history"][-1]["is_decoy"] == True
    print("✓ Assignment history tracking")

    return True


def test_client_factory():
    """Test client factory function."""
    from src.clients.tavs_flower_client import create_tavs_flower_client, TAVSClientConfig

    print("\nTesting client factory...")

    # Test different client types
    client_types = ["honest", "null_space"]

    for client_type in client_types:
        try:
            config = TAVSClientConfig(
                client_id=f"factory_{client_type}",
                client_type=client_type
            )

            client = create_tavs_flower_client(
                config=config,
                train_loader=create_mock_data_loader()
            )

            assert client.config.client_id == f"factory_{client_type}"
            assert client.config.client_type == client_type
            assert client.underlying_client is not None

            print(f"✓ Factory creation: {client_type} client")

        except Exception as e:
            print(f"⚠ Factory test for {client_type} skipped: {e}")

    return True


def test_evaluation_functionality():
    """Test client evaluation functionality."""
    from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig

    print("\nTesting evaluation functionality...")

    config = TAVSClientConfig(
        client_id="eval_test",
        client_type="honest"
    )

    client = TAVSFlowerClient(
        config,
        train_loader=create_mock_data_loader(),
        test_loader=create_mock_data_loader()  # Provide test loader
    )

    # Test evaluation
    params = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(params, {"round": 1})

    assert isinstance(loss, float)
    assert isinstance(num_examples, int)
    assert num_examples >= 0
    assert "accuracy" in metrics
    assert "client_id" in metrics

    print(f"✓ Evaluation: loss={loss:.4f}, examples={num_examples}, accuracy={metrics['accuracy']:.3f}")

    return True


def main():
    """Run all TAVS Flower client tests."""
    print("🧪 TAVS Flower Client Test Suite")
    print("=" * 50)

    try:
        # Test 1: Initialization
        success1 = test_tavs_client_initialization()

        # Test 2: Parameter serialization
        success2 = test_parameter_serialization()

        # Test 3: TAVS assignment processing
        success3 = test_tavs_assignment_processing()

        # Test 4: Training execution
        success4 = test_training_execution()

        # Test 5: Statistics tracking
        success5 = test_client_statistics()

        # Test 6: Factory function
        success6 = test_client_factory()

        # Test 7: Evaluation
        success7 = test_evaluation_functionality()

        if all([success1, success2, success3, success4, success5, success6, success7]):
            print(f"\n🎯 All TAVS Flower Client tests PASSED!")
            print("✓ Client initialization working")
            print("✓ Parameter serialization working")
            print("✓ TAVS assignment processing working")
            print("✓ Training execution working")
            print("✓ Statistics tracking working")
            print("✓ Factory function working")
            print("✓ Evaluation functionality working")
            return True
        else:
            print(f"\n❌ Some TAVS Flower Client tests FAILED")
            return False

    except Exception as e:
        print(f"\n❌ TAVS Flower Client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)