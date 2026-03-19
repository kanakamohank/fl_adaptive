#!/usr/bin/env python3
"""
Tests for verification strategy comparison experiment.

Tests the TAVS vs Full Verification comparison functionality to ensure
all components work correctly before running the full experiment.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import logging

# Suppress noisy logs during testing
logging.basicConfig(level=logging.ERROR)

from experiments.verification_strategy_comparison import (
    ComparisonConfig,
    VerificationResults,
    VerificationStrategyComparator
)


class TestVerificationStrategyComparison(unittest.TestCase):
    """Test suite for verification strategy comparison."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = ComparisonConfig(
            num_rounds=2,  # Minimal for testing
            num_clients=4,
            clients_per_round=3,
            byzantine_fraction=0.25,
            results_dir=str(self.temp_dir),
            save_plots=False  # Skip plots in tests
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_comparison_config_creation(self):
        """Test ComparisonConfig initialization and defaults."""
        config = ComparisonConfig()

        # Check default values
        self.assertEqual(config.num_rounds, 15)
        self.assertEqual(config.num_clients, 20)
        self.assertEqual(config.clients_per_round, 8)
        self.assertEqual(config.byzantine_fraction, 0.25)

        # Check attack scenarios are set in __post_init__
        self.assertIsNotNone(config.attack_scenarios)
        self.assertIn("no_attack", config.attack_scenarios)
        self.assertIn("light_attack", config.attack_scenarios)
        self.assertIn("heavy_attack", config.attack_scenarios)

    def test_verification_results_creation(self):
        """Test VerificationResults dataclass."""
        results = VerificationResults(
            strategy_name="Test Strategy",
            config=self.test_config,
            total_verification_time=10.0,
            avg_round_time=5.0,
            total_rounds=2,
            byzantine_detection_accuracy=0.85,
            false_positive_rate=0.05,
            false_negative_rate=0.10,
            trust_convergence_rounds=5,
            final_trust_distribution={"client_1": 0.8, "client_2": 0.6},
            clients_verified_per_round=[3, 3],
            verification_overhead_per_round=[5.0, 5.0],
            final_accuracy=0.87,
            convergence_accuracy=0.85,
            round_metrics=[]
        )

        self.assertEqual(results.strategy_name, "Test Strategy")
        self.assertEqual(results.total_rounds, 2)
        self.assertEqual(len(results.clients_verified_per_round), 2)

    def test_comparator_initialization(self):
        """Test VerificationStrategyComparator initialization."""
        comparator = VerificationStrategyComparator(self.test_config)

        self.assertEqual(comparator.config, self.test_config)
        self.assertTrue(comparator.results_dir.exists())
        self.assertEqual(str(comparator.results_dir), self.temp_dir)

    def test_byzantine_fraction_for_scenarios(self):
        """Test attack scenario Byzantine fraction mapping."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Test scenario mappings
        self.assertEqual(comparator._get_byzantine_fraction_for_scenario("no_attack"), 0.0)
        self.assertEqual(comparator._get_byzantine_fraction_for_scenario("light_attack"), 0.15)
        self.assertEqual(comparator._get_byzantine_fraction_for_scenario("heavy_attack"), 0.25)

        # Test default fallback
        self.assertEqual(
            comparator._get_byzantine_fraction_for_scenario("unknown_scenario"),
            self.test_config.byzantine_fraction
        )

    def test_trust_convergence_calculation(self):
        """Test trust convergence calculation logic."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Test convergence detection
        trust_evolution = {
            "client_1": [0.5, 0.6, 0.65, 0.66, 0.66, 0.66],  # Converges around index 3
            "client_2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]       # No convergence
        }

        convergence_rounds = comparator._calculate_trust_convergence(trust_evolution)
        self.assertGreater(convergence_rounds, 0)
        self.assertLessEqual(convergence_rounds, 6)

    def test_detection_accuracy_calculation(self):
        """Test Byzantine detection accuracy calculation."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Mock results object
        class MockResults:
            def __init__(self, consensus_rate):
                self.security_metrics = {'consensus_rate': consensus_rate}

        # Test with consensus rate
        results = MockResults(0.9)
        accuracy = comparator._calculate_detection_accuracy(results)
        self.assertEqual(accuracy, 0.9)

        # Test with missing security metrics (should use default)
        class EmptyResults:
            security_metrics = None

        empty_results = EmptyResults()
        accuracy = comparator._calculate_detection_accuracy(empty_results)
        self.assertEqual(accuracy, 0.85)  # Default value

    def test_final_trust_distribution_extraction(self):
        """Test final trust distribution extraction."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Mock results with trust evolution
        class MockResults:
            def __init__(self):
                self.trust_evolution = {
                    "client_1": [0.5, 0.6, 0.7],
                    "client_2": [0.3, 0.4, 0.5],
                    "client_3": []  # Empty scores
                }

        results = MockResults()
        final_trust = comparator._extract_final_trust_distribution(results)

        self.assertEqual(final_trust["client_1"], 0.7)
        self.assertEqual(final_trust["client_2"], 0.5)
        self.assertNotIn("client_3", final_trust)  # Empty scores excluded

    def test_strategy_comparison_logic(self):
        """Test strategy comparison calculations."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Create mock results
        tavs_results = VerificationResults(
            strategy_name="TAVS",
            config=self.test_config,
            total_verification_time=10.0,
            avg_round_time=2.0,  # Faster
            total_rounds=2,
            byzantine_detection_accuracy=0.85,
            false_positive_rate=0.05,
            false_negative_rate=0.10,
            trust_convergence_rounds=3,
            final_trust_distribution={},
            clients_verified_per_round=[3, 3],  # Fewer clients
            verification_overhead_per_round=[2.0, 2.0],
            final_accuracy=0.85,
            convergence_accuracy=0.85,
            round_metrics=[]
        )

        full_results = VerificationResults(
            strategy_name="Full Verification",
            config=self.test_config,
            total_verification_time=20.0,
            avg_round_time=8.0,  # Slower
            total_rounds=2,
            byzantine_detection_accuracy=0.90,  # Slightly better
            false_positive_rate=0.02,
            false_negative_rate=0.05,
            trust_convergence_rounds=0,
            final_trust_distribution={},
            clients_verified_per_round=[4, 4],  # More clients
            verification_overhead_per_round=[8.0, 8.0],
            final_accuracy=0.87,
            convergence_accuracy=0.87,
            round_metrics=[]
        )

        comparison = comparator._compare_strategies(tavs_results, full_results)

        # Check efficiency improvement
        expected_efficiency = 8.0 / 2.0  # full_time / tavs_time
        self.assertAlmostEqual(comparison['efficiency_improvement'], expected_efficiency)

        # Check accuracy difference
        expected_accuracy_diff = 0.85 - 0.90
        self.assertAlmostEqual(comparison['accuracy_difference'], expected_accuracy_diff)

        # Check resource efficiency
        expected_resource_efficiency = 4.0 / 3.0  # full_clients / tavs_clients
        self.assertAlmostEqual(comparison['resource_efficiency'], expected_resource_efficiency, places=2)

        # Check that advantages are listed
        self.assertIn('tavs_advantages', comparison)
        self.assertIn('full_verification_advantages', comparison)
        self.assertIn('recommendation', comparison)

    def test_minimal_tavs_experiment_setup(self):
        """Test TAVS experiment setup (without full execution)."""
        comparator = VerificationStrategyComparator(self.test_config)

        # Test Byzantine fraction calculation for different scenarios
        no_attack_fraction = comparator._get_byzantine_fraction_for_scenario("no_attack")
        light_attack_fraction = comparator._get_byzantine_fraction_for_scenario("light_attack")
        heavy_attack_fraction = comparator._get_byzantine_fraction_for_scenario("heavy_attack")

        self.assertEqual(no_attack_fraction, 0.0)
        self.assertEqual(light_attack_fraction, 0.15)
        self.assertEqual(heavy_attack_fraction, 0.25)

        # These fractions should be used to configure pipeline
        self.assertNotEqual(no_attack_fraction, light_attack_fraction)
        self.assertLess(light_attack_fraction, heavy_attack_fraction)

    def test_results_directory_creation(self):
        """Test results directory creation and cleanup."""
        temp_results_dir = Path(self.temp_dir) / "test_results"
        config = ComparisonConfig(results_dir=str(temp_results_dir))

        comparator = VerificationStrategyComparator(config)

        # Check directory was created
        self.assertTrue(temp_results_dir.exists())
        self.assertTrue(temp_results_dir.is_dir())


class TestVerificationComparisonIntegration(unittest.TestCase):
    """Integration tests for verification comparison (more expensive)."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Minimal configuration for integration testing
        self.integration_config = ComparisonConfig(
            num_rounds=2,
            num_clients=4,
            clients_per_round=3,
            byzantine_fraction=0.25,
            attack_scenarios=["no_attack"],  # Single scenario for speed
            results_dir=str(self.temp_dir),
            save_plots=False
        )

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skip("Skip expensive integration test by default")
    def test_minimal_full_comparison(self):
        """Test minimal full comparison (expensive - run manually)."""
        # This test actually runs the comparison but is skipped by default
        # Remove @unittest.skip to run during development

        comparator = VerificationStrategyComparator(self.integration_config)

        try:
            results = comparator.run_full_comparison()

            # Basic result validation
            self.assertIn('no_attack', results)
            self.assertIn('overall_comparison', results)
            self.assertIn('meta', results)

            # Check that both strategies were tested
            scenario_results = results['no_attack']
            self.assertIn('tavs', scenario_results)
            self.assertIn('full_verification', scenario_results)
            self.assertIn('comparison', scenario_results)

            print(f"\nIntegration test results:")
            print(f"TAVS avg round time: {scenario_results['tavs'].avg_round_time:.3f}s")
            print(f"Full avg round time: {scenario_results['full_verification'].avg_round_time:.3f}s")
            print(f"Efficiency improvement: {scenario_results['comparison']['efficiency_improvement']:.1f}x")

        except Exception as e:
            self.fail(f"Integration test failed: {e}")


def run_verification_comparison_tests():
    """Run all verification comparison tests."""
    print("Running Verification Strategy Comparison Tests...")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationStrategyComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationComparisonIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_verification_comparison_tests()

    if success:
        print("\n✅ All verification comparison tests passed!")
    else:
        print("\n❌ Some verification comparison tests failed!")
        exit(1)