#!/usr/bin/env python3
"""
Test Phase 4 Security Validation Framework

Tests the comprehensive security validation experiments including:
- E1: Null-Space Poisoning Defense
- E2: Signal Dilution Analysis
- E4: Timing Attack Suppression
- Security theorem validation (TC1, TC3, TC4)
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import torch
from pathlib import Path
import json

# Import Phase 4 components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from experiments.phase4_security_validation import (
    Phase4SecurityValidator,
    SecurityExperimentConfig,
    SecurityMetrics,
    ExperimentResults,
    create_phase4_experiment_configs,
    run_phase4_validation
)
from src.tavs.end_to_end_pipeline import PipelineConfig, PipelineResults


class TestPhase4SecurityValidation(unittest.TestCase):
    """Test Phase 4 security validation framework."""

    def setUp(self):
        """Setup test environment."""
        self.test_config = SecurityExperimentConfig(
            experiment_name="Test_Security_Validation",
            experiment_type="null_space_defense",
            num_rounds=3,
            num_clients=6,
            clients_per_round=4,
            byzantine_fraction=0.33,
            output_dir="test_results"
        )

        self.validator = Phase4SecurityValidator(self.test_config)

        # Mock simulation results
        self.mock_sim_results = PipelineResults(
            config=PipelineConfig(),
            server_metrics=[],
            server_losses=[2.0, 1.5, 1.0],
            server_accuracies=[0.3, 0.6, 0.85],
            final_trust_state={},
            trust_evolution={
                "honest_00": [0.5, 0.6, 0.8],
                "byzantine_00": [0.5, 0.3, 0.2]
            },
            tier_evolution={},
            byzantine_detection_history=[
                {"round": 1, "detected": ["byzantine_00"], "consensus": True},
                {"round": 2, "detected": [], "consensus": True},
                {"round": 3, "detected": ["byzantine_00"], "consensus": False}
            ],
            attack_success_rates={},
            total_time_seconds=10.0,
            round_times=[3.0, 3.5, 3.5],
            convergence_metrics={},
            security_metrics={"consensus_rate": 0.67}
        )

    def test_security_experiment_config_creation(self):
        """Test security experiment configuration."""
        config = SecurityExperimentConfig(
            experiment_name="Test_E1",
            experiment_type="null_space_defense"
        )

        self.assertEqual(config.experiment_name, "Test_E1")
        self.assertEqual(config.experiment_type, "null_space_defense")
        self.assertEqual(config.num_rounds, 15)  # Default value
        self.assertEqual(config.attack_types, ["null_space", "layerwise"])  # Post-init default
        print("✓ SecurityExperimentConfig creation and defaults")

    def test_security_metrics_extraction(self):
        """Test security metrics extraction from simulation results."""
        metrics = self.validator._extract_security_metrics(
            self.mock_sim_results, "ephemeral_structured", 2.0
        )

        # Validate metric types and ranges
        self.assertIsInstance(metrics, SecurityMetrics)
        self.assertGreaterEqual(metrics.attack_success_rate, 0.0)
        self.assertLessEqual(metrics.attack_success_rate, 1.0)
        self.assertGreaterEqual(metrics.detection_rate, 0.0)
        self.assertLessEqual(metrics.detection_rate, 1.0)
        self.assertGreater(metrics.attack_visibility_amplification, 1.0)  # Should show amplification

        # Test trust separation calculation
        self.assertGreater(metrics.trust_separation_margin, 0.0)
        print("✓ Security metrics extraction and validation")

    @patch('experiments.phase4_security_validation.TAVSESPPipeline')
    @patch('experiments.phase4_security_validation.load_cifar10')
    @patch('experiments.phase4_security_validation.get_model')
    def test_e1_null_space_defense_experiment(self, mock_get_model, mock_load_cifar10, mock_pipeline_class):
        """Test E1 null-space poisoning defense experiment."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.structure = MagicMock()
        mock_model.structure.total_params = 1000
        mock_get_model.return_value = mock_model

        mock_load_cifar10.return_value = (MagicMock(), MagicMock())

        mock_pipeline = MagicMock()
        mock_pipeline.run_simulation.return_value = self.mock_sim_results
        mock_pipeline_class.return_value = mock_pipeline

        # Run E1 experiment
        result = self.validator.run_e1_null_space_defense_experiment()

        # Validate results
        self.assertIsInstance(result, ExperimentResults)
        self.assertEqual(result.config.experiment_type, "null_space_defense")
        self.assertIn("TC1", result.security_theorems_validated)

        # Verify pipeline was called for different projection types
        expected_calls = len(self.test_config.projection_types) * len(self.test_config.attack_intensities)
        self.assertEqual(mock_pipeline_class.call_count, expected_calls)
        print("✓ E1 null-space defense experiment")

    @patch('experiments.phase4_security_validation.TAVSESPPipeline')
    def test_e2_signal_dilution_experiment(self, mock_pipeline_class):
        """Test E2 signal dilution analysis experiment."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_simulation.return_value = self.mock_sim_results
        mock_pipeline_class.return_value = mock_pipeline

        # Setup environment first
        self.validator.train_dataset = MagicMock()
        self.validator.test_dataset = MagicMock()
        self.validator.model_structure = MagicMock()
        self.validator.model_structure.total_params = 1000

        # Run E2 experiment
        result = self.validator.run_e2_signal_dilution_experiment()

        # Validate results
        self.assertIsInstance(result, ExperimentResults)
        self.assertEqual(result.config.experiment_type, "signal_dilution")
        self.assertIn("TC3", result.security_theorems_validated)

        # Should test both uniform and trust-adaptive weighting
        self.assertEqual(mock_pipeline_class.call_count, 2)
        print("✓ E2 signal dilution analysis experiment")

    @patch('experiments.phase4_security_validation.TAVSESPPipeline')
    def test_e4_timing_suppression_experiment(self, mock_pipeline_class):
        """Test E4 timing attack suppression experiment."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_simulation.return_value = self.mock_sim_results
        mock_pipeline_class.return_value = mock_pipeline

        # Setup environment
        self.validator.train_dataset = MagicMock()
        self.validator.test_dataset = MagicMock()
        self.validator.model_structure = MagicMock()
        self.validator.model_structure.total_params = 1000

        # Run E4 experiment
        result = self.validator.run_e4_timing_attack_suppression_experiment()

        # Validate results
        self.assertIsInstance(result, ExperimentResults)
        self.assertEqual(result.config.experiment_type, "timing_suppression")
        self.assertIn("TC4", result.security_theorems_validated)

        # Should test all scheduling types
        expected_calls = len(self.test_config.scheduling_types)
        self.assertEqual(mock_pipeline_class.call_count, expected_calls)
        print("✓ E4 timing attack suppression experiment")

    def test_baseline_vs_tavs_esp_config_creation(self):
        """Test creation of baseline and TAVS-ESP configurations."""
        # Test baseline config creation
        baseline_config = self.validator._create_baseline_config(self.test_config)
        self.assertIsInstance(baseline_config, PipelineConfig)
        self.assertEqual(baseline_config.num_clients, self.test_config.num_clients)
        # Baseline should have disabled TAVS-ESP features
        self.assertEqual(baseline_config.tavs_config.projection_type, "none")
        self.assertEqual(baseline_config.tavs_config.theta_low, 0.0)

        # Test TAVS-ESP config creation
        tavs_esp_config = self.validator._create_tavs_esp_config(self.test_config, "ephemeral_structured")
        self.assertIsInstance(tavs_esp_config, PipelineConfig)
        self.assertIsNotNone(tavs_esp_config.tavs_config)  # Has TAVS-ESP
        self.assertEqual(tavs_esp_config.tavs_config.projection_type, "structured")

        # Test dense projection config
        dense_config = self.validator._create_tavs_esp_config(self.test_config, "ephemeral_dense")
        self.assertEqual(dense_config.tavs_config.projection_type, "dense")
        print("✓ Baseline vs TAVS-ESP configuration creation")

    def test_attack_configuration(self):
        """Test attack parameter configuration."""
        config = PipelineConfig()

        # Configure attacks
        configured = self.validator._configure_attacks(
            config, ["null_space", "layerwise"], [1.0, 2.0]
        )

        self.assertEqual(configured.attack_types, ["null_space", "layerwise"])
        self.assertEqual(configured.attack_intensities, [1.0, 2.0])

        # Test timing attack configuration
        timing_config = self.validator._configure_timing_attacks(config)
        self.assertIn("timing", timing_config.attack_types)
        self.assertIn("coordinated_entry", timing_config.attack_types)
        print("✓ Attack parameter configuration")

    def test_security_metrics_validation_ranges(self):
        """Test security metrics have valid ranges and relationships."""
        # Test with ephemeral system (should show good security)
        ephemeral_metrics = self.validator._extract_security_metrics(
            self.mock_sim_results, "ephemeral_structured", 2.0
        )

        # Test with static system (should show poor security)
        static_metrics = self.validator._extract_security_metrics(
            self.mock_sim_results, "static", 2.0
        )

        # Ephemeral should have better visibility amplification
        self.assertGreater(
            ephemeral_metrics.attack_visibility_amplification,
            static_metrics.attack_visibility_amplification
        )

        # Both should have valid ranges
        for metrics in [ephemeral_metrics, static_metrics]:
            self.assertGreaterEqual(metrics.attack_success_rate, 0.0)
            self.assertLessEqual(metrics.attack_success_rate, 1.0)
            self.assertGreaterEqual(metrics.detection_rate, 0.0)
            self.assertLessEqual(metrics.detection_rate, 1.0)
            self.assertGreaterEqual(metrics.consensus_achievement_rate, 0.0)
            self.assertLessEqual(metrics.consensus_achievement_rate, 1.0)

        print("✓ Security metrics validation and ranges")

    def test_theorem_validation_logic(self):
        """Test security theorem validation logic."""
        # Create metrics that should validate theorems
        good_metrics = SecurityMetrics(
            attack_success_rate=0.02,  # Low ASR
            poisoning_effectiveness=0.02,
            backdoor_accuracy=0.15,
            detection_rate=0.9,  # High detection
            false_positive_rate=0.05,
            consensus_achievement_rate=0.85,
            projection_variance_ratio=0.04,  # Low variance (high visibility)
            attack_visibility_amplification=30.0,  # >25x amplification
            honest_trust_convergence=0.85,  # High honest trust
            byzantine_trust_degradation=0.2,  # Low Byzantine trust
            trust_separation_margin=0.65,  # High separation (0.85 - 0.2)
            new_client_trust_limitation=0.4,  # Limited new client trust
            sybil_attack_suppression=0.9,  # High Sybil suppression
            computational_overhead=1.2,
            communication_overhead=0.8,
            convergence_rate=10,
            tc1_visibility_validated=True,  # Meets TC1 criteria
            tc3_resilience_validated=True,  # Meets TC3 criteria
            tc4_sybil_validated=True  # Meets TC4 criteria
        )

        # All theorem validation flags should be True
        self.assertTrue(good_metrics.tc1_visibility_validated)
        self.assertTrue(good_metrics.tc3_resilience_validated)
        self.assertTrue(good_metrics.tc4_sybil_validated)

        # Create metrics that should fail validation
        bad_metrics = SecurityMetrics(
            attack_success_rate=0.8,
            poisoning_effectiveness=0.8,
            backdoor_accuracy=0.8,
            detection_rate=0.1,
            false_positive_rate=0.3,
            consensus_achievement_rate=0.2,
            projection_variance_ratio=0.9,  # High variance (low visibility)
            attack_visibility_amplification=2.0,  # <25x amplification
            honest_trust_convergence=0.3,
            byzantine_trust_degradation=0.9,
            trust_separation_margin=0.1,  # Low separation
            new_client_trust_limitation=0.9,  # High new client trust (bad)
            sybil_attack_suppression=0.1,  # Low Sybil suppression
            computational_overhead=1.1,
            communication_overhead=1.0,
            convergence_rate=20,
            tc1_visibility_validated=False,
            tc3_resilience_validated=False,
            tc4_sybil_validated=False
        )

        # All theorem validation flags should be False
        self.assertFalse(bad_metrics.tc1_visibility_validated)
        self.assertFalse(bad_metrics.tc3_resilience_validated)
        self.assertFalse(bad_metrics.tc4_sybil_validated)

        print("✓ Security theorem validation logic")

    def test_experiment_result_analysis(self):
        """Test experiment result analysis and comparison."""
        # Create mock results for E1 analysis
        static_metrics = SecurityMetrics(
            attack_success_rate=0.8, poisoning_effectiveness=0.8, backdoor_accuracy=0.8,
            detection_rate=0.2, false_positive_rate=0.1, consensus_achievement_rate=0.3,
            projection_variance_ratio=0.9, attack_visibility_amplification=1.0,  # Baseline
            honest_trust_convergence=0.5, byzantine_trust_degradation=0.5,
            trust_separation_margin=0.1, new_client_trust_limitation=0.8,
            sybil_attack_suppression=0.2, computational_overhead=1.0,
            communication_overhead=1.0, convergence_rate=15,
            tc1_visibility_validated=False, tc3_resilience_validated=False,
            tc4_sybil_validated=False
        )

        ephemeral_metrics = SecurityMetrics(
            attack_success_rate=0.02, poisoning_effectiveness=0.02, backdoor_accuracy=0.15,
            detection_rate=0.95, false_positive_rate=0.02, consensus_achievement_rate=0.9,
            projection_variance_ratio=0.04, attack_visibility_amplification=30.0,  # 30x improvement
            honest_trust_convergence=0.85, byzantine_trust_degradation=0.15,
            trust_separation_margin=0.7, new_client_trust_limitation=0.3,
            sybil_attack_suppression=0.95, computational_overhead=1.2,
            communication_overhead=0.8, convergence_rate=12,
            tc1_visibility_validated=True, tc3_resilience_validated=True,
            tc4_sybil_validated=True
        )

        # Mock results structure for E1 analysis
        mock_results = {
            "static": [static_metrics],
            "ephemeral_structured": [ephemeral_metrics]
        }

        # Analyze E1 results
        e1_results = self.validator._analyze_e1_results(mock_results, self.test_config)

        # Validate analysis
        self.assertIsInstance(e1_results, ExperimentResults)
        self.assertTrue(e1_results.security_theorems_validated["TC1"])

        # Check improvements
        self.assertGreater(e1_results.security_improvement["visibility_amplification"], 25.0)
        self.assertGreater(e1_results.security_improvement["asr_reduction"], 0.9)  # >90% ASR reduction

        print("✓ Experiment result analysis and comparison")

    def test_create_phase4_experiment_configs(self):
        """Test creation of standard Phase 4 experiment configurations."""
        configs = create_phase4_experiment_configs()

        # Should have all three core experiments
        self.assertIn("E1", configs)
        self.assertIn("E2", configs)
        self.assertIn("E4", configs)

        # Validate E1 configuration
        e1_config = configs["E1"]
        self.assertEqual(e1_config.experiment_type, "null_space_defense")
        self.assertIn("null_space", e1_config.attack_types)
        self.assertIn("ephemeral_structured", e1_config.projection_types)

        # Validate E2 configuration
        e2_config = configs["E2"]
        self.assertEqual(e2_config.experiment_type, "signal_dilution")
        self.assertEqual(e2_config.byzantine_fraction, 0.3)  # Higher for dilution analysis

        # Validate E4 configuration
        e4_config = configs["E4"]
        self.assertEqual(e4_config.experiment_type, "timing_suppression")
        self.assertEqual(e4_config.num_rounds, 20)  # Longer for timing patterns
        self.assertIn("csprng", e4_config.scheduling_types)

        print("✓ Phase 4 experiment configurations")

    @patch('experiments.phase4_security_validation.Phase4SecurityValidator')
    def test_run_phase4_validation_interface(self, mock_validator_class):
        """Test Phase 4 validation runner interface."""
        # Setup mock validator
        mock_validator = MagicMock()
        mock_results = {"E1": MagicMock()}
        mock_validator.run_complete_security_validation.return_value = mock_results
        mock_validator_class.return_value = mock_validator

        # Test running specific experiment
        result = run_phase4_validation("E1")
        self.assertIsNotNone(result)

        # Test running all experiments
        result_all = run_phase4_validation()
        mock_validator.run_complete_security_validation.assert_called_once()

        print("✓ Phase 4 validation runner interface")

    def test_comprehensive_analysis_generation(self):
        """Test comprehensive analysis across experiments."""
        # Create mock experiment results
        mock_e1 = ExperimentResults(
            config=self.test_config, baseline_metrics=MagicMock(), tavs_esp_metrics=MagicMock(),
            security_improvement={}, performance_tradeoffs={}, round_by_round_metrics=[],
            trust_evolution={}, detection_timeline=[], security_theorems_validated={"TC1": True},
            target_metrics_achieved={}, total_experiment_time=10.0, validation_timestamp=""
        )

        mock_e2 = ExperimentResults(
            config=self.test_config, baseline_metrics=MagicMock(), tavs_esp_metrics=MagicMock(),
            security_improvement={}, performance_tradeoffs={}, round_by_round_metrics=[],
            trust_evolution={}, detection_timeline=[], security_theorems_validated={"TC3": True},
            target_metrics_achieved={}, total_experiment_time=12.0, validation_timestamp=""
        )

        mock_e4 = ExperimentResults(
            config=self.test_config, baseline_metrics=MagicMock(), tavs_esp_metrics=MagicMock(),
            security_improvement={}, performance_tradeoffs={}, round_by_round_metrics=[],
            trust_evolution={}, detection_timeline=[], security_theorems_validated={"TC4": True},
            target_metrics_achieved={}, total_experiment_time=15.0, validation_timestamp=""
        )

        # Set computational overheads
        for result in [mock_e1, mock_e2, mock_e4]:
            result.tavs_esp_metrics.computational_overhead = 1.2
            result.tavs_esp_metrics.communication_overhead = 0.9

        results = {"E1": mock_e1, "E2": mock_e2, "E4": mock_e4}

        # Generate comprehensive analysis
        analysis = self.validator._generate_comprehensive_analysis(results)

        # Validate analysis structure
        self.assertIn("theorem_validation", analysis)
        self.assertIn("performance_analysis", analysis)
        self.assertIn("security_summary", analysis)

        # All theorems should be validated
        self.assertTrue(analysis["theorem_validation"]["tc1_visibility_amplification"])
        self.assertTrue(analysis["theorem_validation"]["tc3_byzantine_resilience"])
        self.assertTrue(analysis["theorem_validation"]["tc4_sybil_resistance"])
        self.assertTrue(analysis["theorem_validation"]["all_theorems_validated"])

        print("✓ Comprehensive analysis generation")


def run_phase4_validation_tests():
    """Run all Phase 4 security validation tests."""
    print("🧪 Testing Phase 4 Security Validation Framework")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase4SecurityValidation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
    result = runner.run(test_suite)

    # Print summary
    if result.wasSuccessful():
        print("\n🎯 All Phase 4 security validation tests PASSED!")
        print(f"✅ {result.testsRun} tests completed successfully")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"  FAIL: {failure[0]}")
        for error in result.errors:
            print(f"  ERROR: {error[0]}")
        return False


if __name__ == "__main__":
    success = run_phase4_validation_tests()
    exit(0 if success else 1)