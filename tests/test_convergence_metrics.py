#!/usr/bin/env python3
"""
Tests for convergence metrics extraction and visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from experiments.verification_strategy_comparison import VerificationStrategyComparator, ComparisonConfig


class TestConvergenceMetrics(unittest.TestCase):
    """Test cases for convergence metrics extraction and visualization."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir)

        # Create test configuration
        self.config = ComparisonConfig(
            num_rounds=5,
            num_clients=10,
            clients_per_round=5,
            byzantine_fraction=0.2,
            attack_scenarios=['no_attack'],
            results_dir=str(self.results_dir),
            save_plots=True
        )

        self.experiment = VerificationStrategyComparator(self.config)

        # Create mock result directories and files
        self._create_mock_results()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_mock_results(self):
        """Create mock result files with training metrics."""

        # Create TAVS results
        tavs_dir = self.results_dir / "tavs_no_attack"
        tavs_dir.mkdir(parents=True)

        tavs_results = {
            "server_losses": [2.3, 1.8, 1.2, 0.8, 0.5],
            "server_accuracies": [0.1, 0.3, 0.5, 0.7, 0.8],
            "config": {"num_rounds": 5},
            "final_metrics": {
                "loss": 0.5,
                "accuracy": 0.8
            }
        }

        with open(tavs_dir / "pipeline_results.json", 'w') as f:
            json.dump(tavs_results, f)

        # Create Full verification results
        full_dir = self.results_dir / "full_verification_no_attack"
        full_dir.mkdir(parents=True)

        full_results = {
            "server_losses": [2.3, 1.7, 1.1, 0.7, 0.4],
            "server_accuracies": [0.1, 0.35, 0.55, 0.75, 0.85],
            "config": {"num_rounds": 5},
            "final_metrics": {
                "loss": 0.4,
                "accuracy": 0.85
            }
        }

        with open(full_dir / "pipeline_results.json", 'w') as f:
            json.dump(full_results, f)

    def test_extract_actual_losses(self):
        """Test actual loss trajectory extraction."""
        losses = self.experiment._extract_actual_losses('no_attack', 'tavs')

        self.assertEqual(len(losses), 5)
        self.assertEqual(losses[0], 2.3)  # Initial loss
        self.assertEqual(losses[-1], 0.5)  # Final loss
        self.assertLess(losses[-1], losses[0])  # Loss should decrease

    def test_extract_actual_accuracies(self):
        """Test actual accuracy trajectory extraction."""
        accuracies = self.experiment._extract_actual_accuracies('no_attack', 'tavs')

        self.assertEqual(len(accuracies), 5)
        self.assertEqual(accuracies[0], 0.1)  # Initial accuracy
        self.assertEqual(accuracies[-1], 0.8)  # Final accuracy
        self.assertGreater(accuracies[-1], accuracies[0])  # Accuracy should increase

    def test_extract_missing_files(self):
        """Test extraction with missing result files."""
        losses = self.experiment._extract_actual_losses('missing_scenario', 'tavs')
        accuracies = self.experiment._extract_actual_accuracies('missing_scenario', 'tavs')

        self.assertEqual(losses, [])
        self.assertEqual(accuracies, [])

    def test_extract_corrupted_files(self):
        """Test extraction with corrupted JSON files."""
        # Create corrupted file
        corrupted_dir = self.results_dir / "corrupted_scenario"
        corrupted_dir.mkdir(parents=True)

        with open(corrupted_dir / "pipeline_results.json", 'w') as f:
            f.write("invalid json content")

        losses = self.experiment._extract_actual_losses('corrupted_scenario', 'tavs')
        accuracies = self.experiment._extract_actual_accuracies('corrupted_scenario', 'tavs')

        self.assertEqual(losses, [])
        self.assertEqual(accuracies, [])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_convergence_plots(self, mock_close, mock_savefig):
        """Test convergence plot creation."""
        # Mock results structure
        all_results = {
            'no_attack': {
                'tavs': MagicMock(),
                'full_verification': MagicMock(),
                'comparison': {'efficiency_improvement': 1.05}
            },
            'meta': {},
            'overall_comparison': {}
        }

        # Should not raise any exceptions
        self.experiment._create_convergence_plots(all_results)

        # Verify plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

        # Check save path
        save_path = mock_savefig.call_args[0][0]
        self.assertIn('model_convergence_analysis.png', str(save_path))

    def test_convergence_metrics_comparison(self):
        """Test comparison of convergence metrics between strategies."""
        tavs_losses = self.experiment._extract_actual_losses('no_attack', 'tavs')
        full_losses = self.experiment._extract_actual_losses('no_attack', 'full_verification')

        tavs_acc = self.experiment._extract_actual_accuracies('no_attack', 'tavs')
        full_acc = self.experiment._extract_actual_accuracies('no_attack', 'full_verification')

        # Both should have same number of rounds
        self.assertEqual(len(tavs_losses), len(full_losses))
        self.assertEqual(len(tavs_acc), len(full_acc))

        # Both should show learning (loss decreasing, accuracy increasing)
        self.assertLess(tavs_losses[-1], tavs_losses[0])
        self.assertLess(full_losses[-1], full_losses[0])
        self.assertGreater(tavs_acc[-1], tavs_acc[0])
        self.assertGreater(full_acc[-1], full_acc[0])

    def test_evaluate_function_creation(self):
        """Test the evaluate function creation in pipeline."""
        from src.tavs.end_to_end_pipeline import TAVSESPPipeline, PipelineConfig

        # Create minimal pipeline config
        pipeline_config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            output_dir=str(Path(self.temp_dir) / "test_pipeline")
        )

        pipeline = TAVSESPPipeline(pipeline_config)

        # Test evaluate function creation
        eval_fn = pipeline._create_evaluate_function()
        self.assertIsNotNone(eval_fn)
        self.assertTrue(callable(eval_fn))

    def test_convergence_metrics_integration(self):
        """Integration test for convergence metrics in full pipeline."""
        # This would be a more comprehensive test that runs a mini experiment
        # and validates that actual metrics are captured

        # Create mock all_results with actual extracted metrics
        all_results = {
            'no_attack': {
                'tavs': MagicMock(),
                'full_verification': MagicMock()
            }
        }

        # Test that metrics can be extracted
        tavs_losses = self.experiment._extract_actual_losses('no_attack', 'tavs')
        tavs_acc = self.experiment._extract_actual_accuracies('no_attack', 'tavs')

        # Verify we get the expected mock data
        self.assertIsInstance(tavs_losses, list)
        self.assertIsInstance(tavs_acc, list)

        if tavs_losses and tavs_acc:
            # Verify convergence properties
            self.assertEqual(len(tavs_losses), len(tavs_acc))
            self.assertTrue(all(isinstance(x, (int, float)) for x in tavs_losses))
            self.assertTrue(all(isinstance(x, (int, float)) for x in tavs_acc))


if __name__ == '__main__':
    unittest.main()