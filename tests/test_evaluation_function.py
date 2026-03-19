#!/usr/bin/env python3
"""
Tests for evaluation function integration in TAVS pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tavs.end_to_end_pipeline import TAVSESPPipeline, PipelineConfig
from src.tavs.tavs_esp_strategy import TavsEspConfig


class TestEvaluationFunction(unittest.TestCase):
    """Test cases for evaluation function integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_evaluate_function_creation(self):
        """Test that evaluation function is created correctly."""
        config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            output_dir=str(Path(self.temp_dir) / "test_eval")
        )

        pipeline = TAVSESPPipeline(config)
        eval_fn = pipeline._create_evaluate_function()

        self.assertIsNotNone(eval_fn)
        self.assertTrue(callable(eval_fn))

    def test_evaluate_function_signature(self):
        """Test evaluation function has correct signature."""
        config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            output_dir=str(Path(self.temp_dir) / "test_sig")
        )

        pipeline = TAVSESPPipeline(config)
        eval_fn = pipeline._create_evaluate_function()

        # Test function signature by checking if it accepts expected parameters
        import inspect
        sig = inspect.signature(eval_fn)
        param_names = list(sig.parameters.keys())

        self.assertIn('server_round', param_names)
        self.assertIn('parameters_ndarrays', param_names)
        self.assertIn('config_dict', param_names)

    @patch('src.utils.data_utils.load_cifar10')
    @patch('src.core.models.get_model')
    def test_evaluate_function_execution(self, mock_get_model, mock_load_cifar10):
        """Test evaluation function execution with mocked dependencies."""
        import torch
        import numpy as np

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([[1.0, 2.0]]), torch.tensor([0.5])]
        mock_model.eval.return_value = None
        mock_get_model.return_value = mock_model

        # Mock data
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_load_cifar10.return_value = (None, mock_dataset)

        # Mock DataLoader and model output
        with patch('torch.utils.data.Subset') as mock_subset, \
             patch('torch.utils.data.DataLoader') as mock_dataloader:

            mock_subset.return_value = mock_dataset
            mock_loader = MagicMock()
            mock_loader.__iter__.return_value = iter([
                (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
            ])
            mock_loader.__len__.return_value = 1
            mock_dataloader.return_value = mock_loader

            # Mock model output
            mock_output = torch.randn(32, 10)
            mock_model.return_value = mock_output

            # Create and test evaluation function
            config = PipelineConfig(
                num_rounds=2,
                num_clients=5,
                clients_per_round=3,
                byzantine_fraction=0.0,
                model_type="test_model",
                output_dir=str(Path(self.temp_dir) / "test_exec")
            )

            pipeline = TAVSESPPipeline(config)
            eval_fn = pipeline._create_evaluate_function()

            # Test execution
            parameters = [np.array([[1.0, 2.0]]), np.array([0.5])]
            loss, metrics = eval_fn(1, parameters, {})

            self.assertIsInstance(loss, float)
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('correct', metrics)
            self.assertIn('total', metrics)

    def test_strategy_config_with_evaluate_fn(self):
        """Test that strategy gets evaluate_fn when it's None."""
        # Create config with None evaluate_fn
        tavs_config = TavsEspConfig(evaluate_fn=None)

        config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            tavs_config=tavs_config,
            output_dir=str(Path(self.temp_dir) / "test_strategy")
        )

        pipeline = TAVSESPPipeline(config)

        # This should not raise an error and should set evaluate_fn
        strategy = pipeline.create_server_strategy()

        self.assertIsNotNone(strategy.config.evaluate_fn)
        self.assertTrue(callable(strategy.config.evaluate_fn))

    def test_evaluate_function_error_handling(self):
        """Test evaluation function error handling."""
        config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            output_dir=str(Path(self.temp_dir) / "test_error")
        )

        pipeline = TAVSESPPipeline(config)
        eval_fn = pipeline._create_evaluate_function()

        # Test with invalid parameters (should return dummy metrics)
        loss, metrics = eval_fn(1, [], {})

        self.assertIsInstance(loss, float)
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)

        # Loss should be reasonable dummy value
        self.assertGreater(loss, 0)

    @patch('src.tavs.end_to_end_pipeline.logger')
    def test_evaluation_logging(self, mock_logger):
        """Test that evaluation function logs correctly."""
        config = PipelineConfig(
            num_rounds=2,
            num_clients=5,
            clients_per_round=3,
            byzantine_fraction=0.0,
            output_dir=str(Path(self.temp_dir) / "test_log")
        )

        pipeline = TAVSESPPipeline(config)
        eval_fn = pipeline._create_evaluate_function()

        # Test with invalid parameters to trigger error path
        eval_fn(1, [], {})

        # Should log warning about evaluation failure
        mock_logger.warning.assert_called()


if __name__ == '__main__':
    unittest.main()