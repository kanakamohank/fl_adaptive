#!/usr/bin/env python3
"""
TAVS Flower Client Integration

This module provides a Flower-compatible client wrapper that integrates with
the TAVS-ESP system while maintaining compatibility with existing honest
and attacker client implementations.

Core Integration:
- Wraps existing HonestClient and attacker implementations
- Handles Flower serialization and communication protocols
- Processes TAVS assignment messages (verified vs promoted)
- Maintains attack coordination capabilities for research validation

Key Innovation: Seamless integration between TAVS trust-adaptive system
and existing FL client implementations without breaking compatibility.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
from dataclasses import dataclass

# Flower imports
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

# TAVS-ESP imports
from .honest_client import HonestClient
from ..attacks.null_space_attack import NullSpaceAttacker
from ..attacks.layerwise_attacks import LayerwiseBackdoorAttacker, DistributedPoisonAttacker

logger = logging.getLogger(__name__)


@dataclass
class TAVSClientConfig:
    """Configuration for TAVS Flower client."""
    client_id: str
    client_type: str  # "honest", "null_space", "layerwise", "distributed"
    model_type: str = "cifar_cnn"
    model_kwargs: Dict[str, Any] = None
    device: str = "cpu"

    # Attack-specific parameters
    attack_intensity: float = 1.0
    target_fraction: float = 0.001  # For layerwise attacks

    # Training parameters
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01


class TAVSFlowerClient(NumPyClient):
    """
    TAVS-compatible Flower client wrapper.

    This client acts as a bridge between the Flower federated learning framework
    and our existing client implementations (honest clients and attackers).

    Key Features:
    - Delegates training to underlying client implementation
    - Processes TAVS assignment messages from server
    - Maintains attack coordination for Byzantine clients
    - Handles parameter serialization for Flower communication
    """

    def __init__(self,
                 config: TAVSClientConfig,
                 train_loader = None,
                 test_loader = None):
        """
        Initialize TAVS Flower client.

        Args:
            config: Client configuration including type and parameters
            train_loader: Training data loader
            test_loader: Test data loader (optional)
        """
        super().__init__()

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize underlying client based on type
        self.underlying_client = self._create_underlying_client()

        # TAVS state tracking
        self.current_assignment = "verified"  # "verified" or "promoted"
        self.trust_score = 0.5  # Current trust score from server
        self.tier = 1  # Current tier assignment
        self.round_number = 0
        self.is_decoy = False  # Whether this is a decoy verification

        # Performance tracking
        self.training_history = []
        self.assignment_history = []

        logger.info(f"TAVS Flower Client initialized: {config.client_id} ({config.client_type})")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as NumPy arrays."""
        try:
            # Delegate to underlying client
            if hasattr(self.underlying_client, 'get_parameters'):
                params = self.underlying_client.get_parameters(config)
            else:
                # Fallback: get parameters from model
                params = self.underlying_client.model.get_weights_flat()

            # Convert to list of NumPy arrays for Flower
            if isinstance(params, torch.Tensor):
                # Handle single flattened tensor
                param_arrays = [params.detach().cpu().numpy()]
            elif isinstance(params, list):
                # Handle list of tensors
                param_arrays = []
                for param in params:
                    if isinstance(param, torch.Tensor):
                        param_arrays.append(param.detach().cpu().numpy())
                    elif isinstance(param, np.ndarray):
                        param_arrays.append(param)
                    else:
                        param_arrays.append(np.array(param))
            else:
                # Handle numpy array
                param_arrays = [np.array(params)]

            logger.debug(f"Client {self.config.client_id}: Retrieved {len(param_arrays)} parameter arrays")
            return param_arrays

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Error getting parameters: {e}")
            # Return dummy parameters to prevent crash
            return [np.array([0.0])]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        try:
            # Convert NumPy arrays back to appropriate format for underlying client
            if hasattr(self.underlying_client, 'set_parameters'):
                # HonestClient expects list of numpy arrays
                self.underlying_client.set_parameters(parameters)
            else:
                # Fallback: set parameters directly on model
                if len(parameters) == 1:
                    param_tensor = torch.tensor(parameters[0], dtype=torch.float32)
                    self.underlying_client.model.set_weights_flat(param_tensor)
                else:
                    # Concatenate multiple arrays if needed
                    all_params = np.concatenate([p.flatten() for p in parameters])
                    param_tensor = torch.tensor(all_params, dtype=torch.float32)
                    self.underlying_client.model.set_weights_flat(param_tensor)

            logger.debug(f"Client {self.config.client_id}: Set {len(parameters)} parameter arrays")

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Error setting parameters: {e}")

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model and return updated parameters.

        This method processes TAVS assignments and delegates training to the
        underlying client implementation.
        """
        try:
            # DEBUG: Log incoming parameters from Flower
            logger.debug(f"Client {self.config.client_id}: FLOWER INPUT - "
                        f"parameters type: {type(parameters)}, length: {len(parameters) if hasattr(parameters, '__len__') else 'N/A'}")
            if hasattr(parameters, '__len__') and len(parameters) > 0:
                for i, param in enumerate(parameters):
                    logger.debug(f"  Param {i}: type={type(param)}, shape={getattr(param, 'shape', 'N/A')}, dtype={getattr(param, 'dtype', 'N/A')}")

            # Update TAVS state from server config
            self._process_tavs_config(config)

            # Set initial parameters
            self.set_parameters(parameters)

            # Execute training based on client type and assignment
            num_examples = self._execute_training(config)

            # Get updated parameters
            updated_parameters = self.get_parameters(config)

            # Prepare metrics for server
            metrics = self._prepare_fit_metrics(num_examples)

            logger.info(f"Client {self.config.client_id}: Training complete "
                       f"(round {self.round_number}, {self.current_assignment}, "
                       f"trust={self.trust_score:.3f})")

            return updated_parameters, num_examples, metrics

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Training failed: {e}")
            # Return original parameters to prevent crash
            return parameters, 0, {"error": str(e)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model and return metrics.

        Optional method for client-side evaluation.
        """
        try:
            if self.test_loader is None:
                return 0.0, 0, {"accuracy": 0.0}

            # Set parameters for evaluation
            self.set_parameters(parameters)

            # Delegate evaluation to underlying client
            if hasattr(self.underlying_client, 'evaluate'):
                loss, num_examples, eval_metrics = self.underlying_client.evaluate(parameters, config)
                accuracy = eval_metrics.get('accuracy', 0.0)
            else:
                # Simple evaluation fallback
                loss = 0.0
                accuracy = 0.0
                num_examples = len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else 100

            metrics = {
                "accuracy": accuracy,
                "client_id": self.config.client_id,
                "client_type": self.config.client_type
            }

            logger.debug(f"Client {self.config.client_id}: Evaluation - "
                        f"loss={loss:.4f}, accuracy={accuracy:.3f}")

            return float(loss), int(num_examples), metrics

        except Exception as e:
            logger.error(f"Client {self.config.client_id}: Evaluation failed: {e}")
            return 0.0, 0, {"error": str(e)}

    def _create_underlying_client(self):
        """Create the underlying client implementation based on configuration."""
        model_kwargs = self.config.model_kwargs or {"num_classes": 10}

        if self.config.client_type == "honest":
            # For honest clients, create directly using HonestClient constructor
            from .honest_client import HonestClient
            return HonestClient(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                device=self.config.device
            )

        elif self.config.client_type == "null_space":
            return NullSpaceAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                attack_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        elif self.config.client_type == "layerwise":
            # Use target layers based on target_fraction
            target_layers = ["fc1"] if self.config.target_fraction > 0 else []
            return LayerwiseBackdoorAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                target_layers=target_layers,
                attack_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        elif self.config.client_type == "distributed":
            return DistributedPoisonAttacker(
                client_id=self.config.client_id,
                model_type=self.config.model_type,
                model_kwargs=model_kwargs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                poison_intensity=self.config.attack_intensity,
                device=self.config.device
            )

        else:
            raise ValueError(f"Unknown client type: {self.config.client_type}")

    def _process_tavs_config(self, config: Dict[str, Scalar]):
        """Process TAVS-specific configuration from server."""
        if "round" in config:
            self.round_number = int(config["round"])

        if "tavs_assignment" in config:
            self.current_assignment = str(config["tavs_assignment"])

        if "trust_score" in config:
            self.trust_score = float(config["trust_score"])

        if "tier" in config:
            self.tier = int(config["tier"])

        if "is_decoy" in config:
            self.is_decoy = bool(config["is_decoy"])

        # Store assignment history
        self.assignment_history.append({
            "round": self.round_number,
            "assignment": self.current_assignment,
            "trust_score": self.trust_score,
            "tier": self.tier,
            "is_decoy": self.is_decoy
        })

        logger.debug(f"Client {self.config.client_id}: TAVS assignment - "
                    f"{self.current_assignment} (tier {self.tier}, trust={self.trust_score:.3f})")

    def _execute_training(self, config: Dict[str, Scalar]) -> int:
        """Execute training based on client type and TAVS assignment."""
        # Convert config to format expected by underlying client
        training_config = {
            "round": self.round_number,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate
        }

        # Get initial parameters in numpy format
        initial_params = self.get_parameters(config)

        # DEBUG: Log parameters being passed to underlying client
        logger.debug(f"Client {self.config.client_id}: UNDERLYING CLIENT INPUT - "
                    f"initial_params type: {type(initial_params)}, length: {len(initial_params) if hasattr(initial_params, '__len__') else 'N/A'}")
        if hasattr(initial_params, '__len__') and len(initial_params) > 0:
            for i, param in enumerate(initial_params):
                logger.debug(f"  Underlying param {i}: type={type(param)}, shape={getattr(param, 'shape', 'N/A')}, dtype={getattr(param, 'dtype', 'N/A')}")

        # Execute training through underlying client
        if self.config.client_type == "honest":
            # Honest clients train normally regardless of assignment
            trained_params, num_examples, metrics = self.underlying_client.fit(
                initial_params, training_config
            )

        else:
            # Attack clients may modify behavior based on assignment
            if self.current_assignment == "verified" or self.is_decoy:
                # If verified or decoy, attackers may behave honestly to avoid detection
                if hasattr(self.underlying_client, 'behave_honestly'):
                    trained_params, num_examples, metrics = self.underlying_client.behave_honestly(
                        initial_params, training_config
                    )
                else:
                    # Fallback: train normally but with reduced intensity
                    trained_params, num_examples, metrics = self.underlying_client.fit(
                        initial_params, training_config
                    )
            else:
                # If promoted, execute full attack
                trained_params, num_examples, metrics = self.underlying_client.fit(
                    initial_params, training_config
                )

        # Update our parameters
        self.set_parameters(trained_params)

        # Store training history
        self.training_history.append({
            "round": self.round_number,
            "assignment": self.current_assignment,
            "num_examples": num_examples,
            "metrics": metrics
        })

        return num_examples

    def _prepare_fit_metrics(self, num_examples: int) -> Dict[str, Scalar]:
        """Prepare metrics to send back to server."""
        metrics = {
            "client_id": self.config.client_id,
            "client_type": self.config.client_type,
            "tavs_assignment": self.current_assignment,
            "trust_score": self.trust_score,
            "tier": self.tier,
            "round": self.round_number,
            "num_examples": num_examples
        }

        # Add attack-specific metrics if applicable
        if self.config.client_type != "honest":
            metrics["attack_intensity"] = self.config.attack_intensity

            if self.config.client_type == "layerwise":
                metrics["target_fraction"] = self.config.target_fraction

        return metrics

    def get_tavs_statistics(self) -> Dict[str, Any]:
        """Get comprehensive TAVS client statistics."""
        return {
            "client_id": self.config.client_id,
            "client_type": self.config.client_type,
            "current_trust_score": self.trust_score,
            "current_tier": self.tier,
            "current_assignment": self.current_assignment,
            "total_rounds": len(self.assignment_history),
            "verification_count": sum(1 for h in self.assignment_history if h["assignment"] == "verified"),
            "promotion_count": sum(1 for h in self.assignment_history if h["assignment"] == "promoted"),
            "decoy_count": sum(1 for h in self.assignment_history if h.get("is_decoy", False)),
            "assignment_history": self.assignment_history[-10:],  # Last 10 rounds
            "training_history": self.training_history[-5:]  # Last 5 rounds
        }


def create_tavs_flower_client(config: TAVSClientConfig,
                             train_loader = None,
                             test_loader = None) -> TAVSFlowerClient:
    """
    Factory function to create TAVS Flower clients.

    Args:
        config: Client configuration
        train_loader: Training data loader
        test_loader: Test data loader (optional)

    Returns:
        Configured TAVSFlowerClient instance
    """
    return TAVSFlowerClient(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader
    )