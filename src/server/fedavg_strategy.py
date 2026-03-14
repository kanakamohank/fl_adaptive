import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from functools import reduce
import torch
from ..core.models import get_model
import logging


logger = logging.getLogger(__name__)


class FedAvgStrategy(Strategy):
    """Federated Averaging strategy implementation."""

    def __init__(
        self,
        model_type: str,
        model_kwargs: Dict,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        super().__init__()

        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        # Initialize global model for parameter initialization
        self.global_model = get_model(model_type, **model_kwargs)

        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = ndarrays_to_parameters(
                [param.cpu().detach().numpy() for param in self.global_model.parameters()]
            )

        # Track training history
        self.fit_metrics_history = []
        self.evaluate_metrics_history = []
        self.round_number = 0

    def __repr__(self) -> str:
        return f"FedAvgStrategy(model_type={self.model_type})"

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        logger.info("Initializing global model parameters")
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        logger.info(f"Configuring fit for round {server_round}")

        # Sample clients
        sample_size = max(int(len(client_manager.all()) * self.fraction_fit), self.min_fit_clients)
        sampled_clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        # Create fit configuration
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["server_round"] = server_round

        # Create fit instructions
        fit_ins = FitIns(parameters, config)

        # Return client/config pairs
        return [(client, fit_ins) for client in sampled_clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        logger.info(f"Configuring evaluation for round {server_round}")

        # Do not configure federated evaluation if fraction eval is 0
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients
        sample_size = max(int(len(client_manager.all()) * self.fraction_evaluate), self.min_evaluate_clients)
        sampled_clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        # Create evaluation configuration
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["server_round"] = server_round

        # Create evaluation instructions
        evaluate_ins = EvaluateIns(parameters, config)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in sampled_clients]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using FedAvg."""
        logger.info(f"Aggregating fit results for round {server_round}")

        if not results:
            logger.warning("No fit results to aggregate")
            return None, {}

        # Handle failures if not accepted
        if not self.accept_failures and failures:
            logger.error(f"Training failures occurred: {failures}")
            return None, {}

        # Extract parameters and metrics
        weights_results = []
        metrics_results = []
        total_examples = 0

        for client_proxy, fit_res in results:
            # Convert parameters to numpy arrays
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples

            weights_results.append((client_weights, num_examples))
            metrics_results.append((num_examples, fit_res.metrics))
            total_examples += num_examples

        # Perform weighted averaging
        aggregated_weights = self._aggregate_weights(weights_results)

        # Convert back to parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)

        # Aggregate metrics
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_results)
        else:
            aggregated_metrics = self._default_fit_metrics_aggregation(metrics_results)

        # Store history
        self.fit_metrics_history.append({
            "round": server_round,
            "total_examples": total_examples,
            "num_clients": len(results),
            "metrics": aggregated_metrics
        })

        logger.info(f"Round {server_round} aggregation complete. Total examples: {total_examples}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        logger.info(f"Aggregating evaluation results for round {server_round}")

        if not results:
            logger.warning("No evaluation results to aggregate")
            return None, {}

        # Handle failures if not accepted
        if not self.accept_failures and failures:
            logger.error(f"Evaluation failures occurred: {failures}")
            return None, {}

        # Extract metrics
        metrics_results = []
        total_examples = 0
        weighted_loss = 0.0

        for client_proxy, evaluate_res in results:
            num_examples = evaluate_res.num_examples
            loss = evaluate_res.loss

            metrics_results.append((num_examples, evaluate_res.metrics))
            weighted_loss += loss * num_examples
            total_examples += num_examples

        # Calculate aggregate loss
        aggregate_loss = weighted_loss / total_examples if total_examples > 0 else 0.0

        # Aggregate metrics
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(metrics_results)
        else:
            aggregated_metrics = self._default_evaluate_metrics_aggregation(metrics_results)

        # Store history
        self.evaluate_metrics_history.append({
            "round": server_round,
            "loss": aggregate_loss,
            "total_examples": total_examples,
            "num_clients": len(results),
            "metrics": aggregated_metrics
        })

        logger.info(f"Round {server_round} evaluation complete. Aggregate loss: {aggregate_loss:.4f}")

        return aggregate_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters."""
        if self.evaluate_fn is None:
            return None

        logger.info(f"Server-side evaluation for round {server_round}")

        # Convert parameters to model weights
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        # Evaluate using the provided function
        loss, metrics = self.evaluate_fn(server_round, parameters_ndarrays, {})

        return loss, metrics

    def _aggregate_weights(self, weights_results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Aggregate weights using weighted averaging (FedAvg)."""
        # Calculate total number of examples
        total_examples = sum(num_examples for _, num_examples in weights_results)

        if total_examples == 0:
            logger.warning("Total examples is 0, cannot aggregate weights")
            return weights_results[0][0]  # Return first client's weights as fallback

        # Initialize aggregated weights with zeros
        first_weights, _ = weights_results[0]
        aggregated_weights = [np.zeros_like(weight) for weight in first_weights]

        # Weighted aggregation
        for weights, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer_weights in enumerate(weights):
                aggregated_weights[i] += weight * layer_weights

        return aggregated_weights

    def _default_fit_metrics_aggregation(self, metrics_results: List[Tuple[int, Dict]]) -> Dict[str, Scalar]:
        """Default aggregation for fit metrics."""
        total_examples = sum(num_examples for num_examples, _ in metrics_results)

        if total_examples == 0:
            return {}

        # Aggregate losses
        weighted_loss = 0.0
        for num_examples, metrics in metrics_results:
            if "final_loss" in metrics:
                weighted_loss += metrics["final_loss"] * num_examples

        return {
            "train_loss": weighted_loss / total_examples,
            "total_examples": total_examples,
            "num_clients": len(metrics_results),
        }

    def _default_evaluate_metrics_aggregation(self, metrics_results: List[Tuple[int, Dict]]) -> Dict[str, Scalar]:
        """Default aggregation for evaluation metrics."""
        total_examples = sum(num_examples for num_examples, _ in metrics_results)

        if total_examples == 0:
            return {}

        # Aggregate accuracy
        total_correct = 0
        for num_examples, metrics in metrics_results:
            if "correct" in metrics:
                total_correct += metrics["correct"]

        accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "num_clients": len(metrics_results),
        }

    def get_training_history(self) -> Dict:
        """Get complete training history."""
        return {
            "fit_metrics": self.fit_metrics_history,
            "evaluate_metrics": self.evaluate_metrics_history
        }