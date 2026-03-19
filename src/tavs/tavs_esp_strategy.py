#!/usr/bin/env python3
"""
TAVS-ESP Federated Learning Strategy

This module implements the complete TAVS-ESP federated learning strategy that
integrates with the Flower framework. It combines Layer 1 (TAVS trust-adaptive
scheduling) with Layer 2 (ESP ephemeral structured projections).

Core Integration:
- configure_fit(): Execute TAVS Layer 1 scheduling using CSPRNG-derived assignments
- aggregate_fit(): Execute ESP Layer 2 projections, verification, and trust updates
- Unified aggregation: [Σ_inliers g_i + Σ_promoted p_i(r)·g_i] / Z(r)

Key Innovation: Co-designed two-layer framework enabling Byzantine-robust
federated learning at LLM scale with O(f_∞·Nk) complexity.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import time
import json
from pathlib import Path

# Flower imports
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    FitRes, FitIns, EvaluateIns, EvaluateRes, Parameters, Scalar, NDArrays,
    parameters_to_ndarrays, ndarrays_to_parameters
)

# TAVS-ESP imports
from .csprng_manager import CSPRNGManager, RoundMaterials
from .scheduler import TavsScheduler, SchedulingDecision, ClientTrustState
from ..core.projection import StructuredJLProjection, DenseJLProjection
from ..core.verification import IsomorphicVerification
from ..core.models import ModelStructure

logger = logging.getLogger(__name__)


@dataclass
class TavsEspConfig:
    """Configuration for TAVS-ESP strategy."""

    # TAVS Layer 1 Configuration
    theta_low: float = 0.3              # Lower threshold for tier classification
    theta_high: float = 0.7             # Upper threshold for tier classification
    alpha: float = 0.9                  # EMA decay factor for trust updates
    gamma_budget: float = 0.35          # Budget constraint (max promoted weight fraction)
    tau_ramp: int = 30                  # Trust ramp-up parameter (Sybil resistance)
    decoy_probability: float = 0.15     # Probability of decoy verification
    c_lambda: float = 4.0               # Sigmoid slope for Bayesian weights

    # ESP Layer 2 Configuration
    target_k: int = 150                 # JL projection target dimension (independent of parameter count)
    detection_threshold: float = 2.0    # Byzantine detection threshold
    min_consensus: float = 0.6          # Minimum consensus for detection
    projection_type: str = "structured" # "structured" or "dense"

    # System Configuration
    min_fit_clients: int = 2            # Minimum clients for training
    min_available_clients: int = 2      # Minimum available clients
    evaluate_fn: Optional[callable] = None  # Server-side evaluation function

    # Security Configuration
    master_key: Optional[bytes] = None  # CSPRNG master key (if None, generates new)
    key_rotation_rounds: int = 10000    # Rotate master key every N rounds

    # Logging and Analysis
    save_trust_trajectory: bool = True  # Save trust score evolution
    save_round_decisions: bool = True   # Save scheduling decisions
    output_dir: str = "tavs_esp_logs"   # Directory for logs and analysis


@dataclass
class RoundAnalytics:
    """Analytics data for a single federated learning round."""
    round_number: int
    scheduling_decision: SchedulingDecision
    projection_time_ms: float
    detection_time_ms: float
    aggregation_time_ms: float
    trust_updates: Dict[str, float]
    byzantine_detected: List[str]
    consensus_achieved: bool
    model_accuracy: Optional[float] = None


class TavsEspStrategy(Strategy):
    """
    Complete TAVS-ESP federated learning strategy.

    Implements Byzantine-robust federated learning using:
    1. Layer 1 (TAVS): Trust-adaptive scheduling with CSPRNG security
    2. Layer 2 (ESP): Ephemeral structured projections with block-variance detection
    3. Unified aggregation combining verified inliers and Bayesian-weighted promoted clients
    """

    def __init__(self,
                 config: TavsEspConfig,
                 model_structure: Optional[ModelStructure] = None,
                 initial_parameters: Optional[Parameters] = None):
        """
        Initialize TAVS-ESP strategy.

        Args:
            config: TAVS-ESP configuration parameters
            model_structure: Model structure for semantic block identification
            initial_parameters: Initial model parameters
        """
        super().__init__()

        self.config = config
        self.model_structure = model_structure
        self.current_parameters = initial_parameters

        # Initialize core components
        self.csprng_manager = CSPRNGManager(
            master_key=config.master_key,
            key_rotation_rounds=config.key_rotation_rounds
        )

        self.scheduler = TavsScheduler(
            csprng_manager=self.csprng_manager,
            theta_low=config.theta_low,
            theta_high=config.theta_high,
            alpha=config.alpha,
            gamma_budget=config.gamma_budget,
            tau_ramp=config.tau_ramp,
            decoy_probability=config.decoy_probability
        )

        self.verification = IsomorphicVerification(
            detection_threshold=config.detection_threshold,
            min_consensus=config.min_consensus
        )

        # Initialize projection system
        self.projection = None  # Will be initialized when model structure is available

        # Analytics and logging
        self.round_analytics: List[RoundAnalytics] = []
        self.round_number = 0

        # Setup logging directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TAVS-ESP Strategy initialized: {config.projection_type} projection, "
                   f"budget={config.gamma_budget}, detection_threshold={config.detection_threshold}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.current_parameters

    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure clients for training round (TAVS Layer 1).

        Executes trust-adaptive scheduling using CSPRNG-derived promotion assignments.

        Args:
            server_round: Current federated learning round
            parameters: Current global model parameters
            client_manager: Flower client manager

        Returns:
            List of (client, FitIns) tuples with TAVS assignments
        """
        self.round_number = server_round
        self.current_parameters = parameters

        # Get available clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        available_clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if len(available_clients) < self.config.min_fit_clients:
            logger.warning(f"Insufficient clients: {len(available_clients)} < {self.config.min_fit_clients}")
            return []

        # Extract client IDs
        client_ids = [proxy.cid for proxy in available_clients]

        # Generate TAVS scheduling decision (Layer 1)
        start_time = time.time()
        scheduling_decision = self.scheduler.generate_scheduling_decision(
            candidate_clients=client_ids,
            round_number=server_round
        )
        scheduling_time = (time.time() - start_time) * 1000

        logger.info(f"Round {server_round} TAVS scheduling: "
                   f"{len(scheduling_decision.verified_clients)} verified, "
                   f"{len(scheduling_decision.promoted_clients)} promoted, "
                   f"budget: {scheduling_decision.budget_utilization:.1%}")

        # Create FitIns configurations
        fit_configurations = []

        for proxy in available_clients:
            # Determine client assignment
            is_verified = proxy.cid in scheduling_decision.verified_clients
            is_promoted = proxy.cid in scheduling_decision.promoted_clients
            is_decoy = proxy.cid in scheduling_decision.decoy_clients

            # Create configuration for this client
            fit_config = {
                "round": server_round,
                "tavs_assignment": "verified" if is_verified else "promoted",
                "is_decoy": is_decoy,
                "trust_score": scheduling_decision.trust_scores.get(proxy.cid, 0.5),
                "tier": scheduling_decision.tier_assignments.get(proxy.cid, 1)
            }

            fit_ins = FitIns(parameters, fit_config)
            fit_configurations.append((proxy, fit_ins))

        logger.debug(f"Configured {len(fit_configurations)} clients for training "
                    f"(scheduling took {scheduling_time:.1f}ms)")

        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates (ESP Layer 2).

        Executes ephemeral structured projections, Byzantine detection, and unified aggregation.

        Args:
            server_round: Current federated learning round
            results: List of (client, FitRes) tuples
            failures: List of failed client results

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            logger.warning(f"No client results to aggregate in round {server_round}")
            return None, {}

        start_time = time.time()

        # Extract client updates and metadata
        client_updates = []
        client_ids = []
        client_assignments = {}

        for proxy, fit_res in results:
            # Convert parameters to numpy arrays
            client_params = parameters_to_ndarrays(fit_res.parameters)
            current_params = parameters_to_ndarrays(self.current_parameters)

            # Compute parameter update (gradient)
            update_arrays = []
            for client_param, current_param in zip(client_params, current_params):
                update = client_param - current_param
                update_arrays.append(update)

            # Flatten update to single vector
            update_flat = np.concatenate([arr.flatten() for arr in update_arrays])
            client_updates.append(torch.tensor(update_flat, dtype=torch.float32))

            client_ids.append(proxy.cid)
            client_assignments[proxy.cid] = fit_res.metrics.get("tavs_assignment", "verified")

        # Initialize projection system if needed
        if self.projection is None:
            self._initialize_projection_system(client_updates[0].shape[0])

        # Generate ephemeral projection matrices (ESP Layer 2)
        projection_start = time.time()
        round_materials = self.csprng_manager.derive_round_materials(server_round)

        if self.config.projection_type == "structured" and self.model_structure is not None:
            # Use structured block-diagonal projection
            projection_matrices = self.projection.generate_ephemeral_projection_matrix(
                round_number=server_round
            )
            # Project each update through structured projection
            projected_updates = []
            for update in client_updates:
                projected = self.projection.project_update(update, projection_matrices)
                projected_updates.append(projected)
        else:
            # Use dense projection
            projection_matrix = self.projection.generate_projection_matrix(round_number=server_round)
            projected_updates = []
            for update in client_updates:
                projected = torch.tensor(projection_matrix @ update.numpy(), dtype=torch.float32)
                projected_updates.append(projected)

        projection_time = (time.time() - projection_start) * 1000

        # Byzantine detection on projected updates
        detection_start = time.time()
        detection_results = self.verification.detect_byzantine_clients(
            projected_updates=projected_updates,
            client_ids=client_ids
        )
        detection_time = (time.time() - detection_start) * 1000

        byzantine_indices = set(detection_results['byzantine_indices'])
        byzantine_clients = [client_ids[i] for i in byzantine_indices]
        inlier_indices = [i for i in range(len(client_ids)) if i not in byzantine_indices]
        inlier_clients = [client_ids[i] for i in inlier_indices]

        logger.info(f"Round {server_round} ESP detection: "
                   f"{len(byzantine_clients)} Byzantine, {len(inlier_clients)} inliers, "
                   f"consensus: {detection_results['consensus_achieved']}")

        # Separate verified and promoted clients
        verified_clients = []
        promoted_clients = []

        for i, client_id in enumerate(client_ids):
            if i in inlier_indices:  # Only consider inliers for aggregation
                assignment = client_assignments[client_id]
                if assignment == "verified":
                    verified_clients.append((i, client_id))
                else:  # promoted
                    promoted_clients.append((i, client_id))

        # Unified aggregation: [Σ_inliers g_i + Σ_promoted p_i(r)·g_i] / Z(r)
        aggregation_start = time.time()

        if len(verified_clients) + len(promoted_clients) == 0:
            logger.error("No inlier clients available for aggregation!")
            return None, {}

        # Aggregate verified client updates (weight = 1.0)
        verified_sum = None
        verified_weight = 0.0

        for idx, client_id in verified_clients:
            if verified_sum is None:
                verified_sum = client_updates[idx].clone()
            else:
                verified_sum += client_updates[idx]
            verified_weight += 1.0

        # Aggregate promoted client updates with Bayesian weights
        promoted_sum = None
        promoted_weight = 0.0

        if promoted_clients:
            promoted_client_ids = [cid for _, cid in promoted_clients]
            bayesian_weights = self.scheduler.compute_bayesian_weights(
                promoted_clients=promoted_client_ids,
                c_lambda=self.config.c_lambda
            )

            for idx, client_id in promoted_clients:
                weight = bayesian_weights.get(client_id, 0.1)

                if promoted_sum is None:
                    promoted_sum = client_updates[idx] * weight
                else:
                    promoted_sum += client_updates[idx] * weight
                promoted_weight += weight

        # Combine verified and promoted updates
        total_weight = verified_weight + promoted_weight

        if verified_sum is not None and promoted_sum is not None:
            aggregated_update = (verified_sum + promoted_sum) / total_weight
        elif verified_sum is not None:
            aggregated_update = verified_sum / verified_weight
        elif promoted_sum is not None:
            aggregated_update = promoted_sum / promoted_weight
        else:
            logger.error("No valid updates to aggregate!")
            return None, {}

        # Apply aggregated update to current parameters
        current_params_arrays = parameters_to_ndarrays(self.current_parameters)
        aggregated_params_arrays = []

        # Unflatten and apply update
        idx = 0
        for param_array in current_params_arrays:
            param_size = param_array.size
            param_update = aggregated_update[idx:idx + param_size].numpy().reshape(param_array.shape)
            updated_param = param_array + param_update
            aggregated_params_arrays.append(updated_param)
            idx += param_size

        aggregated_parameters = ndarrays_to_parameters(aggregated_params_arrays)
        aggregation_time = (time.time() - aggregation_start) * 1000

        # Update trust scores based on verification results
        verification_results = {}
        for i, client_id in enumerate(client_ids):
            if client_assignments[client_id] == "verified":
                # Compute behavioral score based on detection results
                if i in byzantine_indices:
                    behavioral_score = 0.2  # Poor score for detected Byzantine
                else:
                    behavioral_score = 0.8  # Good score for honest behavior
                verification_results[client_id] = behavioral_score

        promoted_client_list = [client_assignments[cid] == "promoted" for cid in client_ids]
        promoted_client_ids = [cid for cid, is_promoted in zip(client_ids, promoted_client_list) if is_promoted]

        trust_updates = self.scheduler.update_trust_scores(
            verification_results=verification_results,
            promoted_clients=promoted_client_ids,
            round_number=server_round
        )

        # Create round analytics
        scheduling_decision = self.scheduler.scheduling_history[-1] if self.scheduler.scheduling_history else None
        round_analytics = RoundAnalytics(
            round_number=server_round,
            scheduling_decision=scheduling_decision,
            projection_time_ms=projection_time,
            detection_time_ms=detection_time,
            aggregation_time_ms=aggregation_time,
            trust_updates=trust_updates,
            byzantine_detected=byzantine_clients,
            consensus_achieved=detection_results['consensus_achieved']
        )

        self.round_analytics.append(round_analytics)

        total_time = time.time() - start_time
        logger.info(f"Round {server_round} aggregation complete: "
                   f"{len(verified_clients)} verified + {len(promoted_clients)} promoted clients, "
                   f"total time: {total_time*1000:.1f}ms")

        # Prepare metrics
        metrics = {
            "round": server_round,
            "num_verified": len(verified_clients),
            "num_promoted": len(promoted_clients),
            "num_byzantine_detected": len(byzantine_clients),
            "consensus_achieved": detection_results['consensus_achieved'],
            "budget_utilization": scheduling_decision.budget_utilization if scheduling_decision else 0.0,
            "projection_time_ms": projection_time,
            "detection_time_ms": detection_time,
            "aggregation_time_ms": aggregation_time,
            "total_time_ms": total_time * 1000
        }

        # Save analytics if configured
        if self.config.save_round_decisions:
            self._save_round_analytics(round_analytics)

        return aggregated_parameters, metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation (standard implementation)."""
        if self.config.evaluate_fn is not None:
            # Server-side evaluation
            return []

        # Client-side evaluation - sample a few clients
        sample_size = min(5, client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=1)

        evaluate_config = {"round": server_round}
        evaluate_ins = EvaluateIns(parameters, evaluate_config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Compute weighted average of accuracy
        total_examples = 0
        total_loss = 0.0
        total_accuracy = 0.0

        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_examples += num_examples
            total_loss += eval_res.loss * num_examples

            if "accuracy" in eval_res.metrics:
                total_accuracy += eval_res.metrics["accuracy"] * num_examples

        if total_examples > 0:
            avg_loss = total_loss / total_examples
            avg_accuracy = total_accuracy / total_examples
        else:
            avg_loss = 0.0
            avg_accuracy = 0.0

        # Update round analytics with accuracy
        if self.round_analytics and self.round_analytics[-1].round_number == server_round:
            self.round_analytics[-1].model_accuracy = avg_accuracy

        logger.info(f"Round {server_round} evaluation: loss={avg_loss:.4f}, accuracy={avg_accuracy:.1%}")

        return avg_loss, {"accuracy": avg_accuracy, "num_examples": total_examples}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model parameters using server-side evaluation function.

        Args:
            server_round: Current server round
            parameters: Global model parameters to evaluate

        Returns:
            Optional tuple of (loss, metrics) or None if no server-side evaluation
        """
        if self.config.evaluate_fn is None:
            return None

        # Convert parameters for evaluation
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        # Execute server-side evaluation
        loss, metrics = self.config.evaluate_fn(server_round, parameters_ndarrays, {})

        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine number of clients to sample for training."""
        sample_size = max(self.config.min_fit_clients, min(num_available_clients, 10))
        min_clients = max(1, self.config.min_fit_clients)
        return sample_size, min_clients

    def _initialize_projection_system(self, parameter_dim: int):
        """Initialize projection system based on configuration."""
        if self.config.projection_type == "structured" and self.model_structure is not None:
            self.projection = StructuredJLProjection(
                model_structure=self.model_structure,
                target_k=self.config.target_k,
                device="cpu"
            )
            logger.info(f"Initialized structured projection with {len(self.model_structure.blocks)} blocks")
        else:
            # For dense projections, use target_k but convert to ratio for compatibility
            k_ratio = min(1.0, self.config.target_k / parameter_dim)
            self.projection = DenseJLProjection(
                original_dim=parameter_dim,
                k_ratio=k_ratio,
                device="cpu"
            )
            logger.info(f"Initialized dense projection: {parameter_dim} → {int(parameter_dim * k_ratio)}")

    def _save_round_analytics(self, analytics: RoundAnalytics):
        """Save round analytics to disk for analysis."""
        analytics_file = self.output_dir / f"round_{analytics.round_number}_analytics.json"

        # Convert to serializable format
        analytics_dict = asdict(analytics)

        # Handle non-serializable objects
        if analytics_dict["scheduling_decision"]:
            # Check if it's already a dict or needs conversion
            if hasattr(analytics_dict["scheduling_decision"], '__dict__'):
                analytics_dict["scheduling_decision"] = asdict(analytics_dict["scheduling_decision"])
            # If it's already a dict, leave it as is

        with open(analytics_file, 'w') as f:
            json.dump(analytics_dict, f, indent=2, default=str)

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trust system statistics."""
        return self.scheduler.get_client_statistics()

    def export_complete_state(self) -> Dict[str, Any]:
        """Export complete TAVS-ESP state for analysis."""
        return {
            "config": asdict(self.config),
            "round_number": self.round_number,
            "trust_state": self.scheduler.export_trust_state(),
            "csprng_stats": self.csprng_manager.get_security_stats(),
            "round_analytics": [asdict(analytics) for analytics in self.round_analytics]
        }