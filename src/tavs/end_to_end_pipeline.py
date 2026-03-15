#!/usr/bin/env python3
"""
End-to-End TAVS-ESP Federated Learning Pipeline

This module provides a complete end-to-end federated learning pipeline
integrating TAVS-ESP server strategy with Flower clients.

Core Components:
- TAVS-ESP server strategy with trust-adaptive scheduling
- TAVS Flower client wrappers (honest + attackers)
- Complete FL execution with security mechanisms
- Analytics and visualization of trust dynamics

Key Innovation: Complete working implementation of TAVS-ESP system
demonstrating Byzantine-robust federated learning with trust adaptation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

# Flower imports
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.simulation import run_simulation

# TAVS-ESP imports
from .tavs_esp_strategy import TavsEspStrategy, TavsEspConfig
from ..clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig, create_tavs_flower_client
from ..core.models import ModelStructure, get_model
from ..utils.data_utils import load_cifar10, create_dirichlet_splits

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for end-to-end TAVS-ESP pipeline."""

    # Federated Learning Settings
    num_rounds: int = 10
    num_clients: int = 20
    clients_per_round: int = 8
    byzantine_fraction: float = 0.25  # 25% Byzantine clients

    # Model and Data Settings
    model_type: str = "cifar_cnn"
    dataset: str = "cifar10"
    data_alpha: float = 0.3  # Dirichlet heterogeneity parameter

    # TAVS-ESP Configuration
    tavs_config: TavsEspConfig = None

    # Client Configuration
    client_epochs: int = 1
    client_batch_size: int = 32
    client_learning_rate: float = 0.01

    # Attack Configuration
    attack_types: List[str] = None  # ["null_space", "layerwise"]
    attack_intensities: List[float] = None  # [1.0, 2.0]

    # Simulation Settings
    simulation_backend: str = "ray"  # or "thread"
    ray_init_args: Dict = None

    # Output Settings
    output_dir: str = "tavs_esp_results"
    save_client_data: bool = False

    def __post_init__(self):
        """Set default values for complex fields."""
        if self.tavs_config is None:
            self.tavs_config = TavsEspConfig()

        if self.attack_types is None:
            self.attack_types = ["null_space", "layerwise"]

        if self.attack_intensities is None:
            self.attack_intensities = [1.0, 2.0]

        if self.ray_init_args is None:
            self.ray_init_args = {"ignore_reinit_error": True, "include_dashboard": False}


@dataclass
class PipelineResults:
    """Results from end-to-end pipeline execution."""

    # Configuration
    config: PipelineConfig

    # Server Results
    server_metrics: List[Dict[str, Any]]
    server_losses: List[float]
    server_accuracies: List[float]

    # Trust Dynamics
    final_trust_state: Dict[str, Any]
    trust_evolution: Dict[str, List[float]]  # client_id -> trust scores over rounds
    tier_evolution: Dict[str, List[int]]     # client_id -> tier assignments over rounds

    # Security Metrics
    byzantine_detection_history: List[Dict[str, Any]]
    attack_success_rates: Dict[str, float]

    # Performance Metrics
    total_time_seconds: float
    round_times: List[float]

    # Analytics
    convergence_metrics: Dict[str, float]
    security_metrics: Dict[str, float]


class TAVSESPPipeline:
    """
    Complete end-to-end TAVS-ESP federated learning pipeline.

    This class orchestrates the entire federated learning process including:
    1. Data preparation and client setup
    2. TAVS-ESP server strategy configuration
    3. Flower simulation execution
    4. Results collection and analysis
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the TAVS-ESP pipeline."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.client_datasets = None
        self.test_dataset = None
        self.model_structure = None
        self.client_configs = None

        logger.info(f"TAVS-ESP Pipeline initialized: {config.num_clients} clients, "
                   f"{config.num_rounds} rounds, {config.byzantine_fraction:.1%} Byzantine")

    def setup_data_and_model(self):
        """Setup federated data splits and model structure."""
        logger.info("Setting up data and model...")

        # Load dataset
        if self.config.dataset == "cifar10":
            train_dataset, self.test_dataset = load_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        # Create federated data splits
        self.client_datasets = create_dirichlet_splits(
            train_dataset,
            num_clients=self.config.num_clients,
            alpha=self.config.data_alpha
        )

        logger.info(f"Created {len(self.client_datasets)} federated data splits")

        # Initialize model structure
        model = get_model(self.config.model_type, num_classes=10)
        if hasattr(model, 'structure'):
            self.model_structure = model.structure
        else:
            # Create basic structure for models without built-in structure
            self.model_structure = ModelStructure()
            total_params = sum(p.numel() for p in model.parameters())
            self.model_structure.add_block('full_model', (total_params,), total_params)

        logger.info(f"Model structure: {len(self.model_structure.blocks)} blocks, "
                   f"{self.model_structure.total_params} parameters")

    def setup_clients(self):
        """Setup TAVS Flower clients (honest + Byzantine)."""
        logger.info("Setting up TAVS clients...")

        self.client_configs = []

        # Calculate number of Byzantine clients
        num_byzantine = int(self.config.num_clients * self.config.byzantine_fraction)
        num_honest = self.config.num_clients - num_byzantine

        # Create honest clients
        for i in range(num_honest):
            config = TAVSClientConfig(
                client_id=f"honest_{i:02d}",
                client_type="honest",
                model_type=self.config.model_type,
                epochs=self.config.client_epochs,
                batch_size=self.config.client_batch_size,
                learning_rate=self.config.client_learning_rate
            )
            self.client_configs.append(config)

        # Create Byzantine clients with different attack types
        for i in range(num_byzantine):
            attack_type = self.config.attack_types[i % len(self.config.attack_types)]
            attack_intensity = self.config.attack_intensities[i % len(self.config.attack_intensities)]

            config = TAVSClientConfig(
                client_id=f"byzantine_{i:02d}",
                client_type=attack_type,
                model_type=self.config.model_type,
                attack_intensity=attack_intensity,
                target_fraction=0.001 if attack_type == "layerwise" else 1.0,
                epochs=self.config.client_epochs,
                batch_size=self.config.client_batch_size,
                learning_rate=self.config.client_learning_rate
            )
            self.client_configs.append(config)

        logger.info(f"Configured {num_honest} honest + {num_byzantine} Byzantine clients")

        # Log attack distribution
        attack_distribution = {}
        for config in self.client_configs:
            attack_distribution[config.client_type] = attack_distribution.get(config.client_type, 0) + 1

        logger.info(f"Attack distribution: {attack_distribution}")

    def create_client_fn(self) -> Callable[[str], TAVSFlowerClient]:
        """Create client function for Flower simulation."""

        def client_fn(cid: str) -> TAVSFlowerClient:
            """Create a TAVS Flower client for the given client ID."""
            try:
                # Handle both Flower-generated IDs ("0", "1", "2") and our config IDs ("honest_00", "byzantine_00")
                client_config = None
                client_idx = None

                # First try direct ID match
                for config in self.client_configs:
                    if config.client_id == cid:
                        client_config = config
                        # Extract index from our config ID format
                        client_idx = int(config.client_id.split('_')[1])
                        break

                # If not found, try mapping numeric Flower IDs to our configs
                if client_config is None:
                    try:
                        numeric_id = int(cid)
                        if 0 <= numeric_id < len(self.client_configs):
                            client_config = self.client_configs[numeric_id]
                            client_idx = numeric_id
                    except ValueError:
                        pass

                if client_config is None:
                    raise ValueError(f"No configuration found for client {cid}")

                # Get client dataset
                if client_idx >= len(self.client_datasets):
                    raise ValueError(f"No dataset found for client index {client_idx}")

                train_loader = self.client_datasets[client_idx]

                # Create TAVS client
                return create_tavs_flower_client(
                    config=client_config,
                    train_loader=train_loader,
                    test_loader=None  # Server-side evaluation
                )

            except Exception as e:
                logger.error(f"Failed to create client {cid}: {e}")
                raise

        return client_fn

    def create_server_strategy(self) -> TavsEspStrategy:
        """Create TAVS-ESP server strategy."""
        logger.info("Creating TAVS-ESP server strategy...")

        # Configure TAVS-ESP strategy
        strategy_config = self.config.tavs_config
        strategy_config.min_fit_clients = min(self.config.clients_per_round, self.config.num_clients)
        strategy_config.min_available_clients = self.config.clients_per_round

        # Create strategy
        strategy = TavsEspStrategy(
            config=strategy_config,
            model_structure=self.model_structure
        )

        logger.info(f"TAVS-ESP strategy configured: "
                   f"theta_low={strategy_config.theta_low}, "
                   f"theta_high={strategy_config.theta_high}, "
                   f"projection_type={strategy_config.projection_type}")

        return strategy

    def run_simulation(self) -> PipelineResults:
        """Run the complete TAVS-ESP federated learning simulation."""
        logger.info("Starting TAVS-ESP simulation...")

        start_time = time.time()

        # Setup components
        self.setup_data_and_model()
        self.setup_clients()

        # Create strategy and client function
        strategy = self.create_server_strategy()
        client_fn = self.create_client_fn()

        # Configure simulation
        server_config = ServerConfig(num_rounds=self.config.num_rounds)

        # Create client resources (for simulation)
        client_resources = None
        if self.config.simulation_backend == "ray":
            client_resources = {"num_cpus": 1, "num_gpus": 0}

        # Run Flower simulation
        logger.info(f"Running Flower simulation: {self.config.num_rounds} rounds, "
                   f"{self.config.clients_per_round} clients per round")

        # Create list of client IDs
        client_ids = [config.client_id for config in self.client_configs]

        try:
            # Run simulation
            history = run_simulation(
                server_app=ServerApp(
                    config=server_config,
                    strategy=strategy
                ),
                client_app=ClientApp(client_fn=client_fn),
                num_supernodes=self.config.num_clients,
                backend_config={
                    "client_resources": client_resources,
                    "init_args": self.config.ray_init_args if self.config.simulation_backend == "ray" else {}
                }
            )

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

        # Calculate total time
        total_time = time.time() - start_time

        # Extract results
        results = self._extract_results(history, strategy, total_time)

        # Save results
        self._save_results(results)

        logger.info(f"TAVS-ESP simulation completed in {total_time:.1f}s")

        return results

    def _extract_results(self, history, strategy: TavsEspStrategy, total_time: float) -> PipelineResults:
        """Extract and analyze results from simulation."""
        logger.info("Extracting simulation results...")

        # Server metrics
        server_losses = [loss for loss, _ in history.losses_distributed]
        server_accuracies = [metrics.get("accuracy", 0.0) for _, metrics in history.metrics_distributed["accuracy"]]

        # Trust dynamics
        trust_state = strategy.export_complete_state()

        # Extract trust evolution
        trust_evolution = {}
        tier_evolution = {}

        for analytics in strategy.round_analytics:
            if analytics.scheduling_decision:
                for client_id, trust_score in analytics.scheduling_decision.trust_scores.items():
                    if client_id not in trust_evolution:
                        trust_evolution[client_id] = []
                        tier_evolution[client_id] = []

                    trust_evolution[client_id].append(trust_score)
                    tier_evolution[client_id].append(analytics.scheduling_decision.tier_assignments.get(client_id, 1))

        # Byzantine detection analysis
        byzantine_detection_history = []
        attack_success_rates = {}

        for analytics in strategy.round_analytics:
            detection_info = {
                "round": analytics.round_number,
                "detected": analytics.byzantine_detected,
                "consensus": analytics.consensus_achieved
            }
            byzantine_detection_history.append(detection_info)

        # Calculate attack success rates by client type
        client_type_results = {"honest": [], "byzantine": []}
        for config in self.client_configs:
            if config.client_type == "honest":
                client_type_results["honest"].append(config.client_id)
            else:
                client_type_results["byzantine"].append(config.client_id)

        # Convergence metrics
        convergence_metrics = {
            "final_loss": server_losses[-1] if server_losses else 0.0,
            "final_accuracy": server_accuracies[-1] if server_accuracies else 0.0,
            "loss_improvement": (server_losses[0] - server_losses[-1]) if len(server_losses) >= 2 else 0.0,
            "accuracy_improvement": (server_accuracies[-1] - server_accuracies[0]) if len(server_accuracies) >= 2 else 0.0
        }

        # Security metrics
        total_detections = sum(len(d["detected"]) for d in byzantine_detection_history)
        total_rounds_with_consensus = sum(1 for d in byzantine_detection_history if d["consensus"])

        security_metrics = {
            "total_byzantine_detections": total_detections,
            "consensus_rate": total_rounds_with_consensus / len(byzantine_detection_history) if byzantine_detection_history else 0.0,
            "avg_detections_per_round": total_detections / len(byzantine_detection_history) if byzantine_detection_history else 0.0
        }

        # Round times (from analytics)
        round_times = [analytics.projection_time_ms + analytics.detection_time_ms + analytics.aggregation_time_ms
                      for analytics in strategy.round_analytics]

        return PipelineResults(
            config=self.config,
            server_metrics=[],  # Would need to extract from history
            server_losses=server_losses,
            server_accuracies=server_accuracies,
            final_trust_state=trust_state,
            trust_evolution=trust_evolution,
            tier_evolution=tier_evolution,
            byzantine_detection_history=byzantine_detection_history,
            attack_success_rates=attack_success_rates,
            total_time_seconds=total_time,
            round_times=round_times,
            convergence_metrics=convergence_metrics,
            security_metrics=security_metrics
        )

    def _save_results(self, results: PipelineResults):
        """Save results to disk."""
        logger.info(f"Saving results to {self.output_dir}")

        # Save complete results
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)

        # Save summary
        summary = {
            "experiment_config": asdict(results.config),
            "final_metrics": {
                "loss": results.server_losses[-1] if results.server_losses else None,
                "accuracy": results.server_accuracies[-1] if results.server_accuracies else None,
                "total_time": results.total_time_seconds,
                "avg_round_time": np.mean(results.round_times) if results.round_times else None
            },
            "trust_summary": {
                "final_trust_distribution": {
                    client_id: scores[-1] if scores else 0.0
                    for client_id, scores in results.trust_evolution.items()
                },
                "trust_convergence": "analyzed" if results.trust_evolution else "no_data"
            },
            "security_summary": results.security_metrics
        }

        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved: {results_file}, summary: {summary_file}")


def run_tavs_esp_experiment(config: PipelineConfig) -> PipelineResults:
    """
    Run a complete TAVS-ESP experiment with the given configuration.

    This is the main entry point for running TAVS-ESP experiments.
    """
    pipeline = TAVSESPPipeline(config)
    return pipeline.run_simulation()


def create_example_configs() -> Dict[str, PipelineConfig]:
    """Create example experiment configurations for different scenarios."""

    configs = {}

    # Small-scale development config
    configs["dev"] = PipelineConfig(
        num_rounds=3,
        num_clients=6,
        clients_per_round=4,
        byzantine_fraction=0.33,
        output_dir="results/dev_test"
    )

    # Security validation config
    configs["security"] = PipelineConfig(
        num_rounds=10,
        num_clients=20,
        clients_per_round=8,
        byzantine_fraction=0.25,
        attack_types=["null_space", "layerwise"],
        attack_intensities=[1.5, 2.0],
        output_dir="results/security_validation"
    )

    # Performance evaluation config
    configs["performance"] = PipelineConfig(
        num_rounds=20,
        num_clients=50,
        clients_per_round=15,
        byzantine_fraction=0.2,
        tavs_config=TavsEspConfig(
            target_k=150,
            projection_type="structured",
            detection_threshold=2.0
        ),
        output_dir="results/performance_eval"
    )

    return configs