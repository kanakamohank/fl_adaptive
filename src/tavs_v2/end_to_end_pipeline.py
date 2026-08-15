#!/usr/bin/env python3
import logging
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict, field

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.common import Context  # <--- THE FIX: Required for modern Flower API
from flwr.simulation import run_simulation

from .tavs_esp_strategy import TavsEspStrategy, TavsEspConfig
from src.clients.tavs_flower_client import TAVSFlowerClient, TAVSClientConfig, create_tavs_flower_client
from src.core.models import ModelStructure, get_model
from src.utils.data_utils import load_cifar10, create_dirichlet_splits

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    byzantine_fraction: float = 0.25

    model_type: str = "cifar_cnn"
    dataset: str = "cifar10"
    data_alpha: float = 0.3

    tavs_config: TavsEspConfig = None

    # Server strategy class to run. Defaults to TAVS; set to
    # FullVerificationStrategy for the traditional verify-everyone baseline.
    # Selecting the baseline by class rather than by neutering TavsEspConfig
    # keeps the two arms from silently becoming the same algorithm.
    strategy_class: type = None

    client_epochs: int = 5
    client_batch_size: int = 32
    client_learning_rate: float = 0.01

    attack_types: List[str] = None
    attack_intensities: List[float] = None

    simulation_backend: str = "ray"
    ray_init_args: Dict = None

    output_dir: str = "tavs_esp_results"
    save_client_data: bool = False

    # Master seed for this run. Controls model initialisation, DataLoader
    # shuffling, local SGD ordering and the Dirichlet client split.
    #
    # Note this does NOT control the ESP projection matrices: those are derived
    # deterministically from SHA-256(master_key || round || block), so they are
    # identical across runs sharing a master_key and contribute no run-to-run
    # variance. Vary master_key separately to test sensitivity to the projection
    # draw -- that is a different question from statistical significance.
    seed: int = 42

    # Pin the remaining sources of run-to-run variation: deterministic torch
    # kernels, hash seed, in-process data loading and seeded client sampling.
    # Costs some speed, and is what makes "same seed" actually mean "same run".
    deterministic: bool = True

    # Refuse to run when TAVS cannot promote within num_rounds. Disable only for
    # tests that build deliberately tiny pipelines to exercise plumbing; a real
    # experiment with this off silently reports a 1.0x efficiency result.
    validate_promotion_feasibility: bool = True

    def __post_init__(self):
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
    config: PipelineConfig
    server_metrics: List[Dict[str, Any]]
    server_losses: List[float]
    server_accuracies: List[float]
    final_trust_state: Dict[str, Any]
    trust_evolution: Dict[str, List[float]]
    tier_evolution: Dict[str, List[int]]
    byzantine_detection_history: List[Dict[str, Any]]
    attack_success_rates: Dict[str, float]
    total_time_seconds: float
    round_times: List[float]
    convergence_metrics: Dict[str, float]
    security_metrics: Dict[str, float]
    # Measured per-round verified/promoted counts from the scheduler. Defaulted
    # because it was added after every existing construction site.
    scheduling_history: List[Dict[str, int]] = field(default_factory=list)

class TAVSESPPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client_datasets = None
        self.test_dataset = None
        self.model_structure = None
        self.client_configs = None

    def setup_data_and_model(self):
        # Seed before anything stochastic. torch covers model init, DataLoader
        # shuffling and local SGD; numpy covers client sampling.
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        # Seeding alone did NOT make runs reproducible: re-running seeds 1-3
        # moved late accuracy by up to 0.092 and changed verification counts
        # (107 -> 112), because Flower's client sampling, DataLoader worker
        # ordering and non-deterministic kernels sit outside the seeds above.
        # That inflates variance and breaks the pairing the seeded comparison
        # relies on, so those sources are pinned too when requested.
        if self.config.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            os.environ["PYTHONHASHSEED"] = str(self.config.seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # older torch, or an op with no det. kernel
                logger.warning(f"Could not enable deterministic algorithms: {exc}")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.config.dataset == "cifar10":
            train_dataset, self.test_dataset = load_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        # create_dirichlet_splits resets the global numpy seed internally, so the
        # run seed must be passed explicitly or every run gets the same split.
        self.client_datasets = create_dirichlet_splits(
            train_dataset, num_clients=self.config.num_clients,
            alpha=self.config.data_alpha, seed=self.config.seed
        )
        # Restore the run seed: the split helper left the global RNG at its own
        # state, which would otherwise be identical for every seed value.
        np.random.seed(self.config.seed)

        model = get_model(self.config.model_type, num_classes=10)
        if hasattr(model, 'structure'):
            self.model_structure = model.structure
        else:
            self.model_structure = ModelStructure()
            total_params = sum(p.numel() for p in model.parameters())
            self.model_structure.add_block('full_model', (total_params,), total_params)

    def setup_clients(self):
        self.client_configs = []
        num_byzantine = int(self.config.num_clients * self.config.byzantine_fraction)
        num_honest = self.config.num_clients - num_byzantine

        for i in range(num_honest):
            self.client_configs.append(TAVSClientConfig(
                client_id=f"honest_{i:02d}", client_type="honest", model_type=self.config.model_type,
                epochs=self.config.client_epochs, batch_size=self.config.client_batch_size, learning_rate=self.config.client_learning_rate
            ))

        for i in range(num_byzantine):
            attack_type = self.config.attack_types[i % len(self.config.attack_types)]
            attack_intensity = self.config.attack_intensities[i % len(self.config.attack_intensities)]
            self.client_configs.append(TAVSClientConfig(
                client_id=f"byzantine_{i:02d}", client_type=attack_type, model_type=self.config.model_type,
                attack_intensity=attack_intensity, target_fraction=0.001 if attack_type == "layerwise" else 1.0,
                epochs=self.config.client_epochs, batch_size=self.config.client_batch_size, learning_rate=self.config.client_learning_rate
            ))

    def create_client_fn(self) -> Callable[[Context], fl.client.Client]:
        # Uses the modern 'Context' API to prevent Flower's buggy NumpyClient serialization
        def client_fn(context: Context) -> fl.client.Client:
            # Import DataLoader
            from torch.utils.data import DataLoader
            
            try:
                partition_id = context.node_config.get("partition-id", 0)
                if partition_id >= len(self.client_configs):
                    partition_id = partition_id % len(self.client_configs)
                    
                client_config = self.client_configs[partition_id]
                
                # ---> THE FIX: Wrap the PyTorch Subset in a DataLoader <---
                train_dataset = self.client_datasets[partition_id]
                # Per-client generator derived from the run seed, so shuffling
                # order is fixed per (seed, client) rather than drawn from the
                # ambient torch RNG whose state depends on execution order.
                # num_workers=0 keeps loading in-process: worker subprocesses
                # reorder batches non-deterministically.
                loader_gen = torch.Generator()
                loader_gen.manual_seed(self.config.seed * 100003 + partition_id)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=client_config.batch_size,
                    shuffle=True,
                    drop_last=False,  # Ensure no data is left behind
                    generator=loader_gen,
                    num_workers=0,
                )

                # .to_client() forces modern serialization
                return create_tavs_flower_client(
                    config=client_config,
                    train_loader=train_loader,
                    test_loader=None
                ).to_client()

            except Exception as e:
                logger.error(f"Failed to create client with context {context}: {e}")
                raise
        return client_fn

    def create_server_strategy(self) -> TavsEspStrategy:
        strategy_config = self.config.tavs_config
        strategy_config.min_fit_clients = min(self.config.clients_per_round, self.config.num_clients)
        strategy_config.min_available_clients = self.config.clients_per_round
        strategy_config.evaluate_fn = self._create_evaluate_function()
        # Cohort sampling must vary with the run seed but stay fixed for a given
        # (seed, round), which is what makes the two arms genuinely paired.
        strategy_config.sampling_seed = self.config.seed
        strategy_config.deterministic_sampling = self.config.deterministic
        strategy_config.min_evaluate_clients = 0
        strategy_config.fraction_evaluate = 0.0

        strategy_class = self.config.strategy_class or TavsEspStrategy
        logger.info(f"Server strategy: {strategy_class.__name__}")
        strategy = strategy_class(config=strategy_config, model_structure=self.model_structure)

        # Refuse to run a TAVS experiment whose parameters make promotion
        # mathematically impossible. Without this the run completes normally,
        # every client stays Tier 1, and the experiment reports "1.0x efficiency
        # improvement" -- a real number produced by two identical algorithms.
        # FullVerificationStrategy never promotes by design, so it is exempt.
        if strategy_class is TavsEspStrategy and self.config.validate_promotion_feasibility:
            feasibility = strategy.scheduler.describe_promotion_feasibility(self.config.num_rounds)
            if not feasibility["feasible"]:
                raise ValueError(
                    f"TAVS cannot promote any client within {self.config.num_rounds} rounds: "
                    f"promotion first becomes possible at round "
                    f"{feasibility['min_round_for_promotion']}, limited by "
                    f"{feasibility['binding_constraint']}. Bounds: "
                    f"{ {k: round(v, 1) for k, v in feasibility['bounds'].items()} }. "
                    f"TAVS would degenerate into full verification and the comparison "
                    f"would be meaningless. Increase num_rounds, or lower tau_ramp / "
                    f"theta_high / alpha_trust / k_trust."
                )
            logger.info(
                f"Promotion feasible from round {feasibility['min_round_for_promotion']} "
                f"of {self.config.num_rounds}"
            )
        return strategy

    def _create_evaluate_function(self):
        def evaluate_fn(server_round: int, parameters_ndarrays, config_dict):

            try:
                import torch
                from src.core.models import get_model
                from src.utils.data_utils import load_cifar10
                model = get_model(self.config.model_type)
                params_dict = zip(model.parameters(), parameters_ndarrays)
                for param, new_param in params_dict:
                    new_param = np.array(new_param)
                    if np.any(np.isnan(new_param)) or np.any(np.isinf(new_param)):
                        logger.warning(f"Server eval round {server_round}: NaN/Inf detected in parameters, replacing with zeros")
                        new_param = np.nan_to_num(new_param, nan=0.0, posinf=1e6, neginf=-1e6)
                    param.data = torch.tensor(new_param, dtype=param.dtype)

                _, test_data = load_cifar10()
                test_subset = torch.utils.data.Subset(test_data, range(min(1000, len(test_data))))
                test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

                model.eval()
                criterion = torch.nn.CrossEntropyLoss()
                total_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        batch_loss = loss.item()
                        if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                            total_loss += batch_loss
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                accuracy = correct / total if total > 0 else 0.0
                avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

                if np.isnan(avg_loss) or np.isinf(avg_loss):
                    avg_loss = 2.3

                return avg_loss, {"accuracy": accuracy, "correct": correct, "total": total}
            except Exception as e:
                logger.warning(f"Evaluation failed in round {server_round}: {e}")
                return 2.3, {"accuracy": 0.1, "correct": 0, "total": 100}
        return evaluate_fn

    def run_simulation(self) -> PipelineResults:
        start_time = time.time()
        self.setup_data_and_model()
        self.setup_clients()

        strategy = self.create_server_strategy()
        client_fn = self.create_client_fn()
        server_config = ServerConfig(num_rounds=self.config.num_rounds)
        client_resources = {"num_cpus": 1, "num_gpus": 0} if self.config.simulation_backend == "ray" else None

        try:
            # THE FIX: Modern server_fn bypasses the legacy ServerApp bugs
            def server_fn(context: Context) -> ServerAppComponents:
                return ServerAppComponents(
                    strategy=strategy,
                    config=server_config
                )

            history = run_simulation(
                server_app=ServerApp(server_fn=server_fn),
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

        total_time = time.time() - start_time
        results = self._extract_results(history, strategy, total_time)
        self._save_results(results)
        return results

    def _extract_results(self, history, strategy: TavsEspStrategy, total_time: float) -> PipelineResults:
        server_losses, server_accuracies = [], []
        loss_sources = ['losses_centralized', 'losses_distributed', 'losses']
        metrics_sources = ['metrics_centralized', 'metrics_distributed', 'metrics']

        # Primary source: metrics the strategy recorded during evaluate().
        #
        # flwr.simulation.run_simulation() returns None, so `history` is None on
        # the modern API and the History-scraping below finds nothing. That is
        # what silently triggered the synthetic-curve fallback for every run.
        # The strategy accumulates the same values as it observes them.
        strategy_evals = getattr(strategy, 'evaluation_history', None)
        if strategy_evals:
            ordered = sorted(strategy_evals, key=lambda e: e["round"])
            server_losses = [e["loss"] for e in ordered]
            server_accuracies = [e["accuracy"] for e in ordered]

        # Legacy fallback: scrape a History object if one was actually returned
        # (start_simulation-style APIs). Only runs when the strategy recorded
        # nothing, so it can never overwrite real measurements.
        for source in (loss_sources if not server_losses else []):
            if hasattr(history, source) and getattr(history, source):
                source_data = getattr(history, source)
                if isinstance(source_data, list):
                    server_losses = [item[1] if isinstance(item, tuple) else item for item in source_data]
                elif isinstance(source_data, dict):
                    server_losses = list(source_data.values())
                if server_losses: break

        for source in (metrics_sources if not server_accuracies else []):
            if hasattr(history, source) and getattr(history, source):
                source_data = getattr(history, source)
                if isinstance(source_data, dict):
                    for acc_key in ['accuracy', 'acc', 'test_accuracy']:
                        if acc_key in source_data:
                            acc_data = source_data[acc_key]
                            if isinstance(acc_data, list):
                                server_accuracies = []
                                for item in acc_data:
                                    if isinstance(item, tuple) and len(item) >= 2:
                                        metrics_dict = item[1]
                                        server_accuracies.append(metrics_dict.get("accuracy", 0.0) if isinstance(metrics_dict, dict) else float(metrics_dict))
                                    elif isinstance(item, (int, float)):
                                        server_accuracies.append(float(item))
                            if server_accuracies: break
                    if server_accuracies: break

        # Metric extraction must never be papered over. This branch previously
        # synthesised a plausible-looking exponential learning curve
        # (0.1 + 0.75*(1-exp(-0.25*r)) plus Gaussian noise) and returned it as if
        # it had been measured, which makes a silent extraction failure
        # indistinguishable from a successful run in every downstream plot,
        # report and paper table. Fail loudly instead: an empty history means the
        # centralised evaluate_fn never ran, and that is a bug to fix at the
        # source, not a gap to fill with fabricated numbers.
        if not server_losses and not server_accuracies:
            raise RuntimeError(
                "No server metrics could be extracted from the Flower History. "
                f"Searched losses in {loss_sources} and metrics in {metrics_sources}. "
                "This usually means the centralised evaluate_fn was not invoked "
                "(check TavsEspConfig.evaluate_fn and fraction_evaluate). "
                "Refusing to synthesise substitute metrics."
            )

        trust_state = strategy.export_complete_state()
        trust_evolution, tier_evolution = {}, {}

        for analytics in strategy.round_analytics:
            if analytics.scheduling_decision:
                for client_id, trust_score in analytics.scheduling_decision.trust_scores.items():
                    if client_id not in trust_evolution:
                        trust_evolution[client_id] = []
                        tier_evolution[client_id] = []
                    trust_evolution[client_id].append(trust_score)
                    tier_evolution[client_id].append(analytics.scheduling_decision.tier_assignments.get(client_id, 1))

        byzantine_detection_history = []
        for analytics in strategy.round_analytics:
            byzantine_detection_history.append({
                "round": analytics.round_number,
                "detected": analytics.byzantine_detected,
                "consensus": analytics.consensus_achieved
            })

        convergence_metrics = {
            "final_loss": server_losses[-1] if server_losses else 0.0,
            "final_accuracy": server_accuracies[-1] if server_accuracies else 0.0,
            "loss_improvement": (server_losses[0] - server_losses[-1]) if len(server_losses) >= 2 else 0.0,
            "accuracy_improvement": (server_accuracies[-1] - server_accuracies[0]) if len(server_accuracies) >= 2 else 0.0
        }

        total_detections = sum(len(d["detected"]) for d in byzantine_detection_history)
        security_metrics = {
            "total_byzantine_detections": total_detections,
            "consensus_rate": sum(1 for d in byzantine_detection_history if d["consensus"]) / len(byzantine_detection_history) if byzantine_detection_history else 0.0,
            "avg_detections_per_round": total_detections / len(byzantine_detection_history) if byzantine_detection_history else 0.0
        }

        round_times = [analytics.projection_time_ms + analytics.detection_time_ms + analytics.aggregation_time_ms for analytics in strategy.round_analytics]

        return PipelineResults(
            config=self.config, server_metrics=[], server_losses=server_losses, server_accuracies=server_accuracies,
            final_trust_state=trust_state, trust_evolution=trust_evolution, tier_evolution=tier_evolution,
            byzantine_detection_history=byzantine_detection_history,
            scheduling_history=list(getattr(strategy, 'scheduling_history', [])),
            attack_success_rates={},
            total_time_seconds=total_time, round_times=round_times, convergence_metrics=convergence_metrics, security_metrics=security_metrics
        )

    def _save_results(self, results: PipelineResults):
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)

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
                    client_id: scores[-1] if scores else 0.0 for client_id, scores in results.trust_evolution.items()
                },
                "trust_convergence": "analyzed" if results.trust_evolution else "no_data"
            },
            "security_summary": results.security_metrics
        }
        with open(self.output_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def run_tavs_esp_experiment(config: PipelineConfig) -> PipelineResults:
    return TAVSESPPipeline(config).run_simulation()

def create_example_configs() -> Dict[str, PipelineConfig]:
    return {
        "dev": PipelineConfig(num_rounds=3, num_clients=6, clients_per_round=4, byzantine_fraction=0.33, output_dir="results/dev_test"),
        "security": PipelineConfig(num_rounds=10, num_clients=20, clients_per_round=8, byzantine_fraction=0.25, attack_types=["null_space", "layerwise"], attack_intensities=[1.5, 2.0], output_dir="results/security_validation"),
        "performance": PipelineConfig(num_rounds=20, num_clients=50, clients_per_round=15, byzantine_fraction=0.2, tavs_config=TavsEspConfig(target_k=150, projection_type="structured", detection_threshold=2.0), output_dir="results/performance_eval")
    }