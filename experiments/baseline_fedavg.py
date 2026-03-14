#!/usr/bin/env python3
"""
Baseline FedAvg experiment for validation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import flwr as fl
from flwr.server import start_server
from flwr.client import start_numpy_client
import argparse
import logging
from typing import Dict, Any
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

from src.core.models import get_model
from src.clients.honest_client import create_honest_client
from src.server.fedavg_strategy import FedAvgStrategy
from src.utils.data_utils import load_cifar10, create_iid_splits, create_dirichlet_splits, analyze_data_distribution


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_server_evaluate_fn(model_type: str, model_kwargs: Dict, test_data):
    """Create server-side evaluation function."""

    def evaluate_fn(server_round: int, parameters_ndarrays, config: Dict):
        # Create model and set parameters
        model = get_model(model_type, **model_kwargs)

        # Set parameters
        params_dict = zip(model.parameters(), parameters_ndarrays)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype)

        # Evaluate
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

        logger.info(f"Server evaluation - Round {server_round}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        return avg_loss, {"accuracy": accuracy, "correct": correct, "total": total}

    return evaluate_fn


def run_client(client_id: str, server_address: str, model_type: str, model_kwargs: Dict,
               train_data, test_data, config: Dict):
    """Run a single client."""
    try:
        logger.info(f"Starting client {client_id}")

        # Create client
        client = create_honest_client(
            client_id=client_id,
            model_type=model_type,
            model_kwargs=model_kwargs,
            train_data=train_data,
            test_data=test_data,
            device=config.get("device", "cpu"),
            batch_size=config.get("batch_size", 32),
            local_epochs=config.get("local_epochs", 5),
            lr=config.get("lr", 0.01)
        )

        # Start client
        start_numpy_client(server_address=server_address, client=client)
        logger.info(f"Client {client_id} completed")

    except Exception as e:
        logger.error(f"Client {client_id} failed: {e}")
        raise


def run_fedavg_experiment(config: Dict):
    """Run the baseline FedAvg experiment."""
    logger.info("Starting FedAvg experiment")
    logger.info(f"Configuration: {config}")

    # Load data
    logger.info("Loading CIFAR-10 dataset")
    trainset, testset = load_cifar10(config["data_dir"])

    # Create client splits
    if config["data_distribution"] == "iid":
        logger.info("Creating IID data splits")
        client_datasets = create_iid_splits(trainset, config["num_clients"], config["seed"])
    else:
        logger.info(f"Creating Dirichlet splits with alpha={config['alpha']}")
        client_datasets = create_dirichlet_splits(
            trainset, config["num_clients"], config["alpha"], num_classes=10, seed=config["seed"]
        )

    # Analyze data distribution
    distribution_analysis = analyze_data_distribution(client_datasets, num_classes=10)
    logger.info(f"Data distribution analysis: {distribution_analysis}")

    # Model configuration
    model_kwargs = {"num_classes": 10}

    # Create strategy
    strategy = FedAvgStrategy(
        model_type=config["model_type"],
        model_kwargs=model_kwargs,
        fraction_fit=config.get("fraction_fit", 1.0),
        fraction_evaluate=config.get("fraction_evaluate", 1.0),
        min_fit_clients=config.get("min_fit_clients", 2),
        min_evaluate_clients=config.get("min_evaluate_clients", 2),
        min_available_clients=config.get("min_available_clients", 2),
        evaluate_fn=get_server_evaluate_fn(config["model_type"], model_kwargs, testset),
    )

    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=config["num_rounds"])

    # Start server in a separate process
    server_address = config.get("server_address", "localhost:8080")

    def start_server_process():
        try:
            logger.info(f"Starting server on {server_address}")
            start_server(
                server_address=server_address,
                config=server_config,
                strategy=strategy,
            )
        except Exception as e:
            logger.error(f"Server failed: {e}")
            raise

    # Start clients
    def start_clients():
        try:
            time.sleep(2)  # Give server time to start

            client_configs = []
            for i in range(config["num_clients"]):
                client_config = {
                    "device": config.get("device", "cpu"),
                    "batch_size": config.get("batch_size", 32),
                    "local_epochs": config.get("local_epochs", 5),
                    "lr": config.get("lr", 0.01)
                }
                client_configs.append((f"client_{i}", server_address, config["model_type"],
                                     model_kwargs, client_datasets[i], testset, client_config))

            # Run clients sequentially for now (can be parallelized later)
            for client_config in client_configs:
                run_client(*client_config)

        except Exception as e:
            logger.error(f"Client startup failed: {e}")
            raise

    # Run server and clients
    try:
        # In a real federated setting, server and clients would be on different machines
        # For simulation, we'll run them sequentially
        logger.info("Note: Running simulation mode (server and clients sequentially)")

        # For now, let's create a simple validation
        logger.info("Validating model and data loading...")

        # Test model creation
        test_model = get_model(config["model_type"], **model_kwargs)
        logger.info(f"Model created successfully: {test_model}")

        # Test client creation
        test_client = create_honest_client(
            client_id="test_client",
            model_type=config["model_type"],
            model_kwargs=model_kwargs,
            train_data=client_datasets[0],
            test_data=testset,
            device=config.get("device", "cpu"),
            batch_size=32,
            local_epochs=1,
            lr=0.01
        )
        logger.info("Test client created successfully")

        # Test one training step
        initial_params = test_client.get_parameters({})
        updated_params, num_examples, metrics = test_client.fit(initial_params, {})
        logger.info(f"Test training completed: {num_examples} examples, metrics: {metrics}")

        # Test evaluation
        loss, num_examples, eval_metrics = test_client.evaluate(updated_params, {})
        logger.info(f"Test evaluation completed: loss={loss:.4f}, metrics: {eval_metrics}")

        logger.info("Baseline FedAvg validation successful!")

        return {
            "status": "success",
            "distribution_analysis": distribution_analysis,
            "test_metrics": {
                "train_loss": metrics.get("final_loss", 0.0),
                "eval_loss": loss,
                "eval_accuracy": eval_metrics.get("accuracy", 0.0)
            }
        }

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Baseline FedAvg Experiment")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--model_type", type=str, default="cifar_cnn", help="Model type")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--data_distribution", type=str, default="iid", choices=["iid", "dirichlet"],
                       help="Data distribution")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha parameter")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create configuration
    config = vars(args)

    # Run experiment
    results = run_fedavg_experiment(config)

    logger.info("Experiment completed")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()