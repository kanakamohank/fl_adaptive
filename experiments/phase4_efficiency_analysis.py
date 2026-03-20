import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from typing import List, Dict, Any
from src.tavs.end_to_end_pipeline import PipelineConfig, TavsEspConfig
# Adjust these imports to match your project's exact structure
from src.tavs.tavs_esp_strategy import TavsEspStrategy
from src.tavs.scheduler import TavsScheduler
# Assuming you have a config module and a client factory

from src.tavs.end_to_end_pipeline import TAVSESPPipeline

class EfficiencyExperimentRunner:
    """Runs Phase 4 efficiency and scalability experiments for TAVS-ESP."""

    def __init__(self, output_dir: str = "./results/efficiency"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_simulation(self, gamma: float, num_rounds: int, num_clients: int, exp_name: str):
        """Helper to run a Flower simulation with a specific TAVS budget."""
        print(f"\n--- Starting {exp_name} (Gamma: {gamma}) ---")

        # 1. Create the specific TAVS config for this run
        tavs_config = TavsEspConfig(
            gamma_budget=gamma,
            theta_low=0.3,
            theta_high=0.8,
            alpha_trust=0.9,
            projection_type="structured",
            detection_threshold=5.0  # BVD Threshold
        )

        # 2. Wrap it in the main PipelineConfig that the strategy expects
        pipeline_config = PipelineConfig(
            num_rounds=num_rounds,
            num_clients=num_clients,
            clients_per_round=num_clients,  # Sample everyone available
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            model_type="cifar_cnn",
            dataset="cifar10",
            output_dir=self.output_dir,
            tavs_config=tavs_config
        )

        # 3. Initialize the Strategy using your custom config
        strategy = TavsEspStrategy(config=pipeline_config)

        # 4. Start Simulation
        start_time = time.time()

        # Note: Replace `client_fn` with your actual client loader when ready
        # client_fn = get_client_fn(dataset="cifar10", byzantine_fraction=0.1)

        # MOCK SIMULATION CALL (Uncomment real fl.simulation.start_simulation in production)
        # history = fl.simulation.start_simulation(
        #     client_fn=client_fn,
        #     num_clients=num_clients,
        #     config=fl.server.ServerConfig(num_rounds=num_rounds),
        #     strategy=strategy,
        # )

        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f} seconds.")

        # MOCK DATA FOR SCRIPT TESTING (Remove when hooking up real simulation)
        mock_accuracy = 0.85 - (0.05 * (1.0 - gamma)) # Slight accuracy penalty for lower gamma
        mock_analytics = []
        for r in range(1, num_rounds + 1):
            promoted = min(int(num_clients * 0.8 * (1 - np.exp(-r/10))), int(num_clients * (1 - gamma)))
            agg_time = 1500 - (promoted * 10) + np.random.normal(0, 20)
            mock_analytics.append({
                "round": r,
                "num_promoted": promoted,
                "aggregation_time_ms": max(200, agg_time)
            })

        return {
            "total_time_seconds": total_time,
            "final_accuracy": mock_accuracy,
            "round_analytics": mock_analytics
        }

    def run_experiment_2_warmup(self):
        """Exp 2: Compute reduction as trust is established."""
        print("\n>>> Running Experiment 2: Warm-Up Compute Reduction")
        num_rounds = 50
        num_clients = 100

        # Run with 30% verification budget
        results = self._run_simulation(gamma=0.3, num_rounds=num_rounds, num_clients=num_clients, exp_name="Exp2_WarmUp")

        # Extract data for plotting
        rounds = [data['round'] for data in results['round_analytics']]
        agg_times = [data['aggregation_time_ms'] for data in results['round_analytics']]
        promoted_percentages = [(data['num_promoted'] / num_clients) * 100 for data in results['round_analytics']]

        # Generate Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Federated Learning Round', fontsize=12)
        ax1.set_ylabel('Aggregation Time (ms)', color=color, fontsize=12)
        ax1.plot(rounds, agg_times, color=color, linewidth=2, label='Compute Time')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('% Clients Promoted (Tier 3)', color=color, fontsize=12)
        ax2.plot(rounds, promoted_percentages, color=color, linestyle='--', linewidth=2, label='Promoted Clients')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 100)

        plt.title('Experiment 2: Compute Reduction via Trust Establishment (TAVS-ESP)', fontsize=14)
        fig.tight_layout()

        plot_path = os.path.join(self.output_dir, "exp2_warmup_reduction.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved Experiment 2 plot to {plot_path}")


    def run_experiment_3_pareto(self):
        """Exp 3: Pareto Efficiency Trade-off (Time vs Accuracy)."""
        print("\n>>> Running Experiment 3: Pareto Efficiency Curve")

        gammas = [0.1, 0.3, 0.5, 0.8, 1.0] # 1.0 is Full Verification (Baseline)
        accuracies = []
        compute_times = []

        for gamma in gammas:
            res = self._run_simulation(gamma=gamma, num_rounds=30, num_clients=50, exp_name=f"Exp3_Gamma_{gamma}")
            accuracies.append(res['final_accuracy'] * 100) # Convert to percentage

            # Sum up total aggregation time across all rounds
            total_agg_time = sum(d['aggregation_time_ms'] for d in res['round_analytics']) / 1000.0 # Convert to seconds
            compute_times.append(total_agg_time)

        # Generate Plot
        plt.figure(figsize=(9, 6))

        # Plot points and connect them to form the Pareto front
        plt.plot(compute_times, accuracies, 'bo-', linewidth=2, markersize=8)

        # Annotate each point with its Gamma value
        for i, gamma in enumerate(gammas):
            label = f"TAVS (γ={gamma})" if gamma < 1.0 else "Full Verif (γ=1.0)"
            plt.annotate(label, (compute_times[i], accuracies[i]),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

        plt.xlabel('Total Server Aggregation Time (Seconds)', fontsize=12)
        plt.ylabel('Final Model Test Accuracy (%)', fontsize=12)
        plt.title('Experiment 3: Pareto Efficiency Trade-off (Accuracy vs Compute)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_path = os.path.join(self.output_dir, "exp3_pareto_efficiency.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved Experiment 3 plot to {plot_path}")


if __name__ == "__main__":
    runner = EfficiencyExperimentRunner()

    # Run Exp 2
    runner.run_experiment_2_warmup()

    # Run Exp 3
    runner.run_experiment_3_pareto()

    print("\nAll efficiency experiments completed successfully!")