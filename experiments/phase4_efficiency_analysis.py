import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Exact imports based on your repository structure
from src.tavs_v2 import TavsEspStrategy
from src.tavs_v2 import PipelineConfig, TavsEspConfig
from src.tavs_v2 import TAVSESPPipeline

class EfficiencyExperimentRunner:
    """Runs Phase 4 efficiency and scalability experiments for TAVS-ESP."""

    def __init__(self, output_dir: str = "./results/efficiency"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_simulation(self, gamma: float, num_rounds: int, num_clients: int, exp_name: str):
        """Helper to run a Flower simulation with a specific TAVS budget."""
        print(f"\n--- Starting {exp_name} (Gamma: {gamma}) ---")

        # 1. Create the specific TAVS config matching your EXACT class signature
        tavs_config = TavsEspConfig(
            projection_type="structured",
            # BVD runs on JL-compressed vectors; target_k=150 across ~10 blocks ⇒ tiny k_m
            # and huge variance → almost all honest clients marked outliers and dropped.
            detection_threshold=25.0,
            target_k=2048,
            theta_low=0.3,            # Tier 1 threshold
            theta_high=0.8,           # Tier 3 threshold
            gamma_budget=gamma,        # The active budget constraint for this run
            tau_ramp= 30,        # Trust ramp-up parameter (Sybil resistance)
            decoy_probability= 0.15
        )

        # 2. Wrap it in the PipelineConfig matching your EXACT class signature
        pipeline_config = PipelineConfig(
            num_rounds=num_rounds,
            num_clients=num_clients,
            clients_per_round=num_clients,  # Sample everyone
            byzantine_fraction=0.1,         # 10% attackers
            model_type="cifar_cnn",
            dataset="cifar10",
            output_dir=self.output_dir,
            tavs_config=tavs_config,
            simulation_backend="thread"
        )
        # 3. Initialize the Strategy safely
        strategy = TavsEspStrategy(config=tavs_config)

        # 4. Start Simulation (Using mock timing for instantaneous plot generation)
        start_time = time.time()

        # 3. Launch the REAL PyTorch Training Loop via your native Pipeline
        print("Launching PyTorch Training Loop... this will take a while!")
        pipeline = TAVSESPPipeline(pipeline_config)

        # This single call partitions the CIFAR-10 data, spawns the clients, and runs Flower
        results = pipeline.run_simulation()

        print(f"REAL Simulation completed in {results.total_time_seconds:.2f} seconds.")

        # 4. Extract REAL Data for the Plots
        final_accuracy = results.server_accuracies[-1] if results.server_accuracies else 0.0

        # Calculate promoted clients per round from the tier_evolution dictionary
        # Tier 3 clients are the ones that bypassed verification
        promoted_counts = [0] * num_rounds
        if results.tier_evolution:
            for cid, tiers in results.tier_evolution.items():
                for r_idx, tier in enumerate(tiers):
                    if tier == 3 and r_idx < num_rounds:
                        promoted_counts[r_idx] += 1

        # Map the extracted data to the format the plotting function expects
        real_analytics = []
        for r in range(num_rounds):
            # results.round_times contains the exact ms spent on projection/detection
            time_ms = results.round_times[r] if r < len(results.round_times) else 0.0

            real_analytics.append({
                "round": r + 1,
                "num_promoted": promoted_counts[r],
                "aggregation_time_ms": time_ms
            })

        return {
            "total_time_seconds": results.total_time_seconds,
            "final_accuracy": final_accuracy,
            "round_analytics": real_analytics
        }

    def run_experiment_2_warmup(self):
        """Exp 2: Compute reduction as trust is established."""
        print("\n>>> Running Experiment 2: Warm-Up Compute Reduction")
        num_rounds = 50
        num_clients = 20

        # Run with 30% verification budget
        results = self._run_simulation(gamma=0.4, num_rounds=num_rounds, num_clients=num_clients, exp_name="Exp2_WarmUp")

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

        #gammas = [0.1, 0.3, 0.5, 0.8, 1.0] # 1.0 is Full Verification (Baseline)
        gammas = [0.5]
        accuracies = []
        compute_times = []

        for gamma in gammas:
            res = self._run_simulation(gamma=gamma, num_rounds=10, num_clients=20, exp_name=f"Exp3_Gamma_{gamma}")
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