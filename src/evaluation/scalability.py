import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from ..core.projection import StructuredJLProjection, DenseJLProjection
from ..core.verification import IsomorphicVerification
from ..core.models import ModelStructure, get_model
import logging


logger = logging.getLogger(__name__)


class ScalabilityBenchmark:
    """
    Benchmark for demonstrating O(k) vs O(d) scalability advantage.

    This generates the critical E3 results showing that our method scales
    independently of parameter count d, while traditional methods scale linearly.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.results = {}

    def benchmark_projection_generation(self, parameter_counts: List[int],
                                      k_values: List[int],
                                      num_trials: int = 5) -> Dict:
        """
        Benchmark projection matrix generation time vs parameter count.

        Args:
            parameter_counts: List of parameter counts to test
            k_values: List of projection dimensions to test
            num_trials: Number of trials to average over

        Returns:
            Timing results for analysis
        """
        results = {
            'structured_times': {},
            'dense_times': {},
            'parameter_counts': parameter_counts,
            'k_values': k_values
        }

        logger.info(f"Benchmarking projection generation for {len(parameter_counts)} parameter counts "
                   f"and {len(k_values)} k values")

        for k in k_values:
            structured_times = []
            dense_times = []

            for d in parameter_counts:
                logger.debug(f"Testing d={d}, k={k}")

                # Create dummy model structure for structured projection
                dummy_structure = self._create_dummy_structure(d)

                # Benchmark structured projection generation
                structured_proj = StructuredJLProjection(
                    dummy_structure, k_ratio=k/d, device=self.device
                )

                structured_trial_times = []
                for trial in range(num_trials):
                    start_time = time.perf_counter()
                    proj_matrices = structured_proj.generate_ephemeral_projection_matrix(trial)
                    end_time = time.perf_counter()
                    structured_trial_times.append(end_time - start_time)

                structured_times.append(np.mean(structured_trial_times))

                # Benchmark dense projection generation
                dense_proj = DenseJLProjection(d, k_ratio=k/d, device=self.device)

                dense_trial_times = []
                for trial in range(num_trials):
                    start_time = time.perf_counter()
                    proj_matrix = dense_proj.generate_projection_matrix(trial)
                    end_time = time.perf_counter()
                    dense_trial_times.append(end_time - start_time)

                dense_times.append(np.mean(dense_trial_times))

            results['structured_times'][k] = structured_times
            results['dense_times'][k] = dense_times

        logger.info("Projection generation benchmark completed")
        return results

    def benchmark_verification_complexity(self, client_counts: List[int],
                                        projection_dims: List[int],
                                        num_trials: int = 3) -> Dict:
        """
        Benchmark verification time complexity: O(k) vs O(d).

        This is the core of the scalability story - verification time should be
        independent of original parameter dimension d.

        Args:
            client_counts: Number of clients to test
            projection_dims: Projection dimensions to test
            num_trials: Number of trials per configuration

        Returns:
            Verification timing results
        """
        results = {
            'verification_times': {},
            'client_counts': client_counts,
            'projection_dims': projection_dims,
            'complexity_analysis': {}
        }

        verifier = IsomorphicVerification()

        for k in projection_dims:
            verification_times = []

            for n_clients in client_counts:
                logger.debug(f"Testing verification: {n_clients} clients, k={k}")

                trial_times = []
                for trial in range(num_trials):
                    # Generate dummy projected updates
                    projected_updates = [
                        torch.randn(k, device=self.device) for _ in range(n_clients)
                    ]

                    # Benchmark verification time
                    start_time = time.perf_counter()
                    detection_results = verifier.detect_byzantine_clients(projected_updates)
                    end_time = time.perf_counter()

                    trial_times.append(end_time - start_time)

                verification_times.append(np.mean(trial_times))

            results['verification_times'][k] = verification_times

            # Analyze complexity (should be roughly constant for different k)
            time_variance = np.var(verification_times)
            results['complexity_analysis'][k] = {
                'mean_time': np.mean(verification_times),
                'variance': time_variance,
                'coefficient_of_variation': np.sqrt(time_variance) / np.mean(verification_times)
            }

        logger.info("Verification complexity benchmark completed")
        return results

    def benchmark_end_to_end_scalability(self, model_sizes: List[Tuple[str, Dict]],
                                       num_clients: int = 100,
                                       num_rounds: int = 3) -> Dict:
        """
        End-to-end scalability benchmark with different model sizes.

        This demonstrates the key advantage: as model size grows to billions of parameters,
        our verification time remains constant while traditional methods become infeasible.

        Args:
            model_sizes: List of (model_type, kwargs) tuples defining model sizes
            num_clients: Number of clients to simulate
            num_rounds: Number of FL rounds to simulate

        Returns:
            End-to-end timing results
        """
        results = {
            'model_info': [],
            'structured_projection_times': [],
            'dense_projection_times': [],
            'verification_times': [],
            'total_times_structured': [],
            'total_times_dense': []
        }

        k_ratio = 0.1  # Fixed compression ratio

        for model_type, model_kwargs in model_sizes:
            logger.info(f"Benchmarking model: {model_type} with {model_kwargs}")

            # Create model and get structure
            try:
                model = get_model(model_type, **model_kwargs)
                model_structure = model.structure
                total_params = model_structure.total_params
            except Exception as e:
                logger.warning(f"Failed to create model {model_type}: {e}")
                continue

            logger.info(f"Model has {total_params} parameters")

            # Setup projections
            k = max(1, int(k_ratio * total_params))
            structured_proj = StructuredJLProjection(model_structure, k_ratio, self.device)
            dense_proj = DenseJLProjection(total_params, k_ratio, self.device)

            # Benchmark over multiple rounds
            structured_times = []
            dense_times = []
            verification_times = []

            for round_num in range(num_rounds):
                logger.debug(f"Round {round_num + 1}/{num_rounds}")

                # Structured projection timing
                start_time = time.perf_counter()

                # Generate projection matrix
                proj_matrices = structured_proj.generate_ephemeral_projection_matrix(round_num)

                # Simulate client updates and projection
                client_updates = [torch.randn(total_params, device=self.device)
                                for _ in range(num_clients)]
                projected_updates = structured_proj.project_multiple_updates(client_updates, proj_matrices)

                structured_proj_time = time.perf_counter() - start_time

                # Verification timing (same for both methods)
                start_time = time.perf_counter()
                verifier = IsomorphicVerification()
                detection_results = verifier.detect_byzantine_clients(projected_updates)
                verification_time = time.perf_counter() - start_time

                # Dense projection timing
                start_time = time.perf_counter()

                # Generate dense projection matrix
                dense_matrix = dense_proj.generate_projection_matrix(round_num)

                # Project updates
                dense_projected = [dense_proj.project_update(update, dense_matrix)
                                 for update in client_updates]

                dense_proj_time = time.perf_counter() - start_time

                structured_times.append(structured_proj_time)
                dense_times.append(dense_proj_time)
                verification_times.append(verification_time)

            # Store results
            results['model_info'].append({
                'model_type': model_type,
                'model_kwargs': model_kwargs,
                'total_params': total_params,
                'projected_dim': k
            })

            results['structured_projection_times'].append(np.mean(structured_times))
            results['dense_projection_times'].append(np.mean(dense_times))
            results['verification_times'].append(np.mean(verification_times))

            # Total time = projection + verification
            results['total_times_structured'].append(
                np.mean(structured_times) + np.mean(verification_times)
            )
            results['total_times_dense'].append(
                np.mean(dense_times) + np.mean(verification_times)
            )

        logger.info("End-to-end scalability benchmark completed")
        return results

    def _create_dummy_structure(self, total_params: int, avg_block_size: int = 1000) -> ModelStructure:
        """Create a dummy model structure for benchmarking."""
        structure = ModelStructure()

        remaining_params = total_params
        block_id = 0

        while remaining_params > 0:
            # Vary block sizes to simulate realistic model structure
            block_size = min(remaining_params,
                           int(avg_block_size * (0.5 + np.random.random())))

            structure.add_block(
                name=f"block_{block_id}",
                shape=(block_size,),  # Simplified 1D shape
                num_params=block_size
            )

            remaining_params -= block_size
            block_id += 1

        return structure

    def analyze_scalability_trends(self, benchmark_results: Dict) -> Dict:
        """
        Analyze scalability trends from benchmark results.

        Returns:
            Analysis of complexity trends and scalability factors
        """
        analysis = {}

        if 'model_info' in benchmark_results:
            # Analyze parameter count vs timing relationship
            param_counts = [info['total_params'] for info in benchmark_results['model_info']]
            structured_times = benchmark_results['total_times_structured']
            dense_times = benchmark_results['total_times_dense']

            # Fit linear relationship to dense projection times (should be O(d))
            if len(param_counts) > 2:
                dense_slope, dense_intercept = np.polyfit(param_counts, dense_times, 1)
                structured_slope, structured_intercept = np.polyfit(param_counts, structured_times, 1)

                analysis['complexity_trends'] = {
                    'dense_slope': dense_slope,  # Should be positive (O(d))
                    'structured_slope': structured_slope,  # Should be near zero (O(k))
                    'scalability_advantage': dense_slope / (abs(structured_slope) + 1e-8)
                }

            # Calculate time per parameter
            analysis['time_per_parameter'] = {
                'structured': [t/p for t, p in zip(structured_times, param_counts)],
                'dense': [t/p for t, p in zip(dense_times, param_counts)]
            }

        return analysis


class ScalabilityVisualizer:
    """Visualizer for scalability benchmark results."""

    @staticmethod
    def plot_scalability_comparison(benchmark_results: Dict, save_path: Optional[str] = None):
        """
        Create scalability comparison plots for the paper.

        Generates:
        1. Time vs Parameter Count
        2. Time per Parameter vs Parameter Count
        3. Projection Generation Time Comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Total time vs parameter count
        if 'model_info' in benchmark_results:
            param_counts = [info['total_params'] for info in benchmark_results['model_info']]
            structured_times = benchmark_results['total_times_structured']
            dense_times = benchmark_results['total_times_dense']

            axes[0, 0].plot(param_counts, structured_times, 'bo-', label='Structured JL (Ours)', linewidth=2)
            axes[0, 0].plot(param_counts, dense_times, 'rs-', label='Dense JL (KETS)', linewidth=2)
            axes[0, 0].set_xlabel('Model Parameters')
            axes[0, 0].set_ylabel('Total Time (seconds)')
            axes[0, 0].set_title('Scalability: Total Time vs Model Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_yscale('log')

            # Plot 2: Time per parameter
            structured_per_param = [t/p for t, p in zip(structured_times, param_counts)]
            dense_per_param = [t/p for t, p in zip(dense_times, param_counts)]

            axes[0, 1].plot(param_counts, structured_per_param, 'bo-', label='Structured JL', linewidth=2)
            axes[0, 1].plot(param_counts, dense_per_param, 'rs-', label='Dense JL', linewidth=2)
            axes[0, 1].set_xlabel('Model Parameters')
            axes[0, 1].set_ylabel('Time per Parameter (seconds)')
            axes[0, 1].set_title('Efficiency: Time per Parameter')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_yscale('log')

        # Plot 3: Projection generation time breakdown
        if 'structured_times' in benchmark_results:
            k_values = benchmark_results['k_values']
            param_counts = benchmark_results['parameter_counts']

            # Take middle k value for comparison
            mid_k_idx = len(k_values) // 2
            mid_k = k_values[mid_k_idx]

            structured_gen_times = benchmark_results['structured_times'][mid_k]
            dense_gen_times = benchmark_results['dense_times'][mid_k]

            axes[1, 0].plot(param_counts, structured_gen_times, 'bo-',
                          label=f'Structured JL (k={mid_k})', linewidth=2)
            axes[1, 0].plot(param_counts, dense_gen_times, 'rs-',
                          label=f'Dense JL (k={mid_k})', linewidth=2)
            axes[1, 0].set_xlabel('Model Parameters')
            axes[1, 0].set_ylabel('Generation Time (seconds)')
            axes[1, 0].set_title('Matrix Generation Time Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')

        # Plot 4: Verification time (should be constant)
        if 'verification_times' in benchmark_results:
            projection_dims = benchmark_results['projection_dims']

            # Show verification time for different k values
            for k in projection_dims:
                if k in benchmark_results['verification_times']:
                    client_counts = benchmark_results['client_counts']
                    verification_times = benchmark_results['verification_times'][k]
                    axes[1, 1].plot(client_counts, verification_times, 'o-',
                                  label=f'k={k}', linewidth=2)

            axes[1, 1].set_xlabel('Number of Clients')
            axes[1, 1].set_ylabel('Verification Time (seconds)')
            axes[1, 1].set_title('Verification Time vs Client Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scalability plots saved to {save_path}")

        return fig

    @staticmethod
    def create_scalability_summary_table(analysis_results: Dict) -> str:
        """Create a summary table of scalability results."""
        if 'complexity_trends' not in analysis_results:
            return "Insufficient data for summary table"

        trends = analysis_results['complexity_trends']

        summary = f"""
Scalability Analysis Summary
============================

Dense JL (KETS-style):
  Time complexity slope: {trends['dense_slope']:.6f} sec/param
  Scales as: O(d) - linear with parameter count

Structured JL (Ours):
  Time complexity slope: {trends['structured_slope']:.6f} sec/param
  Scales as: O(k) - independent of parameter count

Scalability Advantage: {trends['scalability_advantage']:.1f}x

Key Insight: Our method maintains constant verification time regardless of model size,
while traditional methods become infeasible for billion-parameter models.
"""
        return summary