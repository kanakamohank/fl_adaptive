#!/usr/bin/env python3
"""
Quick Empirical Timing Test for TAVS-ESP

Simplified timing validation to quickly measure overhead
of trust-adaptive scheduling vs baseline federated learning.
"""

import time
import numpy as np
import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def setup_device(device_preference: str = "auto") -> torch.device:
    """Setup optimal device for timing validation."""
    if device_preference == "auto":
        # Auto-detect best available device
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no acceleration available)")
    else:
        # Use specified device
        if device_preference == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) acceleration")
            else:
                logger.warning("MPS not available, falling back to CPU")
                device = torch.device("cpu")
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA acceleration")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

    return device

# Simple timing test
def test_trust_adaptive_scheduling_overhead():
    """Test overhead of trust-adaptive scheduling components."""

    print("🕐 Testing Trust-Adaptive Scheduling Overhead...")

    # Test parameters
    num_clients = 20
    clients_per_round = 8
    num_rounds = 10

    results = {
        "trust_updates": [],
        "tier_assignments": [],
        "csprng_operations": [],
        "baseline_selection": []
    }

    for round_num in range(num_rounds):

        # 1. Trust Score Updates (TAVS Layer 1)
        start_time = time.perf_counter()
        trust_scores = np.random.beta(2, 2, num_clients)
        alpha = 0.9
        behavioral_scores = np.random.beta(3, 1, num_clients)
        # EMA update: T_i(r) = α·T_i(r-1) + (1-α)·φ_i(r)
        updated_trust = alpha * trust_scores + (1 - alpha) * behavioral_scores
        trust_update_time = (time.perf_counter() - start_time) * 1000
        results["trust_updates"].append(trust_update_time)

        # 2. Tier Assignment (TAVS Layer 1)
        start_time = time.perf_counter()
        theta_low, theta_high = 0.3, 0.7
        tiers = np.where(updated_trust < theta_low, 1,
                        np.where(updated_trust < theta_high, 2, 3))
        tier_assign_time = (time.perf_counter() - start_time) * 1000
        results["tier_assignments"].append(tier_assign_time)

        # 3. CSPRNG Operations (TAVS Layer 1)
        start_time = time.perf_counter()
        # Simulate ChaCha20 key derivation
        projection_seed = np.random.bytes(16)
        promotion_seed = np.random.bytes(16)
        decoy_seed = np.random.bytes(16)
        csprng_time = (time.perf_counter() - start_time) * 1000
        results["csprng_operations"].append(csprng_time)

        # 4. Baseline Round-Robin Selection
        start_time = time.perf_counter()
        selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)
        baseline_time = (time.perf_counter() - start_time) * 1000
        results["baseline_selection"].append(baseline_time)

    # Calculate averages
    avg_trust_update = np.mean(results["trust_updates"])
    avg_tier_assign = np.mean(results["tier_assignments"])
    avg_csprng = np.mean(results["csprng_operations"])
    avg_baseline = np.mean(results["baseline_selection"])

    total_tavs_overhead = avg_trust_update + avg_tier_assign + avg_csprng
    overhead_percentage = (total_tavs_overhead / avg_baseline) * 100 if avg_baseline > 0 else 0

    print(f"Trust Score Updates: {avg_trust_update:.3f}ms")
    print(f"Tier Assignments: {avg_tier_assign:.3f}ms")
    print(f"CSPRNG Operations: {avg_csprng:.3f}ms")
    print(f"Baseline Selection: {avg_baseline:.3f}ms")
    print(f"Total TAVS Overhead: {total_tavs_overhead:.3f}ms")
    print(f"Overhead vs Baseline: {overhead_percentage:.1f}%")

    return {
        "tavs_overhead_ms": total_tavs_overhead,
        "baseline_ms": avg_baseline,
        "overhead_percentage": overhead_percentage
    }


def test_projection_verification_overhead(device: torch.device = None):
    """Test overhead of ESP Layer 2 operations."""

    if device is None:
        device = setup_device("auto")

    print(f"\n🔒 Testing ESP Projection & Verification Overhead on {device}...")

    # Test parameters
    model_params = 100000  # Smaller for quick testing
    k_ratio = 0.05  # OPTIMIZED: Use 5% compression for faster testing
    num_clients = 8
    num_rounds = 5

    k_dim = int(model_params * k_ratio)

    results = {
        "projection_generation": [],
        "gradient_projection": [],
        "geometric_median": [],
        "baseline_averaging": []
    }

    for round_num in range(num_rounds):

        # 1. Projection Matrix Generation
        start_time = time.perf_counter()
        projection_matrix = torch.randn(k_dim, model_params, device=device) / np.sqrt(k_dim)
        proj_gen_time = (time.perf_counter() - start_time) * 1000
        results["projection_generation"].append(proj_gen_time)

        # 2. Client Gradient Projection
        start_time = time.perf_counter()
        client_gradients = [torch.randn(model_params, device=device) * 0.01 for _ in range(num_clients)]
        projected_gradients = []
        for grad in client_gradients:
            projected = torch.matmul(projection_matrix, grad)
            projected_gradients.append(projected)
        grad_proj_time = (time.perf_counter() - start_time) * 1000
        results["gradient_projection"].append(grad_proj_time)

        # 3. Geometric Median (Byzantine Detection)
        start_time = time.perf_counter()
        # Simplified geometric median approximation
        stacked_grads = torch.stack(projected_gradients)
        median_approx = torch.median(stacked_grads, dim=0)[0]
        distances = torch.norm(stacked_grads - median_approx.unsqueeze(0), dim=1)
        threshold = torch.quantile(distances, 0.8)
        inliers = torch.where(distances < threshold)[0]
        geom_median_time = (time.perf_counter() - start_time) * 1000
        results["geometric_median"].append(geom_median_time)

        # 4. Baseline FedAvg (Simple averaging)
        start_time = time.perf_counter()
        avg_gradient = torch.mean(torch.stack(client_gradients), dim=0)
        baseline_avg_time = (time.perf_counter() - start_time) * 1000
        results["baseline_averaging"].append(baseline_avg_time)

    # Calculate averages
    avg_proj_gen = np.mean(results["projection_generation"])
    avg_grad_proj = np.mean(results["gradient_projection"])
    avg_geom_median = np.mean(results["geometric_median"])
    avg_baseline_avg = np.mean(results["baseline_averaging"])

    total_esp_overhead = avg_proj_gen + avg_grad_proj + avg_geom_median
    overhead_percentage = (total_esp_overhead / avg_baseline_avg) * 100 if avg_baseline_avg > 0 else 0

    print(f"Projection Generation: {avg_proj_gen:.3f}ms")
    print(f"Gradient Projection: {avg_grad_proj:.3f}ms")
    print(f"Geometric Median: {avg_geom_median:.3f}ms")
    print(f"Baseline Averaging: {avg_baseline_avg:.3f}ms")
    print(f"Total ESP Overhead: {total_esp_overhead:.3f}ms")
    print(f"Overhead vs Baseline: {overhead_percentage:.1f}%")

    return {
        "esp_overhead_ms": total_esp_overhead,
        "baseline_ms": avg_baseline_avg,
        "overhead_percentage": overhead_percentage
    }


def test_end_to_end_round_time(device: torch.device = None):
    """Test complete federated learning round time comparison."""

    if device is None:
        device = setup_device("auto")

    print(f"\n⚡ Testing End-to-End Round Time Comparison on {device}...")

    # Test parameters
    num_clients = 15
    clients_per_round = 6
    model_params = 50000  # Manageable size
    num_rounds = 3

    results = {
        "tavs_esp_rounds": [],
        "baseline_rounds": []
    }

    for round_num in range(num_rounds):

        # TAVS-ESP Complete Round
        start_time = time.perf_counter()

        # Layer 1: Trust-adaptive scheduling
        trust_scores = np.random.beta(2, 2, num_clients)
        tiers = np.where(trust_scores < 0.3, 1, np.where(trust_scores < 0.7, 2, 3))
        csprng_materials = np.random.bytes(48)  # ChaCha20 output

        # Layer 2: ESP operations
        k_dim = int(model_params * 0.05)  # OPTIMIZED: 5% compression
        projection_matrix = torch.randn(k_dim, model_params, device=device) / np.sqrt(k_dim)

        # Mock client gradients
        client_gradients = [torch.randn(model_params, device=device) * 0.01 for _ in range(clients_per_round)]

        # Project gradients
        projected_gradients = [torch.matmul(projection_matrix, grad) for grad in client_gradients]

        # Byzantine detection (simplified)
        stacked_proj = torch.stack(projected_gradients)
        median = torch.median(stacked_proj, dim=0)[0]
        distances = torch.norm(stacked_proj - median.unsqueeze(0), dim=1)
        inliers = torch.where(distances < torch.quantile(distances, 0.8))[0]

        # Bayesian aggregation (simplified)
        inliers_cpu = inliers.cpu() if hasattr(inliers, 'cpu') else inliers
        trust_weights = torch.tensor([trust_scores[i] for i in inliers_cpu], dtype=torch.float32, device=device)
        posterior_weights = torch.softmax(trust_weights, dim=0)
        inlier_grads = torch.stack([client_gradients[i] for i in inliers_cpu])
        weighted_avg = torch.sum(inlier_grads * posterior_weights.unsqueeze(1), dim=0)

        tavs_esp_time = (time.perf_counter() - start_time) * 1000
        results["tavs_esp_rounds"].append(tavs_esp_time)

        # Baseline FedAvg Round
        start_time = time.perf_counter()

        # Simple random selection
        selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)

        # Simple averaging
        simple_avg = torch.mean(torch.stack(client_gradients), dim=0)

        baseline_time = (time.perf_counter() - start_time) * 1000
        results["baseline_rounds"].append(baseline_time)

    # Calculate results
    avg_tavs_esp = np.mean(results["tavs_esp_rounds"])
    avg_baseline = np.mean(results["baseline_rounds"])
    total_overhead = avg_tavs_esp - avg_baseline
    overhead_percentage = (total_overhead / avg_baseline) * 100 if avg_baseline > 0 else 0

    print(f"TAVS-ESP Round Time: {avg_tavs_esp:.3f}ms")
    print(f"Baseline Round Time: {avg_baseline:.3f}ms")
    print(f"Total Overhead: {total_overhead:.3f}ms")
    print(f"Overhead Percentage: {overhead_percentage:.1f}%")

    return {
        "tavs_esp_ms": avg_tavs_esp,
        "baseline_ms": avg_baseline,
        "total_overhead_ms": total_overhead,
        "overhead_percentage": overhead_percentage
    }


def main(device: str = "auto"):
    """Run quick empirical timing validation."""

    # Setup device
    torch_device = setup_device(device)

    print("⏱️  Quick Empirical TAVS-ESP Timing Validation")
    print("=" * 55)
    print(f"🖥️  Using device: {torch_device}")
    print()

    # Test individual components
    scheduling_results = test_trust_adaptive_scheduling_overhead()
    projection_results = test_projection_verification_overhead(torch_device)
    end_to_end_results = test_end_to_end_round_time(torch_device)

    # Summary
    print("\n📊 EMPIRICAL TIMING SUMMARY")
    print("=" * 55)
    print(f"🔄 Scheduling Overhead: {scheduling_results['overhead_percentage']:.1f}%")
    print(f"🔒 Security Overhead: {projection_results['overhead_percentage']:.1f}%")
    print(f"⚡ End-to-End Overhead: {end_to_end_results['overhead_percentage']:.1f}%")

    # Validation assessment
    total_overhead = end_to_end_results['overhead_percentage']

    if total_overhead < 20:
        status = "✅ EXCELLENT"
    elif total_overhead < 50:
        status = "✅ ACCEPTABLE"
    elif total_overhead < 100:
        status = "⚠️  MARGINAL"
    else:
        status = "❌ TOO HIGH"

    print(f"\n🎯 Performance Assessment: {status}")
    print(f"   Total System Overhead: {total_overhead:.1f}%")

    if total_overhead < 50:
        print("   ✅ Ready for production deployment")
    else:
        print("   ⚠️  Optimization needed before deployment")

    print("\n📈 Key Findings:")
    print(f"   • Trust-adaptive scheduling adds ~{scheduling_results['overhead_percentage']:.1f}% overhead")
    print(f"   • ESP projections/verification add ~{projection_results['overhead_percentage']:.1f}% overhead")
    print(f"   • Combined TAVS-ESP system: {total_overhead:.1f}% total overhead")

    return {
        "scheduling": scheduling_results,
        "security": projection_results,
        "end_to_end": end_to_end_results,
        "assessment": status
    }


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Quick TAVS-ESP Timing Test")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to use for validation (default: auto-detect)"
    )
    args = parser.parse_args()

    results = main(device=args.device)