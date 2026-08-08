import pytest
import torch
import math

# Assuming you save the new aggregator in src/tavs_v2/algo3_bvd_aggregation.py
from src.tavs_v2.algo3_bvd_aggregation import BlockVarianceDetector, UnifiedBayesianAggregator

@pytest.fixture
def bvd_detector():
    """Initializes the BVD Detector with a threshold tau_z = 10.0."""
    return BlockVarianceDetector(
        tau_z=10.0,
        alpha_sigma=0.9,
        epsilon_stab=1e-5
    )

def test_proposition_1_localized_attack_detection(bvd_detector):
    """
    Validates Proposition 1: Non-IID/malicious separation.
    A localized attack on a single block MUST spike Z_i for that block
    and result in the client being classified as an Outlier.
    """
    blocks = ["attention_1", "attention_2", "ffn_1"]
    dim = 50
    
    # 1. Create 9 Honest Clients (Simulating natural non-IID heterogeneity)
    # They have random variations across ALL blocks
    projected_updates = {}
    verified_clients = set()
    
    for i in range(9):
        cid = f"honest_{i}"
        verified_clients.add(cid)
        projected_updates[cid] = {
            m: torch.randn(dim) * 2.0  # Natural variance
            for m in blocks
        }
        
    # 2. Create 1 Byzantine Client (Localized Sleeper Agent Attack)
    # They behave normally in most blocks, but inject massive poison in "attention_2"
    byz_cid = "byzantine_attacker"
    verified_clients.add(byz_cid)
    projected_updates[byz_cid] = {
        "attention_1": torch.randn(dim) * 2.0, # Normal
        "attention_2": torch.randn(dim) * 2.0 + 50.0, # MASSIVE LOCALIZED POISON
        "ffn_1": torch.randn(dim) * 2.0        # Normal
    }
    
    # 3. Run Detection
    inliers, outliers, behavior_scores = bvd_detector.detect_outliers(projected_updates, verified_clients)
    
    # 4. Assertions proving Proposition 1
    assert byz_cid in outliers, "Proposition 1 Failed: Localized attacker evaded detection!"
    assert len(outliers) == 1, "False Positives detected: Honest clients were incorrectly flagged."
    
    # The behavior score \varphi_i(r) for the attacker must be severely penalized (close to 0)
    assert behavior_scores[byz_cid] < 0.1, "Attacker was not penalized heavily enough in Layer 1 trust."
    # Honest clients should have near-perfect behavior scores
    assert behavior_scores["honest_0"] > 0.9, "Honest client behavior score was unfairly degraded."

def test_variance_ema_updated_only_by_inliers(bvd_detector):
    r"""
    Validates Section 4.3 equation: \hat{\sigma}_m^2(r) is updated using ONLY \mathcal{L}(r-1).
    If an attacker submits a massive vector, it must NOT artificially inflate the variance threshold.
    """
    # Create simple 1D scenario
    cid_honest = "honest_1"
    cid_byz = "byzantine_1"
    
    updates = {
        cid_honest: {"block_A": torch.tensor([2.0])},
        cid_byz: {"block_A": torch.tensor([100.0])} # Massive outlier
    }
    
    # Run round 1 (Detector starts with sigma^2 = 1.0)
    inliers, outliers, _ = bvd_detector.detect_outliers(updates, {cid_honest, cid_byz})
    
    assert cid_byz in outliers
    assert cid_honest in inliers
    
    # The variance state MUST reflect ONLY the honest client's distance from the median.
    # The median of [2.0, 100.0] in PyTorch is 2.0. 
    # Distance of honest from median: (2.0 - 2.0)^2 = 0.0.
    # EMA update: 0.9 * 1.0 (old) + 0.1 * 0.0 (new) = 0.9.
    
    # If the attacker was included, the variance would have spiked to ~500!
    assert math.isclose(bvd_detector.sigma_sq["block_A"], 0.9, rel_tol=1e-4), \
        "Variance EMA was poisoned by the attacker! It did not isolate inliers."

def test_bayesian_unified_aggregation_rule():
    r"""
    Validates Algorithm 3b (Section 4.4):
    w(r) = w(r-1) + \eta * [ \sum_{L} g_i + \sum_{S} p_i * g_i ] / Z(r)
    """
    # 1. Setup Data
    # Inlier contributes a gradient of 2.0
    verified_updates = {
        "honest_verified": {"block_1": torch.tensor([2.0])}
    }
    verified_weights = {"honest_verified": 1.0}

    # Promoted contributes a gradient of 10.0
    promoted_updates = {
        "honest_promoted": {"block_1": torch.tensor([10.0])}
    }
    promoted_weights = {"honest_promoted": 0.5}

    # 2. Run Aggregator
    agg_result = UnifiedBayesianAggregator.aggregate(
        verified_updates, verified_weights, promoted_updates, promoted_weights
    )
    
    # 3. Calculate expected math manually:
    # Z(r) = |L| + sum(p_i) = 1 + 0.5 = 1.5
    # Numerator = (1.0 * 2.0) + (0.5 * 10.0) = 2.0 + 5.0 = 7.0
    # Expected Agg = 7.0 / 1.5 = 4.6666...
    
    expected_val = 7.0 / 1.5
    actual_val = agg_result["block_1"].item()
    
    assert math.isclose(actual_val, expected_val, rel_tol=1e-4), \
        f"Unified Aggregator failed. Expected {expected_val}, got {actual_val}"