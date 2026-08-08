import pytest
import math
# Assuming you save the new scheduler in src/tavs_v2/algo1_tavs_scheduler.py
from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler

@pytest.fixture
def baseline_scheduler():
    """Provides a scheduler configured with the exact defaults from Section 5.1 of the paper."""
    return TavsScheduler(
        gamma_budget=0.35,
        theta_low=0.3,
        theta_high=0.8,
        alpha_trust=0.9,
        tau_ramp=30.0,
        k_trust=10,
        p_decoy=0.15,
        c_lambda=8.0,
        master_key=b'neurips_test_key'
    )

def test_mechanism_3_trust_initialization(baseline_scheduler):
    r"""
    Validates Mechanism 3: T_i^{max}(r) = 1 - \exp(-(r-r_0)/\tau_{ramp})
    New clients must be rate-limited regardless of their raw behavior.
    """
    # Simulate round 1
    V, P, D = baseline_scheduler.schedule_verifications(["client_new"], round_num=1)
    
    # Raw trust initialized to 0.5, but effective trust at r=1 is 1 - exp(0) = 0.0
    effective_t = baseline_scheduler.get_effective_trust("client_new", 1)
    assert effective_t == 0.0, "Mechanism 3 failed: New client trust not capped at 0.0 in round 1"
    
    # At round 10, cap = 1 - exp(-9/30) ~= 0.259.
    # Effective trust is min(raw_trust, cap). Raw trust (0.25) < cap (0.259),
    # so the raw trust is the binding constraint here — Mechanism 3 only
    # activates when the cap is *lower* than raw trust.
    effective_t_10 = baseline_scheduler.get_effective_trust("client_new", 10)
    expected_cap = 1.0 - math.exp(-9.0 / 30.0)
    raw_trust = baseline_scheduler.trust_scores["client_new"]
    expected_effective = min(raw_trust, expected_cap)
    assert math.isclose(effective_t_10, expected_effective, rel_tol=1e-4), \
        f"Mechanism 3 failed: expected min(raw={raw_trust}, cap={expected_cap})={expected_effective}, got {effective_t_10}"
    
    # Because effective trust < theta_low (0.3), it MUST be in Tier 1 (Verified)
    assert "client_new" in V

def test_tier_3_promotion_and_decoy(baseline_scheduler, monkeypatch):
    """
    Validates Tier 3 logic: Requires BOTH T_i > theta_high AND k_trust streak.
    Also tests CSPRNG decoy injection.
    """
    cid = "trusted_client"
    # Artificially inject high trust and long streak
    baseline_scheduler.trust_scores[cid] = 0.9
    baseline_scheduler.join_rounds[cid] = -100  # Bypass Mech 3 cap
    baseline_scheduler.clean_streaks[cid] = 15  # Surpasses k_trust = 10
    baseline_scheduler.gamma_budget = 1.0  # isolate Tier 3 / decoy from Mechanism 1 demotion

    # Test Promotion (Mock CSPRNG to return a high value > p_decoy)
    monkeypatch.setattr(baseline_scheduler, "_csprng_roll", lambda c, r: 0.99)
    V, P, D = baseline_scheduler.schedule_verifications([cid], round_num=1)
    assert cid in P, "Tier 3 client was not promoted despite passing CSPRNG roll"
    
    # Test Decoy Verification (Mock CSPRNG to return a low value < p_decoy)
    monkeypatch.setattr(baseline_scheduler, "_csprng_roll", lambda c, r: 0.05)
    V, P, D = baseline_scheduler.schedule_verifications([cid], round_num=2)
    assert cid in V, "Tier 3 client evaded the CSPRNG decoy check"

def test_mechanism_1_budget_constraint(baseline_scheduler):
    """
    Validates Mechanism 1: sum(p_i) / Z_r <= gamma_budget.
    If the budget is exceeded, lowest trusted promoted clients are demoted.
    """
    # Create 10 clients who all qualify for Tier 2/3 and bypass Mech 3 cap
    clients = [f"client_{i}" for i in range(10)]
    for i, cid in enumerate(clients):
        # Give them staggered trust scores: 0.70, 0.71, ..., 0.79
        baseline_scheduler.trust_scores[cid] = 0.70 + (i * 0.01)
        baseline_scheduler.join_rounds[cid] = -100
        baseline_scheduler.clean_streaks[cid] = 10
        
    # We set gamma_budget to a tiny value to force demotions
    baseline_scheduler.gamma_budget = 0.10
    
    V, P, D = baseline_scheduler.schedule_verifications(clients, round_num=1)
    
    # With a 10% budget, the scheduler MUST demote several clients to V
    assert len(V) > 0, "Mechanism 1 failed: Allowed too many promoted clients"
    assert len(P) > 0, "Mechanism 1 failed: Dropped everyone unnecessarily"
    
    # CRITICAL PAPER CHECK: Were the *lowest* trusted clients the ones demoted?
    # client_0 has 0.70, client_9 has 0.79. Client 0 should be in V, client 9 in P.
    assert "client_0" in V, "Mechanism 1 failed: Did not demote lowest trust client"
    assert "client_9" in P, "Mechanism 1 failed: Highest trust client should survive promotion"

def test_trust_ema_update_and_streaks(baseline_scheduler):
    """
    Validates the trust update rule and the k_trust tracking logic.
    """
    cid = "test_client"
    baseline_scheduler.trust_scores[cid] = 0.5
    baseline_scheduler.clean_streaks[cid] = 5
    
    # 1. Good Verification
    baseline_scheduler.update_trust(cid, behavior_score=1.0, was_verified=True)
    # expected: 0.9 * 0.5 + 0.1 * 1.0 = 0.45 + 0.10 = 0.55
    assert math.isclose(baseline_scheduler.trust_scores[cid], 0.55)
    assert baseline_scheduler.clean_streaks[cid] == 6, "Clean streak did not increment"
    
    # 2. Promoted Decay (No verification)
    baseline_scheduler.update_trust(cid, behavior_score=0.0, was_verified=False)
    # expected: 0.9 * 0.55 = 0.495
    assert math.isclose(baseline_scheduler.trust_scores[cid], 0.495)
    assert baseline_scheduler.clean_streaks[cid] == 6, "Clean streak should pause, not reset, during promotion"
    
    # 3. Byzantine Verification (Attack detected)
    baseline_scheduler.update_trust(cid, behavior_score=0.1, was_verified=True)
    # expected: 0.9 * 0.495 + 0.1 * 0.1 = 0.4455 + 0.01 = 0.4555
    assert math.isclose(baseline_scheduler.trust_scores[cid], 0.4555)
    assert baseline_scheduler.clean_streaks[cid] == 0, "Clean streak did not reset after bad behavior"
