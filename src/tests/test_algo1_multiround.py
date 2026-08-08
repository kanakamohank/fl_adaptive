import pytest
import math
import hashlib
from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler


@pytest.fixture
def baseline_scheduler():
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


def test_honest_vs_byzantine_trust_separation(baseline_scheduler):
    """
    Multi-round convergence: honest clients' trust must stay consistently
    above byzantine clients' trust, demonstrating separation.

    Note on dynamics: promoted (unverified) clients receive pure decay
    (T_new = alpha * T_old), so honest clients oscillate around the Tier 1/2
    boundary (~0.28-0.35) rather than converging to 1.0.  Byzantine clients
    converge toward ~0.10 because their low behavior_score pulls trust down
    on every verified round.  The test validates that this gap is stable
    and that honest trust never drops to byzantine levels.
    """
    honest_ids = [f"honest_{i}" for i in range(5)]
    byz_ids = [f"byz_{i}" for i in range(3)]
    all_clients = honest_ids + byz_ids

    baseline_scheduler.schedule_verifications(all_clients, round_num=0)

    num_rounds = 100

    for r in range(1, num_rounds + 1):
        V, P, D = baseline_scheduler.schedule_verifications(all_clients, round_num=r)

        for cid in all_clients:
            was_verified = cid in V
            if cid.startswith("honest"):
                baseline_scheduler.update_trust(cid, behavior_score=0.95, was_verified=was_verified)
            else:
                baseline_scheduler.update_trust(cid, behavior_score=0.1, was_verified=was_verified)

    for cid in byz_ids:
        t = baseline_scheduler.trust_scores[cid]
        assert t < 0.15, f"Byzantine {cid} trust {t:.4f} did not converge below 0.15 after {num_rounds} rounds"

    honest_min = min(baseline_scheduler.trust_scores[c] for c in honest_ids)
    byz_max = max(baseline_scheduler.trust_scores[c] for c in byz_ids)
    gap = honest_min - byz_max
    assert gap > 0.10, (
        f"Trust separation insufficient: honest_min={honest_min:.4f}, byz_max={byz_max:.4f}, gap={gap:.4f}"
    )

    for cid in byz_ids:
        t_eff = baseline_scheduler.get_effective_trust(cid, num_rounds)
        assert t_eff < baseline_scheduler.theta_low, (
            f"Byzantine {cid} effective trust {t_eff:.4f} not below theta_low={baseline_scheduler.theta_low}"
        )


def test_csprng_roll_determinism(baseline_scheduler):
    """
    _csprng_roll is a pure SHA-256 hash: same (client_id, round_num) must
    always produce the same output, regardless of how many times it is called
    or what other state has changed.
    """
    cid = "determinism_client"
    round_num = 42

    results = [baseline_scheduler._csprng_roll(cid, round_num) for _ in range(20)]

    assert all(r == results[0] for r in results), (
        f"_csprng_roll returned different values for same inputs: {set(results)}"
    )

    baseline_scheduler.trust_scores["some_other_client"] = 0.99
    baseline_scheduler.clean_streaks["some_other_client"] = 100
    assert baseline_scheduler._csprng_roll(cid, round_num) == results[0], (
        "_csprng_roll output changed after unrelated state mutation"
    )


def test_csprng_roll_varies_across_rounds(baseline_scheduler):
    """
    Same client in different rounds must get different rolls — otherwise
    a strategic adversary could predict verification scheduling.
    """
    cid = "round_variation_client"
    rolls = {baseline_scheduler._csprng_roll(cid, r) for r in range(100)}
    assert len(rolls) == 100, (
        f"Expected 100 distinct rolls across 100 rounds, got {len(rolls)}"
    )


def test_csprng_roll_varies_across_clients(baseline_scheduler):
    """Different clients in the same round must get different rolls."""
    round_num = 7
    rolls = {baseline_scheduler._csprng_roll(f"client_{i}", round_num) for i in range(100)}
    assert len(rolls) == 100, (
        f"Expected 100 distinct rolls across 100 clients, got {len(rolls)}"
    )


def test_csprng_roll_range(baseline_scheduler):
    """All rolls must be in [0, 1]."""
    for r in range(200):
        val = baseline_scheduler._csprng_roll(f"c_{r}", r)
        assert 0.0 <= val <= 1.0, f"Roll out of [0,1]: {val}"


def test_csprng_roll_manual_computation(baseline_scheduler):
    """Verify _csprng_roll matches the documented SHA-256 formula exactly."""
    cid = "manual_check"
    round_num = 5
    seed_data = f"{baseline_scheduler.master_key.decode()}_{cid}_{round_num}".encode()
    expected = int(hashlib.sha256(seed_data).hexdigest()[:8], 16) / 0xFFFFFFFF

    actual = baseline_scheduler._csprng_roll(cid, round_num)
    assert actual == expected, f"Roll {actual} != manual computation {expected}"


def test_bayesian_weight_sigmoid_curve(baseline_scheduler):
    """
    bayesian_posterior_weight must implement sigma(c_lambda * (t - 0.5)).
    Test at known trust values where the sigmoid output is analytically known.
    """
    c_lambda = baseline_scheduler.c_lambda  # 8.0

    t_mid = baseline_scheduler.bayesian_posterior_weight(0.5)
    assert math.isclose(t_mid, 0.5, abs_tol=1e-9), (
        f"sigma(0) should be 0.5, got {t_mid}"
    )

    t_zero = baseline_scheduler.bayesian_posterior_weight(0.0)
    expected_zero = 1.0 / (1.0 + math.exp(-c_lambda * (0.0 - 0.5)))
    assert math.isclose(t_zero, expected_zero, rel_tol=1e-9), (
        f"At t=0.0: expected {expected_zero}, got {t_zero}"
    )
    assert t_zero < 0.02, f"Weight at t=0.0 should be near 0, got {t_zero}"

    t_one = baseline_scheduler.bayesian_posterior_weight(1.0)
    expected_one = 1.0 / (1.0 + math.exp(-c_lambda * (1.0 - 0.5)))
    assert math.isclose(t_one, expected_one, rel_tol=1e-9), (
        f"At t=1.0: expected {expected_one}, got {t_one}"
    )
    assert t_one > 0.98, f"Weight at t=1.0 should be near 1, got {t_one}"

    test_points = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for t in test_points:
        actual = baseline_scheduler.bayesian_posterior_weight(t)
        expected = 1.0 / (1.0 + math.exp(-c_lambda * (t - 0.5)))
        assert math.isclose(actual, expected, rel_tol=1e-9), (
            f"At t={t}: expected {expected}, got {actual}"
        )

    for i in range(len(test_points) - 1):
        w_lo = baseline_scheduler.bayesian_posterior_weight(test_points[i])
        w_hi = baseline_scheduler.bayesian_posterior_weight(test_points[i + 1])
        assert w_lo < w_hi, (
            f"Sigmoid not monotonically increasing: w({test_points[i]})={w_lo} >= w({test_points[i+1]})={w_hi}"
        )

    w_lo = baseline_scheduler.bayesian_posterior_weight(0.5 - 0.2)
    w_hi = baseline_scheduler.bayesian_posterior_weight(0.5 + 0.2)
    assert math.isclose(w_lo + w_hi, 1.0, abs_tol=1e-9), (
        f"Sigmoid symmetry violated: w(0.3) + w(0.7) = {w_lo + w_hi}, expected 1.0"
    )
