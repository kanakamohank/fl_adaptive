import pytest
import math
import torch
from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler
from src.tavs_v2.algo2_esp_projection import EphemeralStructuredProjection
from src.tavs_v2.algo3_bvd_aggregation import BlockVarianceDetector, UnifiedBayesianAggregator


@pytest.fixture
def model_blocks():
    return {"layer_1": 500, "layer_2": 300}


@pytest.fixture
def scheduler():
    return TavsScheduler(
        gamma_budget=0.35,
        theta_low=0.3,
        theta_high=0.8,
        alpha_trust=0.9,
        tau_ramp=30.0,
        k_trust=10,
        p_decoy=0.15,
        c_lambda=8.0,
        master_key=b'integration_test_key'
    )


@pytest.fixture
def projector(model_blocks):
    return EphemeralStructuredProjection(
        target_k=50,
        model_blocks=model_blocks,
        master_key=b'integration_test_key'
    )


@pytest.fixture
def detector():
    return BlockVarianceDetector(tau_z=10.0, alpha_sigma=0.9, epsilon_stab=1e-5)


def _make_gradient(model_blocks, scale=1.0, bias=0.0):
    return {name: torch.randn(dim) * scale + bias for name, dim in model_blocks.items()}


def test_scheduler_projection_bvd_pipeline(scheduler, projector, detector, model_blocks):
    """
    Full pipeline: schedule → project → detect → update_trust.
    Verifies that all three V2 algorithms compose correctly over multiple
    rounds with honest and byzantine clients.
    """
    honest_ids = [f"honest_{i}" for i in range(6)]
    byz_ids = [f"byz_{i}" for i in range(2)]
    all_clients = honest_ids + byz_ids
    num_rounds = 15

    for r in range(1, num_rounds + 1):
        V, P, D = scheduler.schedule_verifications(all_clients, round_num=r)

        assert V | P | D == set(all_clients), (
            f"Round {r}: V ∪ P ∪ D ≠ available_clients"
        )
        assert len(V) + len(P) + len(D) == len(all_clients), (
            f"Round {r}: client count mismatch (possible duplicates)"
        )

        raw_updates = {}
        for cid in all_clients:
            if cid.startswith("byz"):
                raw_updates[cid] = _make_gradient(model_blocks, scale=1.0, bias=30.0)
            else:
                raw_updates[cid] = _make_gradient(model_blocks, scale=1.0, bias=0.0)

        projected = {}
        for cid in all_clients:
            projected[cid] = projector.project_client_update(raw_updates[cid], round_num=r)

        for cid in all_clients:
            for block_name in model_blocks:
                k_m = projector.k_m_allocations[block_name]
                assert projected[cid][block_name].shape == (k_m,), (
                    f"Projection shape mismatch for {cid}/{block_name} in round {r}"
                )

        if V:
            verified_projected = {cid: projected[cid] for cid in V}
            inliers, outliers, behavior_scores = detector.detect_outliers(
                verified_projected, V
            )

            assert inliers | outliers == V, (
                f"Round {r}: BVD inliers ∪ outliers ≠ verified set"
            )
            assert inliers.isdisjoint(outliers), (
                f"Round {r}: BVD classified a client as both inlier and outlier"
            )

            for cid in V:
                score = behavior_scores.get(cid, 0.95)
                scheduler.update_trust(cid, behavior_score=score, was_verified=True)
        for cid in P:
            scheduler.update_trust(cid, behavior_score=0.0, was_verified=False)

    for cid in honest_ids:
        t = scheduler.trust_scores[cid]
        assert t > 0.0, f"Honest {cid} trust collapsed to {t}"
    for cid in byz_ids:
        t = scheduler.trust_scores[cid]
        assert t < scheduler.trust_scores[honest_ids[0]], (
            f"Byzantine {cid} trust ({t}) not below honest trust"
        )


def test_full_round_conservation(scheduler):
    """
    For every call to schedule_verifications, V ∪ P ∪ D must equal the
    input client set with no duplicates and no missing clients.
    """
    clients = [f"client_{i}" for i in range(20)]

    for r in range(1, 50):
        V, P, D = scheduler.schedule_verifications(clients, round_num=r)

        union = V | P | D
        assert union == set(clients), (
            f"Round {r}: partition leak — missing: {set(clients) - union}, extra: {union - set(clients)}"
        )

        total_assigned = len(V) + len(P) + len(D)
        assert total_assigned == len(clients), (
            f"Round {r}: {total_assigned} assignments for {len(clients)} clients (overlap detected)"
        )

        assert V.isdisjoint(P) and V.isdisjoint(D) and P.isdisjoint(D), (
            f"Round {r}: sets V, P, D are not pairwise disjoint"
        )

        for cid in clients:
            score = 0.85 if int(cid.split("_")[1]) < 15 else 0.2
            was_verified = cid in V
            scheduler.update_trust(cid, behavior_score=score, was_verified=was_verified)


def test_projection_determinism_within_round(projector, model_blocks):
    """
    The same client update projected in the same round must yield identical
    results — projection matrices are cached per round.
    """
    update = _make_gradient(model_blocks)
    proj1 = projector.project_client_update(update, round_num=5)
    proj2 = projector.project_client_update(update, round_num=5)

    for block in model_blocks:
        assert torch.equal(proj1[block], proj2[block]), (
            f"Same update, same round produced different projections for {block}"
        )


def test_projection_changes_across_rounds(projector, model_blocks):
    """
    The ephemeral property: projection matrices change each round, so the
    same update projected in different rounds must differ.
    """
    update = _make_gradient(model_blocks)
    proj_r1 = projector.project_client_update(update, round_num=1)
    proj_r2 = projector.project_client_update(update, round_num=2)

    any_differ = False
    for block in model_blocks:
        if not torch.equal(proj_r1[block], proj_r2[block]):
            any_differ = True
            break
    assert any_differ, "Projection output identical across rounds — ephemeral seeds broken"


def test_aggregator_with_scheduler_weights(scheduler, model_blocks):
    """
    Validates that bayesian_posterior_weight produces valid weights that the
    UnifiedBayesianAggregator can consume without numerical issues.
    """
    clients = [f"client_{i}" for i in range(6)]
    scheduler.schedule_verifications(clients, round_num=0)

    for i, cid in enumerate(clients):
        scheduler.trust_scores[cid] = 0.3 + i * 0.1
        scheduler.join_rounds[cid] = -100

    V, P, D = scheduler.schedule_verifications(clients, round_num=50)

    verified_updates = {}
    verified_weights = {}
    promoted_updates = {}
    promoted_weights = {}

    for cid in V:
        verified_updates[cid] = _make_gradient(model_blocks)
        verified_weights[cid] = 1.0

    for cid in P:
        promoted_updates[cid] = _make_gradient(model_blocks)
        t_eff = scheduler.get_effective_trust(cid, 50)
        w = scheduler.bayesian_posterior_weight(t_eff)
        assert 0.0 < w < 1.0, f"Bayesian weight {w} out of (0,1) for trust {t_eff}"
        promoted_weights[cid] = w

    if not verified_updates and not promoted_updates:
        pytest.skip("No clients in any partition — degenerate scheduling")

    agg = UnifiedBayesianAggregator.aggregate(
        verified_updates, verified_weights, promoted_updates, promoted_weights
    )

    for block in model_blocks:
        assert block in agg, f"Aggregated result missing block {block}"
        assert not torch.isnan(agg[block]).any(), f"NaN in aggregated {block}"
        assert not torch.isinf(agg[block]).any(), f"Inf in aggregated {block}"


def test_budget_constraint_holds_every_round(scheduler):
    """
    Mechanism 1 invariant: after schedule_verifications, the unverified
    influence ratio sum(p_i)/Z_r must not exceed gamma_budget.
    """
    clients = [f"c_{i}" for i in range(15)]

    for r in range(1, 40):
        V, P, D = scheduler.schedule_verifications(clients, round_num=r)

        sum_p_i = sum(
            scheduler.bayesian_posterior_weight(scheduler.get_effective_trust(c, r))
            for c in P
        )
        Z_r = len(V) + sum_p_i
        if Z_r > 0:
            gamma = sum_p_i / Z_r
            assert gamma <= scheduler.gamma_budget + 1e-9, (
                f"Round {r}: budget violated — gamma={gamma:.4f} > {scheduler.gamma_budget}"
            )

        for cid in clients:
            score = 0.9 if int(cid.split("_")[1]) < 10 else 0.15
            scheduler.update_trust(cid, behavior_score=score, was_verified=(cid in V))
