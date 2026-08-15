"""
Tests for magnitude-bounding of unverified (promoted) updates.

Mechanism 1 constrains gamma(r) = sum(p_i)/Z <= gamma_budget, which bounds the
WEIGHT unverified clients carry but not the MAGNITUDE of what they carry. Since
the aggregate is sum(w_i * g_i)/Z, damage scales with w_i * ||g_i||, and ||g_i||
was unbounded: a promoted client with p_i=0.9 and ||g_i||=1e6 keeps gamma at
0.11, well inside a 0.35 budget, while destroying the model. Measured end to end,
this drove CIFAR-10 accuracy to 0.086 (random = 0.10) with server loss > 2000.

These tests pin the containment property and the invariants it must not break.
"""

import math

import pytest
import torch

from src.tavs_v2.algo3_bvd_aggregation import UnifiedBayesianAggregator


def _cohort(n, blocks=("fc1", "fc2"), spread=0.002, seed=0):
    """A verified cohort clustered around a shared consensus point."""
    gen = torch.Generator().manual_seed(seed)
    center = {b: torch.randn(50, generator=gen) * 0.01 for b in blocks}
    return {
        f"v{i}": {b: center[b] + torch.randn(50, generator=gen) * spread for b in blocks}
        for i in range(n)
    }, center


def _median_center(verified):
    """The centre the implementation actually uses: coordinate-wise median of the
    verified cohort. This is NOT the point the cohort was generated around --
    sampling noise moves the median -- so tests must recompute it rather than
    reuse the generating centre."""
    blocks = next(iter(verified.values())).keys()
    return {
        b: torch.median(torch.stack([u[b] for u in verified.values()]), dim=0).values
        for b in blocks
    }


def _deviation(update, center):
    return math.sqrt(
        sum(((update[b] - center[b]) ** 2).sum().item() for b in center)
    )


# ---------------------------------------------------------------------------
# Core containment property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attack_scale", [1e0, 1e2, 1e4, 1e6, 1e9])
def test_aggregate_norm_is_bounded_regardless_of_attack_magnitude(attack_scale):
    """
    The property that was missing: aggregate magnitude must not scale with the
    attacker's magnitude. Without clipping this grows linearly and without limit.
    """
    verified, _ = _cohort(6)
    weights = {cid: 1.0 for cid in verified}
    attacker = {"a0": {b: torch.ones(50) * attack_scale for b in ("fc1", "fc2")}}

    clipped = UnifiedBayesianAggregator.aggregate(
        verified, weights, attacker, {"a0": 0.9}, clip_promoted=True
    )
    norm = math.sqrt(sum((v ** 2).sum().item() for v in clipped.values()))

    # The verified cohort sits at ~0.01 per coordinate; a bounded aggregate stays
    # within an order of magnitude of that regardless of the attack.
    assert norm < 1.0, f"aggregate norm {norm} grew with a {attack_scale:.0e} attack"


def test_without_clipping_the_aggregate_is_unbounded():
    """Documents the failure mode, so the fix cannot be silently reverted."""
    verified, _ = _cohort(6)
    weights = {cid: 1.0 for cid in verified}

    norms = []
    for scale in (1e2, 1e6):
        attacker = {"a0": {b: torch.ones(50) * scale for b in ("fc1", "fc2")}}
        agg = UnifiedBayesianAggregator.aggregate(
            verified, weights, attacker, {"a0": 0.9}, clip_promoted=False
        )
        norms.append(math.sqrt(sum((v ** 2).sum().item() for v in agg.values())))

    # Unclipped, a 10^4x stronger attack produces a ~10^4x larger aggregate.
    assert norms[1] / norms[0] > 1e3


def test_gamma_budget_alone_does_not_bound_damage():
    """
    The theoretical gap, stated as a test: gamma stays far inside budget while
    the aggregate explodes. This is why clipping is required and not optional.
    """
    verified, _ = _cohort(6)
    weights = {cid: 1.0 for cid in verified}
    promoted_weights = {"a0": 0.9}

    gamma = sum(promoted_weights.values()) / (
        sum(weights.values()) + sum(promoted_weights.values())
    )
    assert gamma < 0.35, "this scenario must sit inside a standard gamma_budget"

    attacker = {"a0": {b: torch.ones(50) * 1e6 for b in ("fc1", "fc2")}}
    unbounded = UnifiedBayesianAggregator.aggregate(
        verified, weights, attacker, promoted_weights, clip_promoted=False
    )
    norm = math.sqrt(sum((v ** 2).sum().item() for v in unbounded.values()))
    assert norm > 1e4, "budget-compliant attacker should still dominate when unclipped"


# ---------------------------------------------------------------------------
# Invariants the fix must not break
# ---------------------------------------------------------------------------


def test_honest_promoted_clients_are_untouched():
    """Clients inside the ball must pass through byte-identically."""
    verified, _ = _cohort(6)
    honest = {"p0": {b: t.clone() for b, t in list(verified.values())[0].items()}}

    out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(verified, honest, 1.0)

    assert stats["num_clipped"] == 0
    for b in honest["p0"]:
        assert torch.equal(out["p0"][b], honest["p0"][b])


def test_clipped_update_lands_on_the_ball_surface():
    """Projection must be exact, not approximate."""
    verified, _generating_center = _cohort(6)
    center = _median_center(verified)
    attacker = {"a0": {b: torch.ones(50) * 1e6 for b in ("fc1", "fc2")}}

    out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(verified, attacker, 1.0)

    assert stats["num_clipped"] == 1
    assert _deviation(out["a0"], center) == pytest.approx(stats["clip_radius"], rel=1e-4)


def test_clipping_preserves_direction():
    """
    Only magnitude may change. A clip that rotated the update would corrupt an
    honest-but-large contribution rather than merely bounding it.
    """
    verified, _generating_center = _cohort(6)
    center = _median_center(verified)
    attacker = {"a0": {b: torch.randn(50, generator=torch.Generator().manual_seed(7)) * 1e3
                       for b in ("fc1", "fc2")}}

    out, _ = UnifiedBayesianAggregator.clip_promoted_to_consensus(verified, attacker, 1.0)

    before = torch.cat([(attacker["a0"][b] - center[b]).flatten() for b in center])
    after = torch.cat([(out["a0"][b] - center[b]).flatten() for b in center])
    cosine = torch.nn.functional.cosine_similarity(
        before.unsqueeze(0), after.unsqueeze(0)
    ).item()
    assert cosine == pytest.approx(1.0, abs=1e-5)


def test_verified_clients_are_never_clipped():
    """
    Verified clients already passed BVD detection this round. Clipping them would
    suppress heterogeneity the detector explicitly vetted.
    """
    verified, _ = _cohort(4)
    # Give one verified client a large but vetted deviation.
    verified["v0"] = {b: t * 50 for b, t in verified["v0"].items()}
    weights = {cid: 1.0 for cid in verified}

    agg_clipped = UnifiedBayesianAggregator.aggregate(verified, weights, {}, {}, clip_promoted=True)
    agg_plain = UnifiedBayesianAggregator.aggregate(verified, weights, {}, {}, clip_promoted=False)

    for b in agg_clipped:
        assert torch.allclose(agg_clipped[b], agg_plain[b])


# ---------------------------------------------------------------------------
# Degenerate inputs must be reported, never silently assumed safe
# ---------------------------------------------------------------------------


def test_no_verified_cohort_reports_why_it_skipped():
    """
    With no vetted cohort there is no trustworthy centre or scale. Skipping is
    correct, but it must be visible rather than looking like successful clipping.
    """
    promoted = {"p0": {"fc1": torch.ones(50)}}
    out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus({}, promoted, 1.0)

    assert stats["clipping_applied"] is False
    assert stats["skipped_reason"] == "no_verified_clients_to_form_consensus"
    assert out is promoted


def test_zero_spread_cohort_does_not_collapse_promoted_clients():
    """
    At round 0 every client still holds the identical initial parameters, so the
    median deviation is 0. Clipping to a zero-radius ball would collapse every
    promoted client onto the centre and erase their contribution entirely.
    """
    identical = {b: torch.ones(50) for b in ("fc1",)}
    verified = {f"v{i}": {b: t.clone() for b, t in identical.items()} for i in range(4)}
    promoted = {"p0": {"fc1": torch.ones(50) * 5}}

    out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(verified, promoted, 1.0)

    assert stats["clipping_applied"] is False
    assert stats["skipped_reason"] == "verified_cohort_has_no_measurable_spread"
    assert torch.equal(out["p0"]["fc1"], promoted["p0"]["fc1"])


def test_empty_promoted_set_is_a_noop():
    verified, _ = _cohort(4)
    out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(verified, {}, 1.0)

    assert out == {}
    assert stats["skipped_reason"] == "no_promoted_clients"


def test_clip_factor_scales_the_radius_linearly():
    """clip_factor is the documented knob; it must behave monotonically."""
    verified, _ = _cohort(6)
    attacker = {"a0": {b: torch.ones(50) * 1e6 for b in ("fc1", "fc2")}}

    radii = []
    for factor in (0.5, 1.0, 2.0):
        _out, stats = UnifiedBayesianAggregator.clip_promoted_to_consensus(
            verified, attacker, factor
        )
        radii.append(stats["clip_radius"])

    assert radii[1] == pytest.approx(2 * radii[0], rel=1e-6)
    assert radii[2] == pytest.approx(2 * radii[1], rel=1e-6)


def test_strategy_defaults_to_clipping_enabled():
    """Containment must be the default; the ablation opts out, not in."""
    from src.tavs_v2.tavs_esp_strategy import TavsEspConfig

    config = TavsEspConfig()
    assert config.clip_promoted_updates is True
    # 2.0, not 1.0: measured selectivity of 0/57 honest clipped vs 16/48 under
    # attack, where 1.0 clipped ~97% in both cases and acted as blanket
    # normalisation rather than an outlier filter.
    assert config.promoted_clip_factor == 2.0
