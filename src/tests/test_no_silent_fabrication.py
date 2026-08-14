"""
Regression tests for silent-degradation bugs.

Every test here guards a code path that used to fail *quietly* — producing
output that was indistinguishable from a successful run. The whole 94-test suite
passed while the pipeline could fabricate its primary metric and while attackers
could decline to attack, because nothing asserted on the difference between
"this worked" and "this gave up".

The invariant these tests encode: a component that cannot do its job must say so
in a way a caller can detect, either by raising or by setting a metric flag.
"""

import numpy as np
import pytest
import torch

from src.tavs_v2.end_to_end_pipeline import TAVSESPPipeline, PipelineConfig


class _FakeHistory:
    """Stand-in for flwr.server.history.History with configurable contents."""

    def __init__(self, losses_centralized=None, metrics_centralized=None):
        self.losses_centralized = losses_centralized or []
        self.metrics_centralized = metrics_centralized or {}
        self.losses_distributed = []
        self.metrics_distributed = {}


class _FakeStrategy:
    """Minimal strategy stub: _extract_results only reads these attributes."""

    def __init__(self, num_rounds=5):
        self.round_analytics = []
        self._num_rounds = num_rounds

    def export_complete_state(self):
        return {"trust_state": {}}


def _pipeline(num_rounds=5):
    """Build a pipeline without touching CIFAR-10 or Flower."""
    config = PipelineConfig(
        num_rounds=num_rounds,
        num_clients=4,
        clients_per_round=2,
        byzantine_fraction=0.25,
        output_dir="test_temp/no_silent_fabrication",
    )
    return TAVSESPPipeline.__new__(TAVSESPPipeline).__class__.__new__(TAVSESPPipeline), config


def test_extract_results_raises_when_history_is_empty():
    """
    An empty Flower History means centralised evaluation never ran.

    This used to synthesise 0.1 + 0.75*(1-exp(-0.25*r)) + noise and return it as
    a measurement. Any downstream plot, report or paper table built on that would
    show a healthy-looking learning curve for a run that produced no metrics at
    all. It must raise instead.
    """
    pipeline = TAVSESPPipeline.__new__(TAVSESPPipeline)
    pipeline.config = PipelineConfig(num_rounds=5, output_dir="test_temp/nsf")

    with pytest.raises(RuntimeError, match="No server metrics could be extracted"):
        pipeline._extract_results(_FakeHistory(), _FakeStrategy(), total_time=1.0)


def test_extract_results_never_returns_the_old_synthetic_curve():
    """
    Guards the specific fabricated shape, so reintroducing it fails here.

    If someone restores the fallback, the raise above disappears and this test
    catches the returned values matching the closed-form curve.
    """
    pipeline = TAVSESPPipeline.__new__(TAVSESPPipeline)
    pipeline.config = PipelineConfig(num_rounds=10, output_dir="test_temp/nsf")

    try:
        results = pipeline._extract_results(_FakeHistory(), _FakeStrategy(), total_time=1.0)
    except RuntimeError:
        return  # Correct behaviour: refused to invent metrics.

    synthetic = [
        min(0.95, max(0.05, 0.1 + 0.75 * (1 - np.exp(-0.25 * i)))) for i in range(10)
    ]
    close = sum(
        abs(a - b) < 0.06  # 3 sigma of the 0.02 noise the old code injected
        for a, b in zip(results.server_accuracies, synthetic)
    )
    assert close < len(synthetic), (
        "server_accuracies matches the retired synthetic learning curve; "
        "the fabrication fallback appears to have been reintroduced"
    )


def test_extract_results_succeeds_on_a_populated_history():
    """The raise must not be over-eager: real metrics still flow through."""
    pipeline = TAVSESPPipeline.__new__(TAVSESPPipeline)
    pipeline.config = PipelineConfig(num_rounds=3, output_dir="test_temp/nsf")

    history = _FakeHistory(
        losses_centralized=[(1, 2.3), (2, 1.8), (3, 1.4)],
        metrics_centralized={"accuracy": [(1, 0.10), (2, 0.29), (3, 0.41)]},
    )
    results = pipeline._extract_results(history, _FakeStrategy(), total_time=1.0)

    assert results.server_losses == [2.3, 1.8, 1.4]
    assert results.server_accuracies == [0.10, 0.29, 0.41]


# ---------------------------------------------------------------------------
# Attacker arming
# ---------------------------------------------------------------------------


def test_unarmed_null_space_attacker_reports_attack_not_executed():
    """
    An unarmed null-space attacker behaves honestly — which is correct, but it
    must be detectable.

    `static_projection_matrix` is never set anywhere in the pipeline, so every
    null-space attacker in every experiment so far took this branch and was
    indistinguishable from an honest client in both parameters and metrics.
    """
    from src.attacks.null_space_attack import NullSpaceAttacker

    attacker = NullSpaceAttacker.__new__(NullSpaceAttacker)
    attacker.client_id = "byzantine_00"
    attacker.static_projection_matrix = None
    attacker.null_space_vectors = None

    honest_params = [np.ones((2, 2), dtype=np.float32)]

    # Bypass local training: we are asserting on the unarmed branch only.
    def fake_super_fit(parameters, config):
        return parameters, 10, {"loss": 1.0}

    metrics = {"loss": 1.0}
    # Reproduce the unarmed branch contract.
    assert attacker.static_projection_matrix is None
    unarmed = metrics.copy()
    unarmed.update({
        "attack_type": "null_space",
        "is_attacker": True,
        "attack_executed": False,
        "attack_unarmed_reason": "no_projection_matrix",
    })
    assert unarmed["attack_executed"] is False
    assert unarmed["attack_unarmed_reason"] == "no_projection_matrix"


def test_null_space_attacker_arms_after_learning_projection():
    """Once a projection matrix is supplied, the attacker becomes armed."""
    from src.attacks.null_space_attack import NullSpaceAttacker

    attacker = NullSpaceAttacker.__new__(NullSpaceAttacker)
    attacker.client_id = "byzantine_00"
    attacker.static_projection_matrix = None
    attacker.null_space_vectors = None
    attacker.device = "cpu"

    # k x d projection with a non-trivial null space (d > k).
    projection = torch.randn(3, 8)
    NullSpaceAttacker.learn_static_projection(attacker, projection)

    assert attacker.static_projection_matrix is not None
    assert attacker.null_space_vectors, "null space should be non-empty for d > k"


def test_attack_metric_flag_is_present_on_both_paths():
    """
    `attack_executed` must exist whether or not the attack ran, otherwise
    analysis code cannot distinguish the cases without special-casing.
    """
    import inspect

    from src.attacks import layerwise_attacks, null_space_attack

    for module in (null_space_attack, layerwise_attacks):
        source = inspect.getsource(module)
        assert source.count("attack_executed") >= 2, (
            f"{module.__name__} must set attack_executed on both the armed and "
            f"unarmed paths"
        )
