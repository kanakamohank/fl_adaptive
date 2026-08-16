"""
Tests for the seeded TAVS-vs-full comparison statistics.

This script had no test coverage at all, so a full green suite said nothing
about it -- which is how a wrong statistical test (unpaired applied to paired
data) and a zero-variance divergence both survived unnoticed.
"""

import statistics

import pytest


def _rows(tavs_vals, full_vals, key="late_accuracy"):
    seeds = list(range(1, len(tavs_vals) + 1))
    rows = [{"arm": "tavs", "seed": s, key: v} for s, v in zip(seeds, tavs_vals)]
    rows += [{"arm": "full_verification", "seed": s, key: v}
             for s, v in zip(seeds, full_vals)]
    return rows, seeds


def test_paired_difference_matches_hand_computation():
    from experiments.tavs_vs_full_seeded import paired_difference

    rows, seeds = _rows([0.50, 0.48, 0.52], [0.53, 0.52, 0.54])
    r = paired_difference(rows, seeds, "late_accuracy")

    expected = [0.50 - 0.53, 0.48 - 0.52, 0.52 - 0.54]
    assert r["mean"] == pytest.approx(statistics.mean(expected))
    assert r["sd"] == pytest.approx(statistics.stdev(expected))
    assert r["n_negative"] == 3 and r["n_positive"] == 0
    assert r["p"] < 0.05


def test_zero_variance_is_reported_not_declared_significant():
    """
    Identical paired differences give sd=0, where scipy returns t=+/-inf and
    p=0.0. Left unguarded that prints as overwhelming significance while the
    input actually carries no information about variability.
    """
    from experiments.tavs_vs_full_seeded import paired_difference

    rows, seeds = _rows([0.50, 0.50, 0.50], [0.53, 0.53, 0.53])
    r = paired_difference(rows, seeds, "late_accuracy")

    assert r["mean"] == pytest.approx(-0.03)
    assert r["t"] is None and r["p"] is None, "must not emit an infinite t"
    assert "degenerate" in r


def test_single_seed_returns_no_test():
    from experiments.tavs_vs_full_seeded import paired_difference

    rows, seeds = _rows([0.50], [0.53])
    r = paired_difference(rows, seeds, "late_accuracy")
    assert r["p"] is None and r["n"] == 1


def test_pairing_cancels_seed_level_variation():
    """
    The reason the paired test is the right one: a large per-seed offset shared
    by both arms must not affect the result. Under an unpaired comparison it
    would swamp the effect.
    """
    from experiments.tavs_vs_full_seeded import paired_difference

    offsets = [0.0, 0.15, -0.12]          # big seed-level swings, same for both arms
    tavs = [0.50 + o for o in offsets]
    full = [0.53 + o for o in offsets]
    rows, seeds = _rows(tavs, full)
    r = paired_difference(rows, seeds, "late_accuracy")

    assert r["mean"] == pytest.approx(-0.03)
    # Unpaired would see sd ~0.14 per arm; paired sees ~0.
    assert r["sd"] < 1e-9


def test_late_window_is_a_fraction_of_rounds_not_a_constant():
    """
    The window must scale with run length. A hardcoded 5 rounds left ~0.015 of
    noise against a ~0.033 effect at 200 rounds and produced a false negative.
    """
    import inspect

    from experiments import tavs_vs_full_seeded as m

    src = inspect.getsource(m.run_one)
    assert "late_window_frac" in src
    assert "server_accuracies[-5:]" not in src, "fixed 5-round window must be gone"

    for rounds, frac, expected in ((200, 0.25, 50), (100, 0.25, 25), (20, 0.25, 5)):
        assert max(1, int(round(rounds * frac))) == expected


def test_default_window_fraction_is_documented_and_stable():
    """The default is part of the pre-specification; changing it silently would
    undo the point of fixing it before the confirmatory seeds ran."""
    import argparse
    import inspect

    from experiments import tavs_vs_full_seeded as m

    src = inspect.getsource(m.main)
    assert '"--late-window-frac"' in src
    assert "default=0.25" in src
