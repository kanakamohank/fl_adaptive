#!/usr/bin/env python3
"""Unit tests for experiments/analysis/oracle/_signals.py helpers.

Covers the computation functions used by every analysis script:
detrending, midpoint-threshold classification, per-round sweep,
Mann-Whitney U. These helpers run against pipeline_results.json data
in the field; the tests feed them minimal in-memory fixtures.
"""
import math
import os
import sys

# Allow `from experiments.analysis.oracle._signals import ...` at test run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.analysis.oracle import _signals as sig  # noqa: E402


def _fake_osh(signal_values_per_round):
    """Build the oracle_signal_history shape {round_str: {cid: {signal: v}}}
    from {round_int: {cid: v}}."""
    out = {}
    for r, row in signal_values_per_round.items():
        out[str(r)] = {cid: {"only_signal": v} for cid, v in row.items()}
    return out


def test_detrending_removes_round_mean():
    """Shifting an entire round by a constant must leave per-cid detrended
    means unchanged -- that's the whole point of detrending."""
    osh_base = _fake_osh({
        1: {"a": 1.0, "b": 2.0, "c": 3.0},
        2: {"a": 1.0, "b": 2.0, "c": 3.0},
    })
    osh_shifted = _fake_osh({
        1: {"a": 11.0, "b": 12.0, "c": 13.0},  # round-1 mean +10
        2: {"a": 1.0, "b": 2.0, "c": 3.0},
    })
    m_base = sig.per_cid_detrended_mean(osh_base, "only_signal")
    m_shifted = sig.per_cid_detrended_mean(osh_shifted, "only_signal")
    for cid in ("a", "b", "c"):
        assert abs(m_base[cid] - m_shifted[cid]) < 1e-12, (
            f"{cid}: detrending did not absorb the round shift"
        )
    print("✓ detrending absorbs per-round constant shifts")


def test_detrending_skips_absent_or_nan_values():
    """A cid that produced NaN or None on a round must not contribute to
    its round mean, and must not corrupt its own per-cid mean."""
    import json
    osh = {
        "1": {"a": {"s": 1.0}, "b": {"s": 3.0}, "c": {"s": float("nan")}},
        "2": {"a": {"s": 1.0}, "b": {"s": 3.0}, "c": {"s": None}},
    }
    m = sig.per_cid_detrended_mean(osh, "s")
    # Round means (over finite values) are (1+3)/2 = 2 both rounds, so a's
    # detrended series is [-1, -1] (mean -1) and b's is [+1, +1] (mean +1).
    # c has no finite values -> absent from result.
    assert "c" not in m
    assert abs(m["a"] - (-1.0)) < 1e-12
    assert abs(m["b"] - 1.0) < 1e-12
    print("✓ NaN/None values skipped without corrupting round means")


def test_midpoint_threshold_separates_two_separable_groups():
    """Clean at 0, noisy at 1. Classifier should produce zero errors."""
    pcm = {f"clean_{i}": 0.0 for i in range(10)}
    pcm.update({f"noisy_{i}": 1.0 for i in range(5)})
    is_noisy = lambda c: c.startswith("noisy_")
    r = sig.midpoint_threshold_errors(pcm, is_noisy)
    assert r["total_errors"] == 0
    assert r["accuracy"] == 1.0
    assert r["clean_n"] == 10 and r["noisy_n"] == 5
    print(f"✓ midpoint classifier: 0 errors on separable groups "
          f"(threshold={r['threshold']})")


def test_midpoint_threshold_handles_direction_both_ways():
    """When noisy group is LOWER (e.g. cosine signal), the predicate still
    yields the right side of the midpoint."""
    pcm = {f"clean_{i}": 1.0 for i in range(10)}
    pcm.update({f"noisy_{i}": 0.0 for i in range(5)})
    r = sig.midpoint_threshold_errors(pcm, lambda c: c.startswith("noisy_"))
    assert r["total_errors"] == 0, (
        "midpoint classifier failed when noisy is on the low side"
    )
    print("✓ midpoint classifier works whichever side 'noisy' lands on")


def test_midpoint_threshold_counts_overlap_errors():
    """One wrong clean and one wrong noisy -> 2 total errors."""
    pcm = {
        "clean_0": 0.0, "clean_1": 0.0, "clean_2": 0.9,  # one above threshold
        "noisy_0": 1.0, "noisy_1": 1.0, "noisy_2": 0.1,  # one below threshold
    }
    r = sig.midpoint_threshold_errors(pcm, lambda c: c.startswith("noisy_"))
    assert r["clean_errors"] == 1
    assert r["noisy_errors"] == 1
    assert r["total_errors"] == 2
    print(f"✓ midpoint classifier counts crossover errors "
          f"(acc={r['accuracy']:.2f} on 6 clients)")


def test_mann_whitney_u_rejects_uniform_vs_shifted():
    """Classic sanity check: samples drawn from clearly-shifted
    distributions produce a tiny p-value."""
    low = list(range(1, 21))       # 1..20
    high = list(range(31, 51))     # 31..50
    U, z, p = sig.mann_whitney_u(low, high)
    assert z < 0, "z-score should be strongly negative (low < high)"
    assert p < 1e-6, f"p-value should be tiny on fully-separated samples; got {p}"
    print(f"✓ Mann-Whitney U rejects uniform null on separated samples "
          f"(U={U:.0f}, z={z:.2f}, p={p:.2e})")


def test_mann_whitney_u_accepts_identical_samples():
    """When x and y are identical, U cannot distinguish them and p should
    be ~1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    _, _, p = sig.mann_whitney_u(x, y)
    assert p > 0.5, f"identical samples should yield large p; got {p}"
    print(f"✓ Mann-Whitney U accepts null on identical samples (p={p:.2f})")


def test_per_round_accuracy_falls_back_to_none_on_small_groups():
    """Rounds with fewer than 2 clients in either group must return None
    (classifier is ill-defined)."""
    osh = {
        "1": {"a": {"s": 1.0}},                    # 1 client, too few
        "2": {"a": {"s": 1.0}, "b": {"s": 1.0},
              "c": {"s": 2.0}, "d": {"s": 2.0}},   # 2+2, computable
    }
    is_noisy = lambda c: c in ("c", "d")
    row = sig.per_round_accuracy(osh, "s", is_noisy)
    assert row[1] is None
    assert row[2] == 1.0
    print("✓ per_round_accuracy returns None on undersized rounds")


def test_seed_number_parsing():
    assert sig.seed_number("/a/b/full_verify_seed7/pipeline_results.json") == 7
    assert sig.seed_number("/a/b/weird_name/pipeline_results.json") == -1
    print("✓ seed_number parses `_seed<N>` suffix")


def main():
    print("🧪 Oracle analysis helpers test suite")
    print("=" * 60)
    tests = [
        test_detrending_removes_round_mean,
        test_detrending_skips_absent_or_nan_values,
        test_midpoint_threshold_separates_two_separable_groups,
        test_midpoint_threshold_handles_direction_both_ways,
        test_midpoint_threshold_counts_overlap_errors,
        test_mann_whitney_u_rejects_uniform_vs_shifted,
        test_mann_whitney_u_accepts_identical_samples,
        test_per_round_accuracy_falls_back_to_none_on_small_groups,
        test_seed_number_parsing,
    ]
    for t in tests:
        t()
    print("\n🎯 All oracle analysis helper tests PASSED!")


if __name__ == "__main__":
    main()
