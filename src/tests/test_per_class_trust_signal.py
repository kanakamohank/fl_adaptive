#!/usr/bin/env python3
"""Regression tests for trust_signal='per_class_loss_var'.

Context. Pooled per-sample loss variance scores AUROC 0.734 under a
Dirichlet(0.3) partition and inverts to 0.436 on milder noise, because under
class imbalance a clean client holding rare classes shows a wide spread that
mimics a noisy one (FedNoRo IJCAI 2023, Sec. 1). Scoring per class and
normalising each class against the cohort reaches 0.899 on the same logs.
This wires that score into the trust path.

What these lock down:
  * the scoring helper's arithmetic, against hand-computed values;
  * that the strategy actually consumes it and overrides behaviour scores;
  * that BVD keeps being computed and logged in parallel, so one run still
    yields a side-by-side comparison;
  * that the score needs no rescale window -- it is bounded in [0, 1] by
    construction, unlike pretrain_loss_var which needed a hand-tuned one.

The scoring helper is tested here WITHOUT mocking flwr (per_class_score.py is
stdlib-only). The strategy tests live in test_tavs_esp_strategy.py, which
installs the flwr mock.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_scores_match_hand_computed_values():
    from src.tavs_v2.per_class_score import per_class_cohort_scores

    # Two classes, three clients. Class 0: {1, 2, 3} -> normalised {0, .5, 1}
    #                             Class 1: {4, 4, 4} -> zero span
    # A class where every client reports the same value carries no
    # discriminating information this round, so it must contribute 0 rather
    # than an arbitrary constant that would inflate every norm equally.
    cohort = [("a", [1.0, 4.0]), ("b", [2.0, 4.0]), ("c", [3.0, 4.0])]
    s = per_class_cohort_scores(cohort)
    d = math.sqrt(2)  # scale: sqrt(num_classes)
    assert abs(s["a"] - 0.0) < 1e-9, f"a should score 0, got {s['a']}"
    assert abs(s["b"] - (0.5 / d)) < 1e-9, f"b: got {s['b']}, want {0.5/d}"
    assert abs(s["c"] - (1.0 / d)) < 1e-9, f"c: got {s['c']}, want {1.0/d}"

    # Bounded in [0, 1] regardless of class count -- this is what removes the
    # need for a rescale window.
    for C in (2, 10, 100):
        worst = per_class_cohort_scores(
            [("hi", [1.0] * C), ("lo", [0.0] * C)])
        assert abs(worst["hi"] - 1.0) < 1e-9, f"C={C}: max should be 1.0"
        assert abs(worst["lo"] - 0.0) < 1e-9, f"C={C}: min should be 0.0"

    print("✓ per-class scores match hand-computed values, bounded [0,1], "
          "zero-span classes contribute nothing")
    return True


def test_missing_classes_filled_with_cohort_minimum():
    """FedNoRo Sec 4.1 step 1: a class the client holds no samples of
    contributes no noisy samples, so it is filled with the cohort minimum
    for that class. Filling with 0, or with the client's own mean, would
    make a client that simply lacks a class look anomalous in it."""
    from src.tavs_v2.per_class_score import per_class_cohort_scores

    nan = float("nan")
    # Client "gap" is missing class 0. Cohort min for class 0 is 1.0, so it
    # should score exactly as if it had reported 1.0.
    with_nan = per_class_cohort_scores(
        [("gap", [nan, 5.0]), ("a", [1.0, 5.0]), ("b", [3.0, 5.0])])
    explicit = per_class_cohort_scores(
        [("gap", [1.0, 5.0]), ("a", [1.0, 5.0]), ("b", [3.0, 5.0])])
    assert abs(with_nan["gap"] - explicit["gap"]) < 1e-9, (
        f"NaN should be filled with the cohort minimum (1.0); "
        f"got {with_nan['gap']} vs {explicit['gap']}"
    )
    # None is treated the same as NaN (history decoded from JSON may use it).
    with_none = per_class_cohort_scores(
        [("gap", [None, 5.0]), ("a", [1.0, 5.0]), ("b", [3.0, 5.0])])
    assert abs(with_none["gap"] - explicit["gap"]) < 1e-9
    print("✓ absent classes fill with the cohort minimum (NaN and None alike)")
    return True


def test_degenerate_cohorts_do_not_raise():
    """A single client, an empty cohort, or ragged vectors must return
    something usable rather than raising inside aggregate_fit, where an
    exception would take down the whole round."""
    from src.tavs_v2.per_class_score import per_class_cohort_scores

    assert per_class_cohort_scores([]) == {}
    assert per_class_cohort_scores([("a", [])]) == {}
    # Ragged vectors are a logging bug, not something to guess around.
    assert per_class_cohort_scores([("a", [1.0, 2.0]), ("b", [1.0])]) == {}
    # A lone client has no cohort to compare against: every class has zero
    # span, so it scores 0 (maximally trusted) rather than being penalised
    # for being alone.
    solo = per_class_cohort_scores([("a", [1.0, 2.0])])
    assert abs(solo["a"]) < 1e-9, f"solo client should score 0, got {solo['a']}"
    print("✓ empty / ragged / single-client cohorts degrade safely")
    return True


def test_decode_vector_accepts_json_and_lists():
    from src.tavs_v2.per_class_score import decode_vector
    import json

    assert decode_vector(json.dumps([1.0, 2.0])) == [1.0, 2.0]
    assert decode_vector([1.0, 2.0]) == [1.0, 2.0]
    assert decode_vector(None) is None
    assert decode_vector("not json") is None
    got = decode_vector(json.dumps([1.0, None]))
    assert got is not None and math.isnan(got[1]), (
        "JSON null must decode to NaN so the fill rule applies to it"
    )
    print("✓ decode_vector handles JSON strings, raw lists, nulls and junk")
    return True


def main():
    print("🧪 per-class trust-signal regression suite")
    print("=" * 60)
    for t in (test_scores_match_hand_computed_values,
              test_missing_classes_filled_with_cohort_minimum,
              test_degenerate_cohorts_do_not_raise,
              test_decode_vector_accepts_json_and_lists):
        assert t() is True
    print("\n🎯 All per-class trust-signal tests PASSED!")


if __name__ == "__main__":
    main()
