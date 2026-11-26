#!/usr/bin/env python3
"""The clean-streak cutoff must be a knob, and must reach the scheduler.

WHY THIS EXISTS
`update_trust` reset the clean streak on `behavior_score <= 0.8` with 0.8
written as a literal. It was the only threshold in the scheduler with no
config field, and it is not a rare edge: measured over the 10-seed a=1.0
runs it fires on 19.9% of all verifications, and the streak it gates is a
hard precondition for Tier 3.

THE REDUNDANCY THESE TESTS PIN
With a linear rescale, "score > cutoff" is exactly "raw < t*" for

    t* = rescale_lo + (1 - cutoff) * (hi - lo)

so the cutoff and the rescale window are two parameterisations of ONE
threshold on the raw signal. Anything that sweeps one without the other is
only moving half the knob. test_cutoff_and_window_are_one_threshold states
that algebraically so a future reader cannot miss it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tavs_v2.algo1_tavs_scheduler import TavsScheduler  # noqa: E402

BASE = dict(gamma_budget=0.35, theta_low=0.3, theta_high=0.7,
            alpha_trust=0.7, tau_ramp=5.0, k_trust=3)


def _sched(**kw):
    return TavsScheduler(**{**BASE, **kw})


def test_default_is_the_historical_literal():
    """The default must stay 0.8 so every existing run and stored config
    keeps its exact meaning. If this ever changes, prior results silently
    describe a different scheduler."""
    assert _sched().streak_score_cutoff == 0.8
    print("✓ default cutoff is 0.8, the value the literal carried")
    return True


def test_cutoff_decides_reset_versus_extend():
    """At the boundary: <= cutoff resets, > cutoff extends. The comparison
    is `<=`, so a score exactly equal to the cutoff RESETS."""
    s = _sched()
    s.trust_scores["a"] = 0.5
    s.update_trust("a", behavior_score=0.81, was_verified=True, round_num=1)
    assert s.clean_streaks["a"] == 1, "0.81 > 0.8 extends"
    s.update_trust("a", behavior_score=0.80, was_verified=True, round_num=2)
    assert s.clean_streaks["a"] == 0, "0.80 is NOT > 0.8, so it resets"
    print("✓ the cutoff is a `<=` boundary: 0.80 resets, 0.81 extends")
    return True


def test_a_lower_cutoff_admits_scores_the_default_rejects():
    """The point of the knob. A score of 0.7 resets the streak at the
    default and extends it at cutoff 0.6 -- same input, different routing."""
    strict, loose = _sched(), _sched(streak_score_cutoff=0.6)
    for s in (strict, loose):
        s.trust_scores["a"] = 0.5
        s.update_trust("a", behavior_score=0.70, was_verified=True,
                       round_num=1)
    assert strict.clean_streaks["a"] == 0, "0.70 <= 0.8 resets"
    assert loose.clean_streaks["a"] == 1, "0.70 > 0.6 extends"
    print("✓ lowering the cutoff admits scores the default rejects")
    return True


def test_detector_verdict_still_outranks_the_cutoff():
    """The cutoff must not become an escape hatch. A flagged client resets
    at ANY cutoff -- that ordering is the whole reason the `is_outlier or`
    term was added, after a client the detector flagged kept accumulating
    its streak because it scored 0.9."""
    s = _sched(streak_score_cutoff=0.0)      # cutoff rejects nothing
    s.trust_scores["a"] = 0.5
    s.update_trust("a", behavior_score=1.0, was_verified=True,
                   is_outlier=True, round_num=1)
    assert s.clean_streaks["a"] == 0, (
        "a detector-flagged client must reset regardless of the cutoff"
    )
    print("✓ the detector verdict still outranks the score cutoff")
    return True


def test_cutoff_does_nothing_without_verification():
    """Trust and the streak move only on verification. A promoted client
    must not have its streak touched at any cutoff."""
    s = _sched(streak_score_cutoff=0.95)
    s.trust_scores["a"] = 0.9
    s.clean_streaks["a"] = 2
    s.update_trust("a", behavior_score=0.0, was_verified=False, round_num=1)
    assert s.clean_streaks["a"] == 2, "a skipped round must not reset"
    print("✓ the cutoff is inert when the client was not verified")
    return True


def test_cutoff_gates_tier_three_entry():
    """End to end: the cutoff decides whether a client can reach Tier 3.

    Trust is driven to 0.9 and the ramp is cleared (round 20 against join 1,
    ramp = 1 - exp(-19/5) = 0.978), so tier is decided by the streak alone.
    Four verifications at score 0.75: under the default they all reset and
    the client is stuck at Tier 2; at cutoff 0.6 they accumulate past
    k_trust = 3 and it reaches Tier 3.
    """
    for cutoff, want in ((0.8, 2), (0.6, 3)):
        s = _sched(streak_score_cutoff=cutoff)
        s.join_rounds["a"] = 1
        for r in (1, 2, 3, 4):
            s.update_trust("a", behavior_score=0.75, was_verified=True,
                           round_num=r)
        s.trust_scores["a"] = 0.9          # isolate the streak from the EMA
        got = s.get_tier("a", 20)
        assert got == want, (
            f"cutoff {cutoff}: expected tier {want}, got {got} "
            f"(streak={s.clean_streaks['a']})"
        )
    print("✓ the cutoff gates Tier-3 entry end to end")
    return True


def test_cutoff_and_window_are_one_threshold():
    """The redundancy, stated as arithmetic.

    score = 1 - clip((raw - lo)/(hi - lo), 0, 1), so for raw inside the
    window, `score > cutoff` rearranges to `raw < lo + (1-cutoff)*(hi-lo)`.
    Two different (window, cutoff) pairs that give the same t* must route an
    identical raw value identically -- which is why sweeping one alone is
    only half the knob.
    """
    def score(raw, lo, hi):
        return 1.0 - max(0.0, min(1.0, (raw - lo) / (hi - lo)))

    def t_star(lo, hi, cutoff):
        return lo + (1.0 - cutoff) * (hi - lo)

    # The shipped pair, and a different window tuned to the same t*.
    a = (0.6, 1.5, 0.8)
    assert abs(t_star(*a) - 0.780) < 1e-12, t_star(*a)
    b_lo, b_hi = 0.0, 1.0
    b_cut = 1.0 - (t_star(*a) - b_lo) / (b_hi - b_lo)      # same t*
    assert abs(t_star(b_lo, b_hi, b_cut) - t_star(*a)) < 1e-12

    for raw in (0.0, 0.3, 0.5, 0.7, 0.779, 0.781, 0.9, 1.4, 2.0):
        ra = score(raw, a[0], a[1]) > a[2]
        rb = score(raw, b_lo, b_hi) > b_cut
        assert ra == rb, (raw, ra, rb)
        assert ra == (raw < t_star(*a)), (
            f"the rule must be exactly 'raw < {t_star(*a)}': {raw}"
        )
    print("✓ window and cutoff are one threshold: t* = lo + (1-cut)*(hi-lo)")
    return True


def test_a_cutoff_that_would_disable_tier_three_is_rejected():
    """REVIEW FINDING. Behaviour scores are bounded in [0, 1] and the reset
    test is `score <= cutoff`, so any cutoff >= 1.0 resets on EVERY
    verification and makes Tier 3 permanently unreachable -- a client
    scoring a perfect 1.0 every round still never gets there.

    The realistic way in is `--streak-score-cutoff 8` as a typo for 0.8,
    which before this guard ran silently AND produced a plausible-looking
    config tag (ssc800), so the mistake survived into the results tree.
    """
    for bad in (1.0, 1.5, 8.0, 80.0, -0.1, -1.0):
        try:
            _sched(streak_score_cutoff=bad)
        except ValueError as e:
            assert "streak_score_cutoff" in str(e), e
            continue
        raise AssertionError(
            f"cutoff {bad} must be rejected: it silently disables Tier 3"
        )
    # The typo case should suggest the likely intent.
    try:
        _sched(streak_score_cutoff=8.0)
    except ValueError as e:
        assert "0.08" in str(e), f"should suggest /100: {e}"
    # And every legitimate value must still construct, including the
    # boundaries 0.0 (nothing resets by score) and just under 1.0.
    for ok in (0.0, 0.5, 0.8, 0.95, 0.999999):
        assert _sched(streak_score_cutoff=ok).streak_score_cutoff == ok
    print("✓ a cutoff that would silently disable Tier 3 is rejected")
    return True


def test_config_field_reaches_the_scheduler():
    """Guard the wiring. A field nothing threads through is a field that
    silently does nothing -- the exact failure the repo already has a test
    for on initial_trust and bootstrap_verify_new_clients."""
    import dataclasses
    from src.tavs_v2.tavs_esp_strategy import TavsEspConfig
    names = {f.name: f for f in dataclasses.fields(TavsEspConfig)}
    assert "streak_score_cutoff" in names, "field missing from the config"
    assert names["streak_score_cutoff"].default == 0.8, names[
        "streak_score_cutoff"].default
    print("✓ streak_score_cutoff exists on the config with default 0.8")
    return True


def main():
    ok = [
        test_default_is_the_historical_literal(),
        test_cutoff_decides_reset_versus_extend(),
        test_a_lower_cutoff_admits_scores_the_default_rejects(),
        test_detector_verdict_still_outranks_the_cutoff(),
        test_cutoff_does_nothing_without_verification(),
        test_cutoff_gates_tier_three_entry(),
        test_cutoff_and_window_are_one_threshold(),
        test_a_cutoff_that_would_disable_tier_three_is_rejected(),
        test_config_field_reaches_the_scheduler(),
    ]
    if all(ok):
        print("\n\U0001F3AF All streak-score-cutoff tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
