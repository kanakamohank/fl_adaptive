#!/usr/bin/env python3
"""Pin the staleness analysis helpers against hand-computed fixtures.

WHY THESE TESTS EXIST
The first version of the analysis they guard reported three wrong things,
and every one was a reading error rather than a data problem:

  1. It read the cap fields off `config` instead of `config['tavs_config']`,
     got None for all of them, and printed the baseline's caps as unset.
  2. It counted a client's first-ever appearance as the staleness cap firing.
     A never-verified client is stale by definition, so that floor is exactly
     one event per distinct client -- 50 of 400 client-rounds, 12.5%. The
     script read that 12.5% as "the cap is still firing at caps 40/40" and
     declared the sweep uninterpretable. It was fully interpretable.
  3. It attributed blocked promotions to the trust bar alone, missing that
     Tier 3 also requires clean_streak >= k_trust -- which turned out to
     block as many client-rounds as the trust bar does.

Each test below targets one of those, plus the gate replay and the
counterfactual, on fixtures small enough to verify by hand.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.analysis.oracle.staleness import (  # noqa: E402
    _paired, cap_authority, distinct_clients, gamma_trace, load_runs, rates,
    replay_gates, tavs_config,
)

TC = {
    "s_max_appearances": 4, "s_max_rounds": 10,
    "theta_high": 0.7, "theta_low": 0.3, "k_trust": 3,
    "gamma_budget": 0.35, "alpha_trust": 0.7, "initial_trust": 0.25,
    "tau_ramp": 5.0,
    "pretrain_loss_var_rescale_lo": 0.6, "pretrain_loss_var_rescale_hi": 1.5,
}


def _run(rounds, trust, osh=None, detected=None, tc=None):
    """Build a minimal pipeline_results-shaped dict.

    `rounds` is {round_key: {category: [client ids]}} exactly as the pipeline
    stores round_assignments.
    """
    return {
        "config": {"tavs_config": dict(tc or TC)},
        "round_assignments": rounds,
        "trust_by_round": trust,
        "scheduling_history": [
            {"round": int(rk),
             "cohort_size": len(c.get("sampled") or []),
             "num_verified": len(c.get("verified") or []),
             "num_promoted": len(c.get("promoted") or []),
             "num_forced_stale": c.get("_forced", 0)}
            for rk, c in sorted(rounds.items(), key=lambda kv: int(kv[0]))
        ],
        "final_trust_state": {"oracle_signal_history": osh or {}},
        "byzantine_detection_history": [
            {"round": int(r), "detected": d}
            for r, d in (detected or {}).items()
        ],
    }


def test_tavs_config_is_read_from_the_nested_block():
    """Defect 1: the cap fields live under config['tavs_config'] only."""
    run = _run({"1": {"sampled": ["a"]}}, {})
    assert tavs_config(run)["s_max_appearances"] == 4
    assert tavs_config(run)["k_trust"] == 3
    # The top-level config must NOT carry them, or the old buggy read would
    # have worked and this test would not be guarding anything.
    assert "s_max_appearances" not in run["config"], (
        "if the pipeline ever flattens the config, this test and the "
        "tavs_config docstring are both stale"
    )
    print("✓ cap fields are read from the nested tavs_config block")
    return True


def test_distinct_clients_counts_the_forced_stale_floor():
    """Defect 2, part 1: the floor is one event per DISTINCT client, counted
    across every category and every round, not per client-round."""
    rounds = {
        "1": {"sampled": ["a", "b"], "verified": ["a", "b"]},
        "2": {"sampled": ["a", "c"], "verified": ["a"], "promoted": ["c"]},
    }
    assert distinct_clients(_run(rounds, {})) == 3, "a, b, c"
    print("✓ distinct_clients counts the forced-stale floor (3 clients)")
    return True


def test_forced_stale_net_subtracts_the_first_appearance_floor():
    """Defect 2, part 2: THE regression. 4 client-rounds, 2 distinct clients,
    2 forced-stale events that are both first appearances. raw = 2/4 = 50%,
    net = 0% -- the cap never fired. Reading raw as the cap rate is what
    made the real sweep look uninterpretable."""
    rounds = {
        "1": {"sampled": ["a", "b"], "verified": ["a", "b"], "_forced": 2},
        "2": {"sampled": ["a", "b"], "verified": ["a"], "promoted": ["b"],
              "_forced": 0},
    }
    r = rates(_run(rounds, {}))
    assert abs(r["forced_stale_raw"] - 0.50) < 1e-12, r
    assert abs(r["forced_stale_net"] - 0.00) < 1e-12, r
    assert abs(r["skip"] - 0.25) < 1e-12, r          # 1 promoted of 4
    assert abs(r["verified"] - 0.75) < 1e-12, r
    print("✓ forced_stale_net subtracts the floor; raw 50% -> net 0%")
    return True


def test_gate_attribution_order_and_streak_gate():
    """Defect 3: a client over the trust bar and past the ramp can still be
    blocked by clean_streak, and that must be attributed to the streak, not
    silently to 'other'.

    Fixture: client 'a' joins at round 1 and is verified cleanly in rounds
    1-3 (score 1.0, since pretrain_loss_var 0.6 sits at the window floor), so
    its streak reaches 3 after round 3. Its logged trust is 0.9 throughout,
    over theta_high, so the trust bar never blocks it.

    Every sampled-but-not-promoted round is attributed, verified rounds
    included -- the breakdown is over non-promoted client-rounds, and a
    verified client is by definition not promoted. The ramp is
    1-exp(-(r-1)/5): 0.000, 0.181, 0.330, 0.451 at rounds 1-4, all below
    theta_high=0.7, so rounds 1-4 are ALL ramp-blocked. By round 9 the ramp
    is 0.798 and the streak is 3, so nothing blocks it and it falls to
    'other'. Getting this fixture wrong the first time (expecting ramp=1) is
    exactly why the arithmetic is spelled out here.
    """
    osh = {str(r): {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}
           for r in (1, 2, 3)}
    rounds = {
        "1": {"sampled": ["a"], "verified": ["a"]},
        "2": {"sampled": ["a"], "verified": ["a"]},
        "3": {"sampled": ["a"], "verified": ["a"]},
        "4": {"sampled": ["a"]},                       # ramp still < 0.7
        "9": {"sampled": ["a"]},                       # ramp 0.798, streak 3
    }
    trust = {rk: {"a": 0.9} for rk in rounds}
    g = replay_gates(_run(rounds, trust, osh=osh))
    assert g["verifications"] == 3, g
    assert g["streak_reset"] == 0, "score 1.0 never resets the streak"
    assert g["left_at_k"] == 1, (
        "only the THIRD clean check leaves the client at streak >= 3; "
        f"counting resets alone would miss that: {g}"
    )
    assert g["sampled"] == 5, g
    assert g["ramp"] == 4, f"rounds 1-4 are all ramp-blocked: {g}"
    assert g["other"] == 1, f"round 9 clears every gate: {g}"
    assert g["trust_bar"] == 0, "logged trust is 0.9, over the bar"
    assert g["streak"] == 0, (
        "the streak gate is only reached once the ramp is cleared, which "
        "happens first at round 9, by which point the streak is already 3"
    )
    print("✓ gate attribution separates trust bar / ramp / streak / other")
    return True


def test_streak_blocks_a_client_that_clears_trust_and_ramp():
    """The finding this guards: clean_streak is a SECOND eligibility gate,
    and it blocks client-rounds the trust bar does not.

    Client 'b' joins at round 1, is verified ONCE with a clean score (streak
    = 1), then is sampled again at round 20. Its logged trust is 0.9, over
    theta_high, and the ramp is 1-exp(-19/5) = 0.978, well past it. The only
    thing left is the streak, at 1 against k_trust=3.
    """
    osh = {"1": {"b": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {"1": {"sampled": ["b"], "verified": ["b"]},
              "20": {"sampled": ["b"]}}
    g = replay_gates(_run(rounds, {"1": {"b": 0.9}, "20": {"b": 0.9}},
                          osh=osh))
    assert g["streak"] == 1, f"round 20 must be streak-blocked: {g}"
    assert g["trust_bar"] == 0 and g["other"] == 0, g
    assert g["ramp"] == 1, "round 1 itself is ramp-blocked (ramp = 0 there)"
    print("✓ clean_streak blocks a client that clears both trust and ramp")
    return True


def test_streak_resets_on_low_score_and_on_detection():
    """The two reset paths must be counted separately, because which one
    dominates is the finding: a score at or below 0.8 resets the streak even
    when the detector is happy."""
    # raw 1.4 -> score 1-(1.4-0.6)/0.9 = 0.111 <= 0.8, so a score reset.
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 1.4},
                 "b": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {"1": {"sampled": ["a", "b"], "verified": ["a", "b"]}}
    g = replay_gates(_run(rounds, {"1": {"a": 0.9, "b": 0.9}}, osh=osh,
                          detected={"1": ["b"]}))
    assert g["verifications"] == 2, g
    assert g["reset_by_score"] == 1, f"a resets on score: {g}"
    assert g["reset_by_detector"] == 1, f"b resets on detection: {g}"
    assert g["streak_reset"] == 2, g
    print("✓ score resets and detector resets are counted separately")
    return True


def test_replay_gates_returns_none_without_round_keyed_trust():
    """The baseline predates the trust_by_round fix. The replay must say so
    rather than silently reporting zeros, which would have put a row of 0.0
    in the table and read as 'nothing blocks promotion'."""
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    assert replay_gates(_run({"1": {"sampled": ["a"]}}, {}, osh=osh)) is None
    print("✓ replay_gates returns None when trust_by_round is empty")
    return True


def test_replay_needs_oracle_signals_or_it_invents_the_streak():
    """REVIEW FINDING 1, the worst of them. Without oracle_signal_history
    there is no behaviour score, so no streak can be replayed. Left
    unguarded, every streak stayed at 0 and every client-round that cleared
    the trust bar and ramp was blamed on the streak gate -- and the
    "reset on X% of verifications" line that would have shown 0 verifications
    is skipped by its own guard, so nothing warned.

    TEST 3's headline number IS the streak count, so this is the difference
    between a measurement and a number invented from a missing file. Both
    replays must refuse.
    """
    rounds = {str(r): {"sampled": ["a"], "verified": ["a"]} for r in (1, 2, 3)}
    rounds["20"] = {"sampled": ["a"]}
    trust = {rk: {"a": 0.9} for rk in rounds}
    osh = {str(r): {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}
           for r in (1, 2, 3)}

    # With signals: streak reaches 3, so round 20 clears every gate.
    g = replay_gates(_run(rounds, trust, osh=osh))
    assert g["other"] == 1 and g["streak"] == 0, g

    # Without them: must refuse, not report {ramp:3, streak:1}.
    assert replay_gates(_run(rounds, trust, osh={})) is None, (
        "replay_gates must refuse a run with no oracle signals"
    )
    assert cap_authority(_run(rounds, trust, osh={})) is None, (
        "cap_authority replays the same streak and must refuse too -- with "
        "every streak stuck at 0 it reports the cap as 100% redundant"
    )
    print("✓ both replays refuse a run with no oracle signals")
    return True


def test_streak_advances_through_a_round_missing_from_trust_by_round():
    """REVIEW FINDING 2. trust_by_round holds only rounds that produced
    round_analytics -- test_trust_by_round_logging pins that a round can be
    absent. The attribution needs trust and must skip such a round, but the
    STREAK replay must not: skipping it drops that round's verifications and
    leaves every later streak one short.

    Fixture: 'a' verified cleanly in rounds 1-3, sampled at round 20. With
    all three trust entries the streak is 3 and round 20 falls to 'other'.
    Drop ONLY round 2's trust entry: round 2's verification must still count,
    so round 20 must still be 'other'. Before the fix it flipped to
    'streak'.
    """
    osh = {str(r): {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}
           for r in (1, 2, 3)}
    rounds = {str(r): {"sampled": ["a"], "verified": ["a"]} for r in (1, 2, 3)}
    rounds["20"] = {"sampled": ["a"]}
    full = {rk: {"a": 0.9} for rk in rounds}
    partial = {rk: v for rk, v in full.items() if rk != "2"}

    g_all = replay_gates(_run(rounds, full, osh=osh))
    g_gap = replay_gates(_run(rounds, partial, osh=osh))
    assert g_all["other"] == 1 and g_all["streak"] == 0, g_all
    assert g_gap["verifications"] == 3, (
        f"round 2's verification must still advance the streak: {g_gap}"
    )
    assert g_gap["other"] == 1 and g_gap["streak"] == 0, (
        f"round 20 must stay 'other'; the streak desynced: {g_gap}"
    )
    assert g_gap["skipped_no_trust"] == 1, (
        f"the untrusted round must be recorded, not silently dropped: {g_gap}"
    )
    assert g_gap["sampled"] == g_all["sampled"] - 1, (
        "the skipped round is excluded from attribution (no trust to "
        f"attribute with) but only that one: {g_gap}"
    )
    print("✓ a round missing from trust_by_round still advances the streak")
    return True


def test_rates_returns_none_for_a_run_with_no_cohort():
    """REVIEW FINDING 4. rates() returned {}, and every caller indexes it, so
    one truncated run turned into `KeyError: 'skip'` and aborted the whole
    comparison. load_seeds deliberately skips such a run instead."""
    assert rates(_run({}, {})) is None
    # And the paired helper must drop that seed rather than raise.
    good = {"1": _run({"1": {"sampled": ["a"], "verified": ["a"]}}, {}),
            "2": _run({}, {})}                      # seed 2 is truncated
    diffs = _paired(good, good, "skip")
    assert len(diffs) == 1, f"seed 2 must be dropped, not indexed: {diffs}"
    assert diffs == [0.0], diffs
    print("✓ rates() returns None on an empty run and _paired drops it")
    return True


def test_cap_authority_is_safe_when_no_clock_ever_fires():
    """REVIEW FINDING 3. A short run where no clock fires returned
    forced == 0, and TEST 5's `blocked / forced` raised ZeroDivisionError
    after every other section had already printed."""
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    c = cap_authority(_run({"1": {"sampled": ["a"], "verified": ["a"]}},
                           {"1": {"a": 0.9}}, osh=osh))
    assert c == {"forced": 0, "blocked": 0, "sampled": 1}, c
    print("✓ cap_authority reports forced=0 instead of crashing the caller")
    return True


def test_gamma_trace_matches_the_schedulers_own_formula():
    """Falsifier F2 is conditional on gamma not binding, so the replay has to
    reproduce the scheduler's definition: gamma = sum_p / (|V| + sum_p) with
    p_i = sigmoid(c_lambda * (t_eff - 0.5)).

    Hand-computed fixture at round 1: one promoted client at trust 0.9 and
    two verified. The ramp at r == join is 1 - exp(0) = 0, so t_eff = 0 and
    p = sigmoid(8 * -0.5) = 1/(1+e^4) = 0.017986. gamma = p / (2 + p).
    Round 20 repeats it past the ramp: t_eff = 0.9, p = sigmoid(3.2) =
    0.960834, gamma = p / (2 + p).
    """
    import math as _m
    rounds = {"1": {"promoted": ["a"], "verified": ["b", "c"],
                    "sampled": ["a", "b", "c"]},
              "20": {"promoted": ["a"], "verified": ["b", "c"],
                     "sampled": ["a", "b", "c"]}}
    trust = {rk: {"a": 0.9, "b": 0.9, "c": 0.9} for rk in rounds}
    g = gamma_trace(_run(rounds, trust))
    p1 = 1.0 / (1.0 + _m.exp(4.0))
    p2 = 1.0 / (1.0 + _m.exp(-3.2))
    assert len(g) == 2, g
    assert abs(g[0] - p1 / (2 + p1)) < 1e-12, (g[0], p1 / (2 + p1))
    assert abs(g[1] - p2 / (2 + p2)) < 1e-12, (g[1], p2 / (2 + p2))
    # The ramp must matter: an unramped client contributes almost no weight.
    assert g[0] < 0.01 < 0.3 < g[1], (
        "if these collapse, get_effective_trust is not being applied"
    )
    assert gamma_trace(_run(rounds, {})) is None
    print("✓ gamma_trace reproduces the scheduler's budget formula")
    return True


def test_load_runs_matches_load_seeds_on_suffixed_dirs():
    """REVIEW FINDING 5. load_seeds matches seed(\\d+) and tolerates a
    suffix; load_runs split on "seed" and called int(), so a rerun left as
    tavs_skip_seed1_rerun killed it with ValueError while the accuracy table
    silently absorbed the stray run. One regex, one file set."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / "cfg" / "skip00" / "tavs_skip_seed1_rerun"
        d.mkdir(parents=True)
        (d / "pipeline_results.json").write_text("{}")
        runs = load_runs(root)
        assert list(runs) == ["1"], (
            f"must parse seed 1 from a suffixed directory, as load_seeds "
            f"does, rather than raising ValueError: {runs}"
        )
    print("✓ load_runs parses suffixed seed dirs the way load_seeds does")
    return True


def test_cap_authority_splits_redundant_from_sole_authority():
    """The counterfactual: a client-round the 4/10 cap would force, where the
    trust-or-streak gate blocks it anyway, is redundant -- not authority.

    Client 'a' is sampled every round 1..12 and never verified after round 1,
    so by round 11 it is 10 rounds past its last verification and the
    s_max_rounds=10 clock fires. Its trust is 0.9 (over the bar) and the ramp
    is past 0.7 by then, but its streak is 1, below k_trust=3 -- so the gate
    blocks it anyway and the cap is redundant.
    """
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {str(r): {"sampled": ["a"], **({"verified": ["a"]} if r == 1
                                            else {"promoted": ["a"]})}
              for r in range(1, 13)}
    c = cap_authority(_run(rounds, {str(r): {"a": 0.9} for r in range(1, 13)},
                           osh=osh))
    assert c["sampled"] == 12, c
    assert c["forced"] > 0, "the 4/10 clocks must fire somewhere"
    assert c["blocked"] == c["forced"], (
        f"streak is 1 < k_trust=3 throughout, so every forced round is "
        f"redundant: {c}"
    )
    print(f"✓ cap_authority marks all {c['forced']} forced rounds redundant")
    return True


def test_load_runs_rejects_a_duplicate_seed(tmp=None):
    """Two config subdirectories under one family is how a rerun silently
    swaps half a series. load_runs must raise, matching load_seeds."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cfg in ("cfgA", "cfgB"):
            d = root / cfg / "skip00" / "tavs_skip_seed1"
            d.mkdir(parents=True)
            (d / "pipeline_results.json").write_text("{}")
        try:
            load_runs(root)
        except RuntimeError as e:
            assert "duplicate seed 1" in str(e), e
            print("✓ load_runs raises on a duplicate seed across configs")
            return True
    raise AssertionError("load_runs accepted two configs for one seed")


def main():
    ok = [
        test_tavs_config_is_read_from_the_nested_block(),
        test_distinct_clients_counts_the_forced_stale_floor(),
        test_forced_stale_net_subtracts_the_first_appearance_floor(),
        test_gate_attribution_order_and_streak_gate(),
        test_streak_blocks_a_client_that_clears_trust_and_ramp(),
        test_streak_resets_on_low_score_and_on_detection(),
        test_replay_gates_returns_none_without_round_keyed_trust(),
        test_replay_needs_oracle_signals_or_it_invents_the_streak(),
        test_streak_advances_through_a_round_missing_from_trust_by_round(),
        test_rates_returns_none_for_a_run_with_no_cohort(),
        test_cap_authority_is_safe_when_no_clock_ever_fires(),
        test_gamma_trace_matches_the_schedulers_own_formula(),
        test_load_runs_matches_load_seeds_on_suffixed_dirs(),
        test_cap_authority_splits_redundant_from_sole_authority(),
        test_load_runs_rejects_a_duplicate_seed(),
    ]
    if all(ok):
        print("\n🎯 All staleness-analysis tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
