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
    _paired, distinct_clients, load_runs, rates, replay_schedule, tavs_config,
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


def test_tier2_is_promoted_not_blocked():
    """THE CORRECTION THIS FILE EXISTS FOR.

    schedule_verifications' Phase 1 ends with `else: P.add(cid)` -- Tier 2.
    So a client with theta_low <= t_eff < theta_high, or with t_eff over
    theta_high but a short streak, is SKIPPED, not blocked.

    An earlier attribution used the Tier-3 condition
    (t_eff >= theta_high AND streak >= k_trust) as the condition for being
    skipped and reported everything failing it as "blocked", producing two
    headline gates -- "trust bar blocks 140 client-rounds, streak blocks 81"
    -- that were really counting Tier-2 clients the scheduler was skipping
    happily. This test pins the branch that was missed.

    Round 1 verifies a, b, c (never verified, so stale -> V, streak -> 1).
    Round 10 has the ramp at 1 - exp(-9/5) = 0.835, so it no longer binds,
    and trust is a=0.2 (Tier 1), b=0.60 and c=0.65 (both Tier 2: over
    theta_low, under theta_high).
    """
    osh = {"1": {c: {"is_verified": True, "pretrain_loss_var": 0.6}
                 for c in "abc"}}
    rounds = {"1": {"sampled": list("abc"), "verified": list("abc")},
              "10": {"sampled": list("abc"), "promoted": ["c"]}}
    trust = {"1": {c: 0.5 for c in "abc"},
             "10": {"a": 0.2, "b": 0.60, "c": 0.65}}
    g = replay_schedule(_run(rounds, trust, osh=osh))
    assert g["stale"] == 3, f"round 1: all three never verified: {g}"
    assert g["tier1"] == 1, f"only 'a' is under theta_low: {g}"
    assert g["tier2_tentative"] == 2, (
        f"b and c are Tier 2 and MUST be tentatively promoted, not "
        f"blocked -- this is the branch the old attribution missed: {g}"
    )
    assert g["tier3_promoted"] == 0, "streak is 1, under k_trust=3"
    print("✓ Tier 2 clients are promoted, not counted as blocked")
    return True


def test_budget_demotes_lowest_trust_first_and_gamma_pre_exceeds_cap():
    """Phase 2, hand-computed, and the reason gamma must be read BEFORE it.

    Continuing the fixture above at round 10: V = {a} (Tier 1), tentative
    P = {b, c}. With c_lambda = 8,
        p(0.60) = sigmoid(0.8) = 0.690000
        p(0.65) = sigmoid(1.2) = 0.768525
    so sum_p = 1.458525 and gamma_pre = 1.458525 / (1 + 1.458525) = 0.59326
    -- far above the 0.35 budget. The loop demotes the LOWEST-trust client,
    b, leaving sum_p = 0.768525 and |V| = 2, so
    gamma_post = 0.768525 / 2.768525 = 0.27759 <= 0.35 and it stops.

    gamma_post sits comfortably under the cap while the budget was biting
    hard. Reading only gamma_post is what made an earlier version of the
    analysis report "the budget does not bind": it was reading the
    constraint's own output, which the loop guarantees is under the cap.
    """
    import math as _m
    osh = {"1": {c: {"is_verified": True, "pretrain_loss_var": 0.6}
                 for c in "abc"}}
    rounds = {"1": {"sampled": list("abc"), "verified": list("abc")},
              "10": {"sampled": list("abc"), "promoted": ["c"]}}
    trust = {"1": {c: 0.5 for c in "abc"},
             "10": {"a": 0.2, "b": 0.60, "c": 0.65}}
    g = replay_schedule(_run(rounds, trust, osh=osh))
    pb = 1.0 / (1.0 + _m.exp(-8 * 0.10))
    pc = 1.0 / (1.0 + _m.exp(-8 * 0.15))
    pre = (pb + pc) / (1 + pb + pc)
    post = pc / (2 + pc)
    assert g["demoted_by_budget"] == 1, f"exactly b is demoted: {g}"
    assert g["replay_promoted"] == 1, f"only c survives: {g}"
    # Round 1 promotes nobody, so sum_p = 0 and its gamma is a structural 0
    # -- a round where the budget CANNOT bind. It must be excluded from the
    # mean, not averaged in: including it halved the figure here, and on the
    # real sweep pulled 0.816 down to 0.775.
    assert g["rounds_empty_pool"] == 1, f"round 1 has an empty pool: {g}"
    assert abs(g["gamma_pre"] - pre) < 1e-9, (g["gamma_pre"], pre)
    assert abs(g["gamma_pre_max"] - pre) < 1e-9, (g["gamma_pre_max"], pre)
    assert abs(g["gamma_post"] - post) < 1e-9, (g["gamma_post"], post)
    assert g["rounds_over_budget"] == 1, (
        f"round 10's pre-demotion gamma 0.593 exceeds the 0.35 budget: {g}"
    )
    assert g["gamma_pre_max"] > TC["gamma_budget"] > post, (
        "the budget bit hard (pre 0.593) yet gamma_post (0.278) sits under "
        "the cap -- reading only the post value hides the constraint"
    )
    assert g["set_match"] == 2, (
        f"both rounds must match the log: round 1 promotes nobody (empty "
        f"set both sides) and round 10 promotes exactly {{'c'}}: {g}"
    )
    print("✓ the budget demotes lowest-trust first; gamma_pre exceeds the cap")
    return True


def test_stale_outranks_tier_and_decoy_is_separated():
    """Branch order: is_stale is checked BEFORE any tier, and a Tier-3 client
    drawn as a decoy goes to V while an undrawn one is promoted.

    'd' is sampled at rounds 1-4 and again at round 20. Its last
    verification is round 4, so at round 20 it is 16 rounds stale against
    s_max_rounds = 10 -- staleness outranks its trust of 0.95 and its
    streak of 3. Separately, at round 8 'e' has a streak of 4, is not stale,
    and is over theta_high, and is logged as a decoy, so it must land in the
    decoy column rather than Tier 3.

    Round 8 and not round 5: the ramp is 1 - exp(-(r-1)/5), which is 0.551
    at round 5 -- under theta_high, so the client would be Tier 2 and the
    decoy branch would never be reached. At round 8 the ramp is 0.753 and
    t_eff = min(0.95, 0.753) clears the bar. Getting this wrong is why the
    ramp arithmetic is written out in every fixture here.
    """
    osh = {str(r): {c: {"is_verified": True, "pretrain_loss_var": 0.6}
                    for c in "de"} for r in (1, 2, 3, 4)}
    base = {str(r): {"sampled": list("de"), "verified": list("de")}
            for r in (1, 2, 3, 4)}

    stale_rounds = dict(base)
    stale_rounds["20"] = {"sampled": ["d"]}
    trust = {rk: {"d": 0.95, "e": 0.95} for rk in stale_rounds}
    g = replay_schedule(_run(stale_rounds, trust, osh=osh))
    assert g["stale"] >= 1, f"round 20 must be stale for 'd': {g}"

    decoy_rounds = dict(base)
    decoy_rounds["8"] = {"sampled": ["e"], "verified": ["e"], "decoy": ["e"]}
    trust5 = {rk: {"d": 0.95, "e": 0.95} for rk in decoy_rounds}
    g5 = replay_schedule(_run(decoy_rounds, trust5, osh=osh))
    assert g5["decoy"] == 1, (
        f"a Tier-3 client drawn as a decoy belongs in the decoy column, "
        f"not Tier 3: {g5}"
    )
    print("✓ stale outranks tier; a Tier-3 decoy is counted separately")
    return True


def test_streak_resets_on_low_score_and_on_detection():
    """The two reset paths must be counted separately, because which one
    dominates is a finding: a score at or below 0.8 resets the streak even
    when the detector is happy. raw 1.4 gives
    1 - (1.4-0.6)/0.9 = 0.111 <= 0.8, so that is a score reset."""
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 1.4},
                 "b": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {"1": {"sampled": ["a", "b"], "verified": ["a", "b"]}}
    g = replay_schedule(_run(rounds, {"1": {"a": 0.9, "b": 0.9}}, osh=osh,
                             detected={"1": ["b"]}))
    assert g["verifications"] == 2, g
    assert g["reset_by_score"] == 1, f"a resets on score: {g}"
    assert g["reset_by_detector"] == 1, f"b resets on detection: {g}"
    assert g["streak_reset"] == 2 and g["left_at_k"] == 0, g
    print("✓ score resets and detector resets are counted separately")
    return True


def test_replay_refuses_a_run_without_the_logs_it_needs():
    """Either log missing must give None, not plausible counts.

    With no trust_by_round there is no round-keyed trust (the baseline run
    predates that fix). With no oracle_signal_history there is no behaviour
    score, so every streak stays 0 -- which silently pushes Tier-3 clients
    down into Tier 2 and misreports the decoy and budget columns.
    """
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {"1": {"sampled": ["a"], "verified": ["a"]}}
    assert replay_schedule(_run(rounds, {}, osh=osh)) is None, (
        "must refuse a run with no trust_by_round"
    )
    assert replay_schedule(_run(rounds, {"1": {"a": 0.9}}, osh={})) is None, (
        "must refuse a run with no oracle signals"
    )
    assert replay_schedule(_run(rounds, {"1": {"a": 0.9}}, osh=osh)) is not None
    print("✓ the replay refuses a run missing either log it needs")
    return True


def test_streak_advances_through_a_round_missing_from_trust_by_round():
    """trust_by_round holds only rounds that produced round_analytics --
    test_trust_by_round_logging pins that a round can be absent. The
    attribution needs trust and must skip such a round, but the STREAK
    replay must not: skipping it drops that round's verifications and leaves
    every later streak one short.

    'a' is verified cleanly in rounds 1-4. Dropping round 2's trust entry
    must not change the verification count or where the streak ends up.
    """
    osh = {str(r): {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}
           for r in (1, 2, 3, 4)}
    rounds = {str(r): {"sampled": ["a"], "verified": ["a"]}
              for r in (1, 2, 3, 4)}
    full = {rk: {"a": 0.9} for rk in rounds}
    partial = {rk: v for rk, v in full.items() if rk != "2"}
    g_all = replay_schedule(_run(rounds, full, osh=osh))
    g_gap = replay_schedule(_run(rounds, partial, osh=osh))
    assert g_all["verifications"] == 4, g_all
    assert g_gap["verifications"] == 4, (
        f"round 2's verification must still advance the streak: {g_gap}"
    )
    assert g_gap["left_at_k"] == g_all["left_at_k"], (
        f"the streak desynced across the gap: {g_gap} vs {g_all}"
    )
    assert g_gap["skipped_no_trust"] == 1, (
        f"the untrusted round must be recorded, not silently dropped: {g_gap}"
    )
    assert g_gap["rounds"] == g_all["rounds"] - 1, g_gap
    print("✓ a round missing from trust_by_round still advances the streak")
    return True


def test_bootstrap_off_moves_first_appearances_from_stale_to_tier1():
    """REVIEW FINDING 1. is_stale() reports a never-verified client as stale
    only when bootstrap_verify_new_clients is True. With that ablation flag
    off it returns False -- its own comment says a never-verified client
    trivially passes the s_max caps -- so the client reaches the tier
    branches and lands in Tier 1, because initial_trust 0.25 is under
    theta_low 0.3.

    Hardcoding the bootstrap put those client-rounds in the stale column and
    made rates() subtract a floor that was not there, giving a NEGATIVE
    forced-stale rate (-0.500 against a raw rate of 0.000).
    """
    osh = {"1": {c: {"is_verified": True, "pretrain_loss_var": 0.6}
                 for c in "ab"}}
    rounds = {"1": {"sampled": ["a", "b"], "verified": ["a", "b"]},
              "2": {"sampled": ["a", "b"], "verified": ["a"],
                    "promoted": ["b"]}}
    trust = {"1": {"a": 0.2, "b": 0.2}, "2": {"a": 0.2, "b": 0.2}}

    # Round 1: both clients are first appearances. Round 2: both have been
    # verified so they are no longer stale, and trust 0.2 < theta_low 0.3
    # puts them in Tier 1 either way -- so only the ROUND-1 pair moves
    # between columns when the flag changes.
    on = replay_schedule(_run(rounds, trust, osh=osh))
    assert on["stale"] == 2 and on["tier1"] == 2, (
        f"bootstrap ON: round 1 stale, round 2 Tier 1: {on}"
    )

    tc_off = dict(TC, bootstrap_verify_new_clients=False)
    off = replay_schedule(_run(rounds, trust, osh=osh, tc=tc_off))
    assert off["stale"] == 0, (
        f"with the bootstrap OFF, a never-verified client is NOT stale: {off}"
    )
    assert off["tier1"] == 4, (
        f"all four client-rounds fall through to Tier 1 instead "
        f"(trust 0.2 < theta_low 0.3): {off}"
    )
    assert off["stale"] + off["tier1"] == on["stale"] + on["tier1"], (
        "the flag moves client-rounds between the two columns; it must not "
        "create or destroy any"
    )

    # And the floor rates() subtracts must follow the same flag.
    r_on = rates(_run(rounds, trust, osh=osh))
    r_off = rates(_run(rounds, trust, osh=osh, tc=tc_off))
    assert r_on["bootstrap"] is True and r_off["bootstrap"] is False
    assert r_off["forced_stale_net"] >= 0.0, (
        f"no floor exists with the bootstrap off, so nothing may be "
        f"subtracted; a negative rate is nonsense: {r_off}"
    )
    print("✓ the bootstrap flag moves first appearances stale -> tier1")
    return True


def test_logged_decoy_wins_over_a_replayed_tier():
    """REVIEW FINDING 4. The decoy draw is ground truth from the log; the
    replayed tier is a reconstruction. Testing the tier first meant a client
    the scheduler really drew as a decoy, but whose replayed streak fell one
    short, was counted as Tier 2 AND put into the tentative pool -- reported
    as skipped when reality had verified it.

    Fixture: 'a' is logged as a decoy at round 8 but has a streak of only 1
    (one clean verification, at round 1), so the Tier-3 test fails. It must
    still land in the decoy column and stay out of the pool.
    """
    osh = {"1": {"a": {"is_verified": True, "pretrain_loss_var": 0.6}}}
    rounds = {"1": {"sampled": ["a"], "verified": ["a"]},
              "8": {"sampled": ["a"], "verified": ["a"], "decoy": ["a"]}}
    trust = {"1": {"a": 0.9}, "8": {"a": 0.9}}
    g = replay_schedule(_run(rounds, trust, osh=osh))
    assert g["decoy"] == 1, (
        f"the logged decoy must be read before the replayed tier: {g}"
    )
    assert g["tier2_tentative"] == 0, (
        f"and it must NOT enter the tentative pool: {g}"
    )
    assert g["replay_promoted"] == 0, g
    print("✓ a logged decoy is read before the replayed tier")
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
        test_tier2_is_promoted_not_blocked(),
        test_budget_demotes_lowest_trust_first_and_gamma_pre_exceeds_cap(),
        test_stale_outranks_tier_and_decoy_is_separated(),
        test_streak_resets_on_low_score_and_on_detection(),
        test_replay_refuses_a_run_without_the_logs_it_needs(),
        test_streak_advances_through_a_round_missing_from_trust_by_round(),
        test_bootstrap_off_moves_first_appearances_from_stale_to_tier1(),
        test_logged_decoy_wins_over_a_replayed_tier(),
        test_rates_returns_none_for_a_run_with_no_cohort(),
        test_load_runs_matches_load_seeds_on_suffixed_dirs(),
        test_load_runs_rejects_a_duplicate_seed(),
    ]
    if all(ok):
        print("\n\U0001F3AF All staleness-analysis tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
