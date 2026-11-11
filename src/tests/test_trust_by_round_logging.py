#!/usr/bin/env python3
"""trust_by_round / tier_by_round must be usable where trust_evolution is not.

THE DEFECT BEING FIXED
`trust_evolution[cid]` is built by appending one value per round starting
from the round that client was first seen, so each client's list has a
different length and carries no record of which round index 0 refers to. In
a real 40-round run the lengths ranged 21-40. Recovering the round by the
obvious offset (num_rounds - len + 1) was wrong for 86 of 123 promoted
client-rounds, so any per-round analysis built on it is silently wrong.

A second, independent defect: `tier_evolution` cannot be read as
"promotable" either. schedule_verifications tentatively promotes Tier-2
clients and only then demotes by the weight budget, so a client promoted in
8 separate rounds had a tier series containing no 3 at all.

These tests pin:
  1. the new fields are keyed by ROUND and every round is present;
  2. a client that joins late appears from its join round onward, with no
     phantom entries before it -- the exact thing the ragged list lost;
  3. the legacy field still has the ragged shape, so the fix is additive and
     old analyses do not change;
  4. the reconstruction the ragged form forced on us is no longer needed:
     the round-keyed trust matches what the scheduler actually held.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class _Decision:
    def __init__(self, trust, tiers):
        self.trust_scores = trust
        self.tier_assignments = tiers


class _Analytics:
    def __init__(self, rnd, trust, tiers):
        self.round_number = rnd
        self.scheduling_decision = _Decision(trust, tiers)


def _build(rounds):
    """Call the PRODUCTION helper, not a copy of it.

    An earlier version of this file re-implemented the pipeline's loop here.
    That made every test vacuous: deleting the round-keyed lines from
    end_to_end_pipeline.py left all of them passing. Importing the real
    function is the whole point -- if the production loop changes shape,
    these tests see it.
    """
    from src.tavs_v2.end_to_end_pipeline import build_trust_logs
    return build_trust_logs(rounds)


def _fixture():
    """3 rounds. 'a' is present throughout; 'b' only joins at round 2.

    That asymmetry is the whole bug: 'b' ends up with a shorter list and
    nothing records that it starts at round 2.
    """
    return [
        _Analytics(1, {"a": 0.25}, {"a": 1}),
        _Analytics(2, {"a": 0.50, "b": 0.25}, {"a": 2, "b": 1}),
        _Analytics(3, {"a": 0.80, "b": 0.40}, {"a": 3, "b": 2}),
    ]


def test_keyed_by_round_and_complete():
    _te, _tv, tbr, tir = _build(_fixture())
    assert sorted(tbr) == ["1", "2", "3"], tbr.keys()
    assert sorted(tir) == ["1", "2", "3"]
    # Round keys are strings so a json round-trip changes nothing.
    import json
    assert json.loads(json.dumps(tbr)) == tbr, "must survive json round-trip"
    print("✓ trust_by_round / tier_by_round are keyed by round and complete")
    return True


def test_late_joiner_is_unambiguous():
    """The defect: 'b' joins at round 2, so its ragged list is shorter and
    the round each entry belongs to is unrecoverable. The round-keyed form
    must place b's values at rounds 2 and 3 and nowhere else."""
    te, _tv, tbr, _tir = _build(_fixture())

    # Round-keyed: exact and unambiguous.
    assert "b" not in tbr["1"], "b had not joined at round 1"
    assert tbr["2"]["b"] == 0.25 and tbr["3"]["b"] == 0.40
    assert tbr["1"]["a"] == 0.25 and tbr["3"]["a"] == 0.80

    # Ragged: the information is genuinely absent. b's list is [0.25, 0.40];
    # read naively as rounds 1..2 it reports b's round-3 value at round 2 and
    # claims b existed at round 1. Assert the ambiguity really is there, so
    # this test fails if someone "fixes" the legacy field in place instead.
    assert te["b"] == [0.25, 0.40] and len(te["b"]) != len(te["a"]), te
    naive_round_for_last = len(te["b"])          # = 2, but the true round is 3
    assert naive_round_for_last != 3, (
        "if the ragged list ever becomes round-aligned, this test and the "
        "field docstring are both stale"
    )
    print("✓ a late joiner is placed at its true rounds; the ragged form "
          "demonstrably cannot do this")
    return True


def test_legacy_fields_unchanged():
    """The fix must be additive: old analyses must see the same ragged data."""
    te, tv, _tbr, _tir = _build(_fixture())
    assert te == {"a": [0.25, 0.50, 0.80], "b": [0.25, 0.40]}, te
    assert tv == {"a": [1, 2, 3], "b": [1, 2]}, tv
    print("✓ legacy trust_evolution / tier_evolution are byte-identical")
    return True


def test_tier_is_recorded_per_round_not_inferred():
    """tier_by_round must carry the scheduler's own tier for every client it
    knew that round, defaulting absent ones to 1 exactly as before -- not an
    inference from promoted/verified, which is what made the legacy field
    unreadable."""
    _te, _tv, _tbr, tir = _build([
        _Analytics(1, {"a": 0.9, "b": 0.2}, {"a": 3}),   # b has no tier entry
    ])
    assert tir["1"] == {"a": 3, "b": 1}, tir
    print("✓ tier_by_round records the scheduler's tier, defaulting to 1")
    return True


def test_dataclass_exposes_the_fields():
    """Guard the wiring: the fields must exist on PipelineResults with dict
    defaults, or the pipeline would construct fine and log nothing."""
    import dataclasses
    from src.tavs_v2.end_to_end_pipeline import PipelineResults
    names = {f.name: f for f in dataclasses.fields(PipelineResults)}
    for f in ("trust_by_round", "tier_by_round"):
        assert f in names, f"{f} missing from PipelineResults"
        assert names[f].default_factory is dict, (
            f"{f} must default to an empty dict so older construction sites "
            f"and cached runs keep working"
        )
    print("✓ PipelineResults exposes both fields with dict defaults")
    return True


def test_round_with_no_decision_is_absent_not_empty():
    """A round that produced no aggregate never reaches round_analytics, so
    it is ABSENT from the round-keyed dicts rather than present-and-empty.

    Pinned because the field promises per-round data and a consumer looping
    range(1, num_rounds+1) would raise KeyError on such a run. The contract
    is "iterate the keys", and this test states it.
    """
    rounds = [_Analytics(1, {"a": 0.25}, {"a": 1}),
              _Analytics(3, {"a": 0.60}, {"a": 2})]      # round 2 missing
    _te, _tv, tbr, _tir = _build(rounds)
    assert sorted(tbr) == ["1", "3"], tbr.keys()
    assert "2" not in tbr, "a round with no analytics must be absent"
    assert tbr.get("2", {}) == {}, ".get(rk, {}) is the safe access pattern"
    print("✓ a round with no analytics is absent, not silently empty")
    return True


def test_falsy_decision_is_skipped():
    """An analytics entry whose scheduling_decision is None must be skipped
    rather than raising -- the production helper uses getattr + truthiness."""
    class _NoDecision:
        round_number = 2
        scheduling_decision = None
    _te, _tv, tbr, _tir = _build([_Analytics(1, {"a": 0.3}, {"a": 1}), _NoDecision()])
    assert sorted(tbr) == ["1"], tbr.keys()
    print("✓ an analytics entry with no scheduling_decision is skipped safely")
    return True


def main():
    ok = [
        test_keyed_by_round_and_complete(),
        test_late_joiner_is_unambiguous(),
        test_legacy_fields_unchanged(),
        test_tier_is_recorded_per_round_not_inferred(),
        test_dataclass_exposes_the_fields(),
        test_round_with_no_decision_is_absent_not_empty(),
        test_falsy_decision_is_skipped(),
    ]
    if all(ok):
        print("\n🎯 All trust-by-round logging tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
