#!/usr/bin/env python3
"""
Unit tests for experiments/analyze_bvd_signal.

The script is a diagnostic: it consumes pipeline_results.json's
scheduling_history (specifically det_per_client), joins against the ground-
truth noisy identity, and produces a verdict. Covers:
  * collect() uses the entry's own "round" field (not enumerate index)
  * noisy_cids() correctly inverts the cid -> client_config_id bridge
  * _verdict() covers all four quadrants of (raw_strong, z_strong) plus
    the two detector-fires-anyway cases, no catch-all, no silent misclassify
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)


def _load_module():
    path = os.path.join(HERE, "experiments", "analyze_bvd_signal.py")
    spec = importlib.util.spec_from_file_location("analyze_bvd_signal", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_noisy_cids_inverts_bridge():
    """noisy_cids returns the Flower cids whose config_id is in the noisy set."""
    print("Testing noisy_cids bridge inversion...")
    m = _load_module()
    pr = {
        "noisy_client_config_ids": ["honest_01", "honest_02"],
        "cid_to_client_config_id": {
            "cid_a": "honest_01",
            "cid_b": "honest_03",   # clean
            "cid_c": "honest_02",
            "cid_d": "honest_09",   # clean
        },
    }
    result = m.noisy_cids(pr)
    assert result == {"cid_a", "cid_c"}, f"got {result!r}"
    # Degenerate: no noisy config set -> empty.
    assert m.noisy_cids({"noisy_client_config_ids": []}) == set()
    # Degenerate: no bridge -> empty, not crash.
    assert m.noisy_cids({"noisy_client_config_ids": ["honest_01"]}) == set()
    print("✓ noisy_cids correctly inverts cid_to_client_config_id")
    return True


def test_collect_uses_explicit_round_field():
    """Historical regression: previously the diagnostic numbered rounds by
    enumerate(start=1), which breaks if scheduling_history is filtered,
    reordered, or if a run skipped a round. collect() must honour the
    entry's own 'round' field."""
    print("Testing collect() round labelling...")
    m = _load_module()
    pr = {
        "noisy_client_config_ids": ["honest_01"],
        "cid_to_client_config_id": {"cid_a": "honest_01", "cid_b": "honest_02"},
        "scheduling_history": [
            # Non-sequential rounds -- simulates filtering or legacy caches
            # where round 2 is missing. enumerate-based labelling would call
            # these rounds 1 and 2; the correct labels are 1 and 7.
            {"round": 1, "det_per_client": {
                "cid_a": {"max_z": 2.1, "raw_dist_at_argmax": 0.5,
                          "sigma_sq_at_argmax": 0.24, "argmax_block": "b0",
                          "rank_raw_at_argmax": 2, "cohort_size_at_argmax": 2},
                "cid_b": {"max_z": 1.0, "raw_dist_at_argmax": 0.24,
                          "sigma_sq_at_argmax": 0.24, "argmax_block": "b0",
                          "rank_raw_at_argmax": 1, "cohort_size_at_argmax": 2},
            }},
            {"round": 7, "det_per_client": {
                "cid_a": {"max_z": 3.0, "raw_dist_at_argmax": 0.72,
                          "sigma_sq_at_argmax": 0.24, "argmax_block": "b0",
                          "rank_raw_at_argmax": 2, "cohort_size_at_argmax": 2},
            }},
        ],
    }
    rows, noisy = m.collect(pr)
    rounds = sorted({r["round"] for r in rows})
    assert rounds == [1, 7], f"expected rounds [1, 7]; got {rounds}"
    assert noisy == {"cid_a"}
    rd7_a = next(r for r in rows if r["round"] == 7 and r["cid"] == "cid_a")
    assert rd7_a["noisy"] is True
    assert rd7_a["max_z"] == 3.0
    assert rd7_a["rank_raw"] == 2
    print("✓ collect() uses explicit round field and labels noisy correctly")
    return True


def test_collect_handles_old_runs_without_det_per_client():
    """A legacy run with scheduling_history but no det_per_client fields
    yields empty rows (not crash). The script's main() turns this into a
    clean SystemExit; collect() just gives nothing."""
    print("Testing collect() legacy-cache path...")
    m = _load_module()
    pr = {
        "noisy_client_config_ids": ["honest_01"],
        "cid_to_client_config_id": {"cid_a": "honest_01"},
        "scheduling_history": [{"round": 1}, {"round": 2}],
    }
    rows, _ = m.collect(pr)
    assert rows == []
    print("✓ legacy cache yields empty rows, no crash")
    return True


def test_verdict_quadrants():
    """_verdict covers all four (raw_strong, z_strong) quadrants plus the
    two detector-fires-anyway cases, with distinct messages. Previously
    the function had an unreachable 'ambiguous' catch-all that misclassified
    the (z_strong, not raw_strong) and (detector_fires, not z_strong) cells.
    Pin them explicitly.

    Derives low/high from the module's exposed constants rather than
    hardcoded 5 and 18, so tightening _SEP_FRAC from 0.6 to e.g. 0.7 does
    not break this test for an unrelated reason."""
    print("Testing _verdict() quadrant coverage...")
    m = _load_module()
    rounds = 20
    sep_frac = getattr(m, "_SEP_FRAC", 0.6)
    fires_frac = getattr(m, "_FIRES_FRAC", 0.25)
    # Pick low/high so they land clearly below/above the threshold even
    # if _SEP_FRAC is tightened within reason (up to ~0.95).
    low = max(0, int(sep_frac * rounds) - 2)                # clearly below
    high = min(rounds, int(sep_frac * rounds) + 2)          # clearly at/above
    # Likewise for detector_fires = cross / noisy_total >= _FIRES_FRAC.
    noisy_total = 20
    fires_cross = max(1, int(fires_frac * noisy_total) + 1)   # clearly fires
    nofire_cross = 0                                           # clearly doesn't

    cases = {}

    # (raw_strong=T, z_strong=T, detector_fires=T)
    v = m._verdict(sep_raw=high, sep_z=high, cross=fires_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "detector fires" in v.lower() and "aggregate across seeds" in v.lower(), v
    cases["TTT"] = v

    # (raw_strong=F, z_strong=F, detector_fires=T) -> long-tail
    v = m._verdict(sep_raw=low, sep_z=low, cross=fires_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "long-tail" in v.lower() or "lucky-high-z" in v.lower(), v
    cases["FFT"] = v

    # (raw_strong=F, z_strong=T, detector_fires=F) -> sigma collapsed
    v = m._verdict(sep_raw=low, sep_z=high, cross=nofire_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "sigma" in v.lower() and "collapse" in v.lower(), v
    cases["FTF"] = v

    # (raw_strong=T, z_strong=F, detector_fires=F) -> sigma eats signal
    v = m._verdict(sep_raw=high, sep_z=low, cross=nofire_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "eats" in v.lower(), v
    cases["TFF"] = v

    # (raw_strong=T, z_strong=T, detector_fires=F) -> calibration miss
    v = m._verdict(sep_raw=high, sep_z=high, cross=nofire_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "calibration" in v.lower(), v
    cases["TTF"] = v

    # (raw_strong=F, z_strong=F, detector_fires=F) -> SNR too low
    v = m._verdict(sep_raw=low, sep_z=low, cross=nofire_cross,
                   noisy_total=noisy_total, rounds=rounds, tau_z=5.0)
    assert "snr" in v.lower() or "noise regime" in v.lower(), v
    cases["FFF"] = v

    # Reviewer: pin that every case produces a DISTINCT message. Keyword-only
    # matching could silently collapse two cells into one shared string.
    distinct = set(cases.values())
    assert len(distinct) == len(cases), (
        f"verdict messages collapsed: got {len(distinct)} unique strings "
        f"for {len(cases)} cases. {cases}"
    )
    print(f"✓ all 6 verdict cells distinct and labelled "
          f"(each returns a unique string)")
    return True


def test_verdict_tau_z_in_calibration_message():
    """Reviewer flagged a stale hardcoded tau_z=5.0 in the calibration-miss
    message that would be WRONG if the actual run used a different threshold.
    tau_z must now be threaded through and appear accurately."""
    print("Testing tau_z is threaded into calibration-miss message...")
    m = _load_module()
    rounds = 20
    sep_frac = getattr(m, "_SEP_FRAC", 0.6)
    high = int(sep_frac * rounds) + 2
    v3 = m._verdict(sep_raw=high, sep_z=high, cross=0, noisy_total=20,
                    rounds=rounds, tau_z=3.0)
    v8 = m._verdict(sep_raw=high, sep_z=high, cross=0, noisy_total=20,
                    rounds=rounds, tau_z=8.0)
    assert "3.0" in v3, f"tau_z=3.0 not reflected in message: {v3!r}"
    assert "8.0" in v8, f"tau_z=8.0 not reflected in message: {v8!r}"
    assert "3.0" not in v8 and "8.0" not in v3, (
        f"tau_z values crossed; v3={v3!r} v8={v8!r}"
    )
    print("✓ _verdict reflects tau_z argument in calibration message")
    return True


def test_verdict_handles_zero_noisy_rows():
    """If noisy_total is 0 (e.g. clean run), detector_fires defaults to False
    and the function must not divide-by-zero."""
    print("Testing _verdict() divide-by-zero guard...")
    m = _load_module()
    v = m._verdict(sep_raw=5, sep_z=5, cross=0, noisy_total=0, rounds=20)
    # falls into "SNR too low" bucket (no raw sep, no Z sep, no fires)
    assert isinstance(v, str) and len(v) > 0
    print("✓ zero noisy rows handled without divide-by-zero")
    return True


def main():
    print("🧪 analyze_bvd_signal Test Suite")
    print("=" * 60)
    tests = [
        test_noisy_cids_inverts_bridge,
        test_collect_uses_explicit_round_field,
        test_collect_handles_old_runs_without_det_per_client,
        test_verdict_quadrants,
        test_verdict_tau_z_in_calibration_message,
        test_verdict_handles_zero_noisy_rows,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All analyze_bvd_signal tests PASSED!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
