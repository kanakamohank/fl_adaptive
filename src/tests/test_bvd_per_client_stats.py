#!/usr/bin/env python3
"""
Unit tests for the per-client diagnostic stats that
BlockVarianceDetector.detect_outliers writes into self.last_stats["per_client"].

The strategy's scheduling_history spreads self.last_stats into every round's
entry (prefixed `det_`), so pipeline_results.json carries det_per_client
verbatim and the analyze_bvd_signal diagnostic consumes it. If any of these
tests regress, the diagnostic report silently stops answering the
"SNR too low / sigma eats signal / tau_z miscalibrated" question that
Step 1 of the BVD investigation plan was designed to answer.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.tavs_v2.algo3_bvd_aggregation import BlockVarianceDetector


def _detector(tau=10.0, alpha_sigma=0.9):
    return BlockVarianceDetector(tau_z=tau, alpha_sigma=alpha_sigma, epsilon_stab=1e-5)


def _toy_cohort(byzantine_z_target_mag=50.0):
    """9 honest clients + 1 byzantine with a localised poison in one block."""
    torch.manual_seed(42)
    blocks = ["b_normal", "b_poisoned", "b_other"]
    dim = 20
    updates, verified = {}, set()
    for i in range(9):
        cid = f"honest_{i}"
        verified.add(cid)
        updates[cid] = {m: torch.randn(dim) * 2.0 for m in blocks}
    byz = "byz"
    verified.add(byz)
    updates[byz] = {
        "b_normal":   torch.randn(dim) * 2.0,
        "b_poisoned": torch.randn(dim) * 2.0 + byzantine_z_target_mag,
        "b_other":    torch.randn(dim) * 2.0,
    }
    return updates, verified, blocks, byz


def test_per_client_populated_and_shaped():
    """Every verified client has an entry with the documented keys."""
    print("Testing per_client stats shape...")
    bvd = _detector()
    updates, verified, _, _ = _toy_cohort()
    bvd.detect_outliers(updates, verified)

    per = bvd.last_stats.get("per_client")
    assert per is not None, "per_client missing from last_stats"
    assert set(per.keys()) == verified, (
        f"per_client keys don't match verified set: "
        f"{set(per.keys()) ^ verified}"
    )
    required = {"max_z", "raw_dist_at_argmax", "sigma_sq_at_argmax",
                "argmax_block", "rank_raw_at_argmax", "cohort_size_at_argmax"}
    for cid, stats in per.items():
        missing = required - set(stats.keys())
        assert not missing, f"{cid}: missing keys {missing}"
        assert isinstance(stats["max_z"], float)
        assert isinstance(stats["rank_raw_at_argmax"], int)
        assert isinstance(stats["cohort_size_at_argmax"], int)
    print("✓ per_client present for every verified client with all required keys")
    return True


def test_per_client_max_z_is_true_cross_block_max():
    """max_z and argmax_block must correspond to the GLOBALLY maximising
    block, not just agree with each other on whichever block was stored.

    Reviewer flagged the earlier version as a tautology: it only verified
    `max_z == raw_dist_at_argmax / (sigma_sq_at_argmax + eps)`, which holds
    by construction regardless of whether argmax was correct. Independent
    reconstruction: for each client, snapshot sigma_sq BEFORE calling
    detect_outliers (so step-3 EMA update doesn't shift the denominator
    the test compares against), compute Z at every block from the
    production LOO-median formula, and verify the stored argmax_block is
    the max-Z block AND the stored max_z equals that Z.
    """
    print("Testing per_client.max_z is true cross-block max...")
    bvd = _detector()
    updates, verified, blocks, _ = _toy_cohort()

    # Prime sigma_sq with a non-trivial known value per block (not the seed
    # case), then snapshot BEFORE the call. Step-1 skips seeding since
    # sigma_sq[m] is already present; step-2 uses this exact sigma for all
    # Z computations; step-3 updates the EMA afterwards.
    torch.manual_seed(123)
    bvd.sigma_sq = {m: 0.5 + 0.1 * i for i, m in enumerate(blocks)}
    sigma_snapshot = dict(bvd.sigma_sq)

    bvd.detect_outliers(updates, verified)
    per = bvd.last_stats["per_client"]

    for cid in verified:
        s = per[cid]
        if not s["argmax_block"]:
            assert s["max_z"] == 0.0
            continue
        # Reconstruct Z at every block from scratch using LOO-median and the
        # pre-call sigma snapshot.
        best_block_true, best_z_true = None, float("-inf")
        for m in blocks:
            others_stack = torch.stack([updates[c][m] for c in verified if c != cid])
            centre = torch.median(others_stack, dim=0).values
            raw = float(torch.sum((updates[cid][m] - centre) ** 2).item())
            z = raw / (sigma_snapshot[m] + bvd.epsilon_stab)
            if z > best_z_true:
                best_z_true, best_block_true = z, m
        assert s["argmax_block"] == best_block_true, (
            f"{cid}: stored argmax_block {s['argmax_block']} != true max-Z "
            f"block {best_block_true}"
        )
        assert math.isclose(s["max_z"], best_z_true, rel_tol=1e-6, abs_tol=1e-9), (
            f"{cid}: stored max_z {s['max_z']:.6g} != independently recomputed "
            f"{best_z_true:.6g}"
        )
        # And the recorded tuple is self-consistent with the recorded max_z.
        recomputed = s["raw_dist_at_argmax"] / (s["sigma_sq_at_argmax"] + bvd.epsilon_stab)
        assert math.isclose(recomputed, s["max_z"], rel_tol=1e-9, abs_tol=1e-12)
    print("✓ argmax_block AND max_z match independent cross-block recomputation")
    return True


def test_per_client_rank_bounds_and_byz_is_top():
    """rank is 1..cohort_size, inclusive. In the localised-poison toy cohort
    the byzantine client should rank highest at its poisoned block (its
    raw_dist is 50^2 bigger than the honest spread), i.e. rank == cohort_size
    at the poisoned block."""
    print("Testing rank_raw_at_argmax bounds and byzantine top-rank...")
    bvd = _detector()
    updates, verified, _, byz = _toy_cohort()
    bvd.detect_outliers(updates, verified)
    per = bvd.last_stats["per_client"]

    for cid, s in per.items():
        if not s["argmax_block"]:
            continue
        assert 1 <= s["rank_raw_at_argmax"] <= s["cohort_size_at_argmax"], (
            f"{cid}: rank {s['rank_raw_at_argmax']} out of bounds "
            f"[1, {s['cohort_size_at_argmax']}]"
        )
        assert s["cohort_size_at_argmax"] == len(verified), (
            f"{cid}: cohort_size {s['cohort_size_at_argmax']} != "
            f"len(verified) {len(verified)}"
        )

    byz_stats = per[byz]
    assert byz_stats["rank_raw_at_argmax"] == byz_stats["cohort_size_at_argmax"], (
        f"byz expected rank=cohort_size, got "
        f"{byz_stats['rank_raw_at_argmax']} / {byz_stats['cohort_size_at_argmax']}"
    )
    assert byz_stats["argmax_block"] == "b_poisoned", (
        f"byz argmax should be the poisoned block, got {byz_stats['argmax_block']}"
    )
    print(f"✓ ranks in [1, N]; byz tops rank at poisoned block "
          f"({byz_stats['rank_raw_at_argmax']}/{byz_stats['cohort_size_at_argmax']})")
    return True


def test_byz_still_flagged_in_toy_cohort():
    """Narrow regression: in the SINGLE toy cohort (seed=42, mag=100), the
    byzantine client is still flagged as outlier after the per_client
    refactor, and no honest client is spuriously flagged. Does not generalise
    beyond this fixture -- a wider claim ("verdict unchanged under all
    inputs") would need parametric testing."""
    print("Testing byz flagged in toy cohort...")
    bvd = _detector(tau=10.0)
    updates, verified, _, byz = _toy_cohort(byzantine_z_target_mag=100.0)
    inliers, outliers, _ = bvd.detect_outliers(updates, verified)
    assert byz in outliers, f"byz not detected: outliers={outliers}"
    assert byz not in inliers
    for cid in verified:
        if cid == byz:
            continue
        assert cid in inliers, f"honest {cid} wrongly flagged outlier"
    print(f"✓ byz flagged (toy cohort only), 9 honest in inliers")
    return True


def test_per_client_nan_numerator_fallthrough():
    """Companion to the NaN-denominator test: force NaN in the gradient
    tensor itself so raw_distance is NaN and sigma is finite. Same NaN
    guard (`math.isnan(z)`) should keep the client out of outliers and
    emit sentinel per_client stats rather than KeyError on best_block=None."""
    print("Testing NaN-numerator fallthrough...")
    bvd = _detector()
    torch.manual_seed(11)
    dim = 10
    blocks = ["a", "b"]
    updates = {f"c{i}": {m: torch.randn(dim) * 2.0 for m in blocks}
               for i in range(5)}
    verified = set(updates.keys())
    # First call seeds sigma_sq cleanly. Then poison one client's gradient
    # block with NaN values so raw_distance for that block becomes NaN.
    bvd.detect_outliers(updates, verified)
    poisoned = "c0"
    updates[poisoned]["a"] = torch.full((dim,), float("nan"))
    updates[poisoned]["b"] = torch.full((dim,), float("nan"))
    inliers, outliers, _ = bvd.detect_outliers(updates, verified)
    assert poisoned not in outliers, (
        f"NaN-gradient client falsely flagged outlier: {outliers}"
    )
    assert poisoned in inliers
    s = bvd.last_stats["per_client"][poisoned]
    assert s["max_z"] == 0.0, f"NaN numerator should fall through to 0, got {s['max_z']}"
    assert s["argmax_block"] == ""
    print("✓ NaN-numerator client falls through to inlier with sentinel stats")
    return True


def test_per_client_nan_sigma_fallthrough():
    """If sigma_sq becomes NaN for a block (contrived; could happen under a
    pathological projector), Z computation yields NaN at that block, and
    the new NaN-safe max tracker skips it rather than KeyError-ing on
    best_block=None. For a client whose EVERY block has NaN sigma, the
    fallthrough path puts them in inliers at max_z=0, argmax_block=''."""
    print("Testing NaN-safe max-Z fallthrough...")
    bvd = _detector()
    torch.manual_seed(5)
    dim = 10
    blocks = ["a", "b"]
    updates = {f"c{i}": {m: torch.randn(dim) * 2.0 for m in blocks}
               for i in range(5)}
    verified = set(updates.keys())
    # First call seeds sigma_sq from LOO median distances; then force a NaN
    # sigma_sq entry to simulate the pathological regime.
    bvd.detect_outliers(updates, verified)
    bvd.sigma_sq = {m: float("nan") for m in blocks}
    # Second call: every per-block Z is NaN -> best_block should fallthrough
    # for every client. All land in inliers; last_stats carries sentinels.
    inliers, outliers, _ = bvd.detect_outliers(updates, verified)
    assert outliers == set(), f"NaN sigma spuriously flagged outliers: {outliers}"
    assert inliers == verified
    for cid in verified:
        s = bvd.last_stats["per_client"][cid]
        assert s["max_z"] == 0.0, f"{cid}: fallthrough expected max_z=0, got {s['max_z']}"
        assert s["argmax_block"] == "", f"{cid}: expected empty argmax_block"
        assert s["rank_raw_at_argmax"] == 0
    print("✓ NaN sigma produces inlier fallthrough with sentinel per_client fields")
    return True


def main():
    print("🧪 BVD per-client diagnostic stats Test Suite")
    print("=" * 60)
    tests = [
        test_per_client_populated_and_shaped,
        test_per_client_max_z_is_true_cross_block_max,
        test_per_client_rank_bounds_and_byz_is_top,
        test_byz_still_flagged_in_toy_cohort,
        test_per_client_nan_sigma_fallthrough,
        test_per_client_nan_numerator_fallthrough,
    ]
    for t in tests:
        assert t() is True
    print("\n🎯 All BVD per-client diagnostic tests PASSED!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
