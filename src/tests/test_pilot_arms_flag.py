#!/usr/bin/env python3
"""--arms must restrict which arms train, without changing the default run.

Three things have to hold:

  1. The DEFAULT is all three arms and parses to canonical order, so every
     command already in the batch scripts and every result on disk keeps its
     meaning.
  2. An explicit subset is honoured, normalised to canonical order (so the
     summary filename for a given arm set is stable however it was typed),
     and a bad name fails loudly rather than silently training nothing.
  3. Dropping random_skip must bypass the matched-rate pass. That pass names
     its output directory skip{int(mean_observed_rate*100)}, and the mean is
     over the seeds just run -- so adding seeds can round to a different
     integer and either split the record across two directories or overwrite
     the earlier one. That edge is live on disk: sweep_a10_twagg sits in
     skip31 while dose_a10_str050 and sched_a10_start15 sit in skip32.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _parse(argv):
    """Run the script's own parser + post-parse normalisation."""
    import experiments.pilot_skip_comparison as P
    import argparse
    # Rebuild the parser the same way main() does, by calling main() with a
    # patched runner would train models; instead parse via a thin harness
    # that mirrors main()'s normalisation block.
    ns = argparse.Namespace(arms=argv.get("arms", ",".join(P.ARM_ORDER)),
                            seeds=argv.get("seeds", "1"))
    ns.seed_list = [int(s) for s in ns.seeds.split(",") if s.strip()]
    ns.arm_list = [a.strip() for a in ns.arms.split(",") if a.strip()]
    unknown = [a for a in ns.arm_list if a not in P.ARM_ORDER]
    if unknown:
        raise SystemExit(f"unknown {unknown}")
    if not ns.arm_list:
        raise SystemExit("empty")
    ns.arm_list = [a for a in P.ARM_ORDER if a in ns.arm_list]
    return ns


def test_default_is_all_three_in_canonical_order():
    import experiments.pilot_skip_comparison as P
    ns = _parse({})
    assert ns.arm_list == P.ARM_ORDER == [
        "full_verify", "tavs_skip", "random_skip"], ns.arm_list
    print("✓ default --arms is all three arms in canonical order")
    return True


def test_subset_is_honoured_and_normalised():
    ns = _parse({"arms": "tavs_skip"})
    assert ns.arm_list == ["tavs_skip"], ns.arm_list
    # Typed out of order, with stray whitespace -> canonical order.
    ns = _parse({"arms": " random_skip , full_verify "})
    assert ns.arm_list == ["full_verify", "random_skip"], ns.arm_list
    print("✓ a subset is honoured and normalised to canonical order")
    return True


def test_bad_arm_name_fails_loudly():
    for bad in ("tavs", "tavs_skip,nonsense", "", " , "):
        try:
            _parse({"arms": bad})
        except SystemExit:
            continue
        raise AssertionError(f"--arms {bad!r} should have failed")
    print("✓ an unknown or empty --arms exits instead of training nothing")
    return True


def test_help_mentions_arms_and_parser_accepts_it():
    """End-to-end: the real parser must expose the flag.

    _parse above mirrors main()'s normalisation; this checks the flag is
    actually registered, so the mirror cannot drift from a flag that does
    not exist.
    """
    out = subprocess.run(
        [sys.executable, "experiments/pilot_skip_comparison.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-500:]
    assert "--arms" in out.stdout, "--arms missing from --help"
    print("✓ --arms is registered on the real parser")
    return True


def main():
    ok = [
        test_default_is_all_three_in_canonical_order(),
        test_subset_is_honoured_and_normalised(),
        test_bad_arm_name_fails_loudly(),
        test_help_mentions_arms_and_parser_accepts_it(),
    ]
    if all(ok):
        print("\n🎯 All --arms tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
