#!/usr/bin/env python3
"""Did the weighting DOSE or the weighting SCHEDULE change the outcome?

Two questions, both at Dirichlet(1.0), CIFAR-10N worst, one signal:

  DOSE      Trust weighting hurts at this partition. Does the harm grow with
            how hard the weighting pulls? trust_weight_strength scales the
            score entering the aggregation weight:
                score' = 1 - strength * (1 - score)
            so 0.0 is weighting-off and 1.0 is the original behaviour.

  SCHEDULE  Every published method switches reliability weighting on LATE --
            FedCNI (ICME 2023) at round Ts=15 of 100, FedLAW (ICLR 2026)
            learns weights only in the first 20 rounds then freezes. We have
            always weighted from round 1. Arms: start at round 6 (FedCNI's
            15% of training, scaled to our 40 rounds) and round 15.

WHY ENDPOINTS ARE REPORTED FOUR WAYS
An earlier claim quoted t=-3.30 that was -1.81 on a different but equally
defensible endpoint. Picking the endpoint after seeing the data is how a
null result becomes a finding. So every comparison is reported on the final
round and on the mean of the last 3 / 5 / 10 rounds, and the output states
HOW MANY endpoints each effect clears significance on, rather than letting a
reader pick the flattering one.

Single-round endpoints are the weakest here and the script says so: the
weighting-off arm's own round-to-round accuracy swing averages 0.024, which
is larger than any effect measured. That is not under-training -- a 100-round
probe at the same partition shows the same 0.027 jitter -- it is 10-of-50
client sampling under a skewed partition. Trust the tail averages.

PAIRED STATISTICS
Arms share seeds, so differences are paired per seed: t = mean(d) /
(stdev(d)/sqrt(n)). With n=5, |t| > 2.78 is p<0.05 and |t| > 4.60 is p<0.01.
The t already accounts for noise, so no separate "noise floor" is applied --
an earlier version of this script compared a single-SEED scatter against
5-seed MEAN effects, which is apples to oranges and told the reader to
discard effects at p<0.01. The per-seed spread is still printed, labelled as
such, for scale only.

PHASE 0 IS A GATE, NOT A RESULT
strength=0.0 must reproduce the weighting-off run. If it does not, something
other than the aggregation weight differs between arms and nothing else here
can be trusted. Checked on every endpoint, not just the final round.

Usage:
  python3 experiments/analysis/oracle/dose_and_schedule.py
  python3 experiments/analysis/oracle/dose_and_schedule.py --self-test
"""
import argparse
import ast
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path

ARM = "tavs_skip"
ENDPOINTS = [("final", 1), ("last3", 3), ("last5", 5), ("last10", 10)]
# Two-sided p<0.05 critical values of Student's t, by degrees of freedom
# (n-1). This MUST follow the sample size: an earlier version hardcoded 2.78
# (the n=5 value) and labelled every table "at n=5", so the moment a run
# family gained seeds the significance column would have been judged against
# too strict a bar -- at n=10 the bar is 2.26, and an effect at t=2.4 would
# have been reported as not significant when it is.
_T_05 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def t_crit(n):
    """Two-sided 0.05 critical t for a paired sample of size n.

    For a d.o.f. not tabulated, the next LOWER tabulated d.o.f. is used,
    which gives a larger (more conservative) threshold -- never a smaller
    one, so this cannot manufacture significance. Returns inf for n<2,
    where no t exists.
    """
    if n < 2:
        return float("inf")
    dof = n - 1
    if dof in _T_05:
        return _T_05[dof]
    lower = [d for d in _T_05 if d < dof]
    return _T_05[max(lower)] if lower else 1.96

# Config fields allowed to differ between a run family and the
# weighting-off reference. Anything else differing means the comparison is
# not one-variable.
EXPECTED_DIFFS = {
    "output_dir",                          # by construction
    "disable_trust_weighted_aggregation",  # the knob itself
    "trust_weight_strength",               # dose arms
    "trust_weighting_start_round",         # schedule arms
    # Serialised with default=str, so this is a function repr carrying a
    # memory address -- different in every process, never a real difference.
    "evaluate_fn",
}

_SET_REPR = re.compile(r"^(?:frozenset|set)\((\{.*\})\)$", re.S)


def _canon(v):
    """Canonicalise a config value for comparison.

    pipeline_results.json is written with default=str, so unordered
    containers arrive as their repr and the TEXT differs run to run while
    the contents are identical. Without this, every arm is flagged as
    confounded on oracle_clean_client_config_ids, whose frozenset repr
    orders differently per process -- a false alarm that would discredit a
    real one.

    Parsed with ast.literal_eval rather than split(",") so that a set of
    tuples, or numbers versus their string forms, cannot be mangled into
    a false match. Anything literal_eval cannot read is returned
    unchanged, which compares as text -- a false POSITIVE (reported as a
    difference) is safe here; a false negative is not.
    """
    if isinstance(v, str):
        s = v.strip()
        if s in ("frozenset()", "set()"):
            return frozenset()
        m = _SET_REPR.match(s)
        if m:
            try:
                parsed = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                return v
            if isinstance(parsed, (set, frozenset)):
                return frozenset(parsed)
            return v
    if isinstance(v, list):
        return tuple(_canon(x) for x in v)
    return v


def tail_mean(accs, n):
    """Mean of the last n accuracies. n=1 is the final round.

    Returns None for an empty series rather than raising, so one truncated
    run cannot abort a whole table.
    """
    if not accs:
        return None
    n = max(1, min(int(n), len(accs)))
    return statistics.mean(accs[-n:])


def paired_t(diffs):
    """(mean, t, n). t is nan when n<2 or the differences have zero spread.

    Zero spread is the bit-identical case. Callers must treat nan as "no
    detectable difference", never as a failure: an earlier version of the
    Phase-0 gate required `not isnan(t)` to pass and therefore reported
    PERFECT agreement as a systematic difference.
    """
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = statistics.mean(diffs)
    if n < 2:
        return m, float("nan"), n
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return m, float("nan"), n
    return m, m / (sd / math.sqrt(n)), n


def load_seeds(root: Path):
    """{seed -> {endpoint -> accuracy}} for the TAVS arm of one run family.

    Raises on a duplicate seed. These directories contain skip-rate
    subdirectories and can accumulate a second config subdirectory after a
    rerun; silently keeping whichever path sorted last would swap the whole
    series for a different configuration while load_config still described
    the original one as clean. results/oracle/ already holds 37 matching
    files for 5 seeds, so this is not hypothetical.
    """
    out, seen = {}, {}
    pat = str(root / "**" / f"{ARM}_seed*" / "pipeline_results.json")
    for p in sorted(glob.glob(pat, recursive=True)):
        m = re.search(rf"{ARM}_seed(\d+)", p)
        if not m:
            continue
        seed = str(int(m.group(1)))   # seed01 and seed1 are the same seed
        accs = json.loads(Path(p).read_text()).get("server_accuracies") or []
        vals = {name: tail_mean(accs, n) for name, n in ENDPOINTS}
        if vals["final"] is None:
            continue
        if seed in out:
            raise RuntimeError(
                f"duplicate seed {seed} under {root}:\n  {seen[seed]}\n  {p}\n"
                f"Two configurations are present. Point the script at the "
                f"specific config subdirectory instead."
            )
        out[seed], seen[seed] = vals, p
    return out


def load_config(root: Path):
    """Flattened config of one run family: experiment keys + tavs_config.

    Reads the first seed found; seeds differ only in `seed`, which is
    dropped, so one file characterises the family.
    """
    pat = str(root / "**" / f"{ARM}_seed*" / "pipeline_results.json")
    files = sorted(glob.glob(pat, recursive=True))
    if not files:
        return {}
    cfg = (json.loads(Path(files[0]).read_text()).get("config") or {}).copy()
    tc = cfg.pop("tavs_config", None) or {}
    cfg.pop("seed", None)
    # Experiment-level keys are prefixed so a name shared with a
    # tavs_config key cannot silently overwrite it. The prefix is stripped
    # with removeprefix, NOT split(".")[-1]: splitting on the last dot would
    # also strip a dotted tavs_config key and could exempt a real confound
    # from EXPECTED_DIFFS.
    merged = {f"exp.{k}": v for k, v in cfg.items()}
    merged.update({f"tavs.{k}": v for k, v in tc.items()})
    return merged


def _bare(key):
    for pre in ("exp.", "tavs."):
        if key.startswith(pre):
            return key[len(pre):]
    return key


def config_diffs(ref: Path, other: Path):
    """Field names that differ, minus the ones expected to differ."""
    a, b = load_config(ref), load_config(other)
    if not a or not b:
        return ["(config unreadable)"]
    bad = []
    for k in sorted(set(a) | set(b)):
        if _bare(k) in EXPECTED_DIFFS:
            continue
        if _canon(a.get(k)) != _canon(b.get(k)):
            bad.append(f"{_bare(k)}: {str(a.get(k))[:40]} -> {str(b.get(k))[:40]}")
    return bad


def compare(ref_seeds, arm_seeds, endpoint):
    """Paired difference arm - ref on shared seeds. Positive = arm better."""
    shared = sorted(set(ref_seeds) & set(arm_seeds))
    diffs = [arm_seeds[s][endpoint] - ref_seeds[s][endpoint] for s in shared]
    m, t, n = paired_t(diffs)
    return m, t, n, sum(1 for d in diffs if d > 0)


def fmt_row(label, ref_seeds, arm_seeds):
    cells, cleared = [], 0
    for name, _ in ENDPOINTS:
        m, t, n, better = compare(ref_seeds, arm_seeds, name)
        if n == 0:
            cells.append(f"{'--':>18}")
            continue
        if not math.isnan(t) and abs(t) > t_crit(n):
            cleared += 1
        ts = "   n/a" if math.isnan(t) else f"{t:+6.2f}"
        cells.append(f"{m:+8.4f} {ts} {better}/{n}")
    return f"{label:<26} " + "  ".join(cells) + f"   [{cleared}/4]"


def header():
    cols = "  ".join(f"{name:>18}" for name, _ in ENDPOINTS)
    line = f"{'arm':<26} {cols}   {'sig':>5}"
    return line, "-" * len(line)


def jitter(root: Path):
    """Mean |acc[r] - acc[r-1]| over the second half of training.

    The scale a single-round endpoint is measured against. Computed from the
    reference arm itself rather than assumed.
    """
    vals = []
    pat = str(root / "**" / f"{ARM}_seed*" / "pipeline_results.json")
    for p in sorted(glob.glob(pat, recursive=True)):
        a = json.loads(Path(p).read_text()).get("server_accuracies") or []
        if len(a) < 4:
            continue
        half = len(a) // 2
        vals += [abs(a[r] - a[r - 1]) for r in range(half, len(a))]
    return statistics.mean(vals) if vals else float("nan")


def phase0_gate(off, s000):
    """Verdict string + per-endpoint rows. Gates on ALL endpoints."""
    rows, verdict, worst_t = [], "PASS", 0.0
    any_shared = False
    crit = float("inf")
    for name, _ in ENDPOINTS:
        m, t, n, _b = compare(off, s000, name)
        if n == 0:
            rows.append(f"    {name:<8} no shared seeds")
            continue
        any_shared = True
        crit = min(crit, t_crit(n))
        # nan t means zero spread, i.e. bit-identical -> the strongest
        # possible pass, NOT a failure.
        ts = "n/a (identical)" if math.isnan(t) else f"t={t:+.2f}"
        rows.append(f"    {name:<8} mean {m:+.4f}   {ts}")
        if not math.isnan(t):
            worst_t = max(worst_t, abs(t))
    if not any_shared:
        verdict = "FAIL (no shared seeds -- the gate did not run)"
    elif worst_t > crit:
        verdict = f"FAIL (|t|={worst_t:.2f} > {crit:.2f} on at least one endpoint)"
    else:
        verdict = f"PASS (|t| <= {worst_t:.2f} on every endpoint, bar {crit:.2f})"
    return verdict, rows


def self_test():
    """Exercise every helper the reported numbers depend on."""
    # --- tail_mean
    assert tail_mean([0.1, 0.2, 0.3], 1) == 0.3, "n=1 must be the final round"
    assert abs(tail_mean([0.1, 0.2, 0.3], 2) - 0.25) < 1e-12
    assert abs(tail_mean([0.1, 0.2], 10) - 0.15) < 1e-12, "n > len must clamp"
    assert tail_mean([], 3) is None
    assert tail_mean([0.4], 5) == 0.4

    # --- paired_t. [1,2,3] has mean 2 and SAMPLE stdev exactly 1, so
    # t = 2/(1/sqrt(3)) = 2*sqrt(3). (An earlier version asserted t=4 for
    # [1,2,2,3] claiming stdev 1; it is sqrt(2/3). The self-test caught it.)
    m, t, n = paired_t([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-12 and n == 3
    assert abs(t - 2.0 * math.sqrt(3.0)) < 1e-12, f"expected 2*sqrt(3), got {t}"
    _m, t0, _n = paired_t([0.0] * 5)
    assert math.isnan(t0), "zero spread must be nan, not inf"
    _m, t1, n1 = paired_t([0.5])
    assert math.isnan(t1) and n1 == 1
    assert paired_t([])[2] == 0

    # --- compare: sign convention and seed intersection
    ref = {s: {"final": 0.30} for s in "12345"}
    arm = {s: {"final": 0.31 + 0.001 * i} for i, s in enumerate("12345")}
    m, t, n, better = compare(ref, arm, "final")
    assert m > 0 and t > 0 and better == 5 == n
    assert compare(ref, {"9": {"final": 0.9}}, "final")[2] == 0

    # --- phase0_gate: bit-identical arms must PASS, not fail.
    eps = {n: 1 for n, _ in ENDPOINTS}
    same = {s: {n: 0.3 for n, _ in ENDPOINTS} for s in "12345"}
    v, _r = phase0_gate(same, {s: dict(d) for s, d in same.items()})
    assert v.startswith("PASS"), f"identical arms must PASS, got {v!r}"
    # A real systematic offset must FAIL.
    off2 = {s: {n: 0.3 + 0.02 + 0.0001 * i for n, _ in ENDPOINTS}
            for i, s in enumerate("12345")}
    v2, _r = phase0_gate(same, off2)
    assert v2.startswith("FAIL"), f"a +0.02 offset must FAIL, got {v2!r}"
    # No shared seeds must FAIL, not pass silently.
    v3, _r = phase0_gate(same, {"99": {n: 0.3 for n, _ in ENDPOINTS}})
    assert v3.startswith("FAIL") and "no shared seeds" in v3, v3
    del eps

    # --- _canon: content, not repr order
    assert _canon("frozenset({'a', 'b'})") == _canon("frozenset({'b', 'a'})")
    assert _canon("frozenset({'a', 'b'})") != _canon("frozenset({'a', 'z'})")
    # Numbers must NOT collide with their string forms (the split(',')
    # version stripped quotes and made these equal).
    assert _canon("frozenset({1, 2})") != _canon("frozenset({'1', '2'})"), \
        "numeric and string set members must stay distinct"
    # A set of tuples must survive intact rather than shredding.
    assert _canon("frozenset({('a', 1)})") == frozenset({("a", 1)})
    assert _canon("frozenset({('a', 1)})") != _canon("frozenset({('a', 2)})")
    # Both empty-set spellings, including the one Python actually emits.
    assert _canon("frozenset()") == frozenset()
    assert _canon(repr(frozenset())) == frozenset()
    # Unparseable text falls through as text (safe: reports a difference).
    assert _canon("frozenset({<object at 0x1>})") == "frozenset({<object at 0x1>})"
    assert _canon("plain") == "plain" and _canon(0.5) == 0.5 and _canon(None) is None

    # --- _bare must strip only the known prefix, not split on the last dot
    assert _bare("exp.num_rounds") == "num_rounds"
    assert _bare("tavs.trust_weight_strength") == "trust_weight_strength"
    assert _bare("tavs.agg.weight") == "agg.weight", \
        "a dotted tavs key must not be reduced to its final component, or it " \
        "could be wrongly exempted from the one-variable check"

    # --- t_crit must FOLLOW the sample size. Hardcoding the n=5 value was a
    # real bug: once an arm gains seeds, the bar must fall, or a genuine
    # effect at t=2.4 with n=10 is reported as not significant.
    assert abs(t_crit(5) - 2.776) < 1e-9, "n=5 -> 4 d.o.f. -> 2.776"
    assert abs(t_crit(10) - 2.262) < 1e-9, "n=10 -> 9 d.o.f. -> 2.262"
    assert t_crit(10) < t_crit(5), "more seeds must LOWER the bar"
    assert t_crit(1) == float("inf") and t_crit(0) == float("inf"), \
        "no t exists below n=2"
    # An untabulated d.o.f. must round to the next LOWER one, giving a
    # LARGER threshold -- conservative, so it cannot manufacture significance.
    assert t_crit(23) == t_crit(21) == _T_05[20], \
        "untabulated d.o.f. must use the next lower tabulated entry"
    assert t_crit(23) > t_crit(26), "the bar must still fall as n grows"

    print("✓ self-test passed: tail_mean, paired_t (known value / zero spread "
          "/ n<2), compare sign + seed intersection, phase0 gate (identical "
          "PASSes, offset FAILs, no-seeds FAILs), _canon (order / numeric vs "
          "string / tuples / empty / unparseable), _bare prefix stripping")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.self_test:
        return 0 if self_test() else 1

    b = args.base
    OFF = b / "sweep_a10_notwagg"
    FULL = b / "sweep_a10_twagg"
    S000 = b / "dose_a10_str000"
    DOSE = [("strength 0.00 (self-check)", S000),
            ("strength 0.25", b / "dose_a10_str025"),
            ("strength 0.50", b / "dose_a10_str050"),
            ("strength 1.00", FULL)]
    SCHED = [("start round 6", b / "sched_a10_start06"),
             ("start round 15", b / "sched_a10_start15"),
             ("start round 1 (= full)", FULL)]

    missing = [str(p) for _l, p in DOSE + SCHED if not p.exists()]
    if not OFF.exists():
        missing.append(str(OFF))
    if missing:
        print("MISSING run directories:")
        for m in missing:
            print("  ", m)
        return 1

    off = load_seeds(OFF)
    print(f"Dirichlet(1.0), CIFAR-10N worst, 50 clients, 40 rounds, "
          f"{len(off)} seeds.")
    print("All numbers are PAIRED differences vs weighting-OFF "
          f"({OFF}).")
    n_off = len(off)
    print(f"Positive = better than not weighting at all. The significance bar "
          f"follows the\nsample size: |t|>{t_crit(n_off):.2f} is p<0.05 at "
          f"n={n_off}.")
    print("[k/4] = how many of the four endpoints clear that threshold.\n")

    print("ONE-VARIABLE CHECK (config fields differing from the OFF reference,")
    print("excluding the knobs under test):")
    confounded = False
    for label, p in DOSE + SCHED:
        d = config_diffs(OFF, p)
        if d:
            confounded = True
            print(f"  {label:<26} CONFOUNDED: {'; '.join(d)}")
        else:
            print(f"  {label:<26} clean")
    print()

    verdict, rows = phase0_gate(off, load_seeds(S000))
    print("PHASE 0 GATE: strength=0.0 must reproduce weighting-off, on EVERY "
          "endpoint")
    for r in rows:
        print(r)
    print(f"  => {verdict}")
    if verdict.startswith("FAIL"):
        print("     Something other than the aggregation weight differs. Do "
              "not interpret\n     the tables below.")
    print()

    j = jitter(OFF)
    print(f"SCALE: the weighting-off arm's own round-to-round accuracy swing "
          f"averages {j:.4f}")
    print("       over the second half of training. That is larger than any "
          "effect below, so")
    print("       the 'final' column is the least reliable of the four. The "
          "paired t already")
    print("       accounts for this; the tail averages are the ones to quote.\n")

    hdr, rule = header()
    print("DOSE SWEEP (does the harm grow with how hard the weighting pulls?)")
    print(hdr); print(rule)
    for label, p in DOSE:
        print(fmt_row(label, off, load_seeds(p)))
    print()

    print("SCHEDULE (does switching the weighting on late avoid the harm?)")
    print(hdr); print(rule)
    for label, p in SCHED:
        print(fmt_row(label, off, load_seeds(p)))
    print()

    # The schedule arms must also be compared to EACH OTHER. Against the OFF
    # baseline they all look similar, but "similar" is not "indistinguishable"
    # until it is tested directly -- and dividing an untested numerator by
    # different round counts to get a per-round harm is exactly how a null
    # result becomes a false trend.
    print("SCHEDULE, head to head (is delaying actually different from not "
          "delaying?)")
    print(hdr); print(rule)
    full = load_seeds(FULL)
    for label, p in SCHED[:2]:
        print(fmt_row(f"{label} vs start 1", full, load_seeds(p)))
    print(fmt_row("start 15 vs start 6", load_seeds(b / "sched_a10_start06"),
                  load_seeds(b / "sched_a10_start15")))
    print()

    print("Absolute accuracies at the last10 endpoint (mean over seeds):")
    for label, p in [("weighting OFF", OFF)] + DOSE[1:] + SCHED[:2]:
        sd = load_seeds(p)
        if not sd:
            continue
        vals = [v["last10"] for v in sd.values()]
        print(f"  {label:<26} {statistics.mean(vals):.4f}  (n={len(vals)})")

    if confounded:
        print("\nAt least one arm is CONFOUNDED above -- read its row as "
              "suggestive only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
