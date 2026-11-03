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
    ns = {compare(ref_seeds, arm_seeds, nm)[2] for nm, _ in ENDPOINTS}
    n_here = max(ns) if ns else 0
    bar = t_crit(n_here)
    bars = "  n=0" if n_here == 0 else f"n={n_here:<2} bar {bar:.2f}"
    return (f"{label:<26} " + "  ".join(cells)
            + f"   [{cleared}/4]  {bars}")


def header():
    cols = "  ".join(f"{name:>18}" for name, _ in ENDPOINTS)
    line = f"{'arm':<26} {cols}   {'sig':>5}  {'sample':<14}"
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


def ci_halfwidth(diffs):
    """Half-width of the 95% CI on a paired mean: t_crit * standard error.

    Reported alongside every head-to-head because "not significant" on its
    own is not a result. A null with a half-width of 0.0078 against a harm of
    0.0139 cannot exclude the arm recovering half the harm; a half-width of
    0.0017 can. Saying only "no difference" hides which of those it is.
    """
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return 0.0
    return t_crit(n) * sd / math.sqrt(n)


def perturbation_tvd(root: Path, strength, lo, hi, floor=0.05):
    """How far weighting at `strength` actually moves the client shares.

    Total variation distance between the weighted and the unweighted share
    vectors, averaged over rounds: 0 means identical to not weighting, 1
    means no overlap. Computed from the scores a real run logged.

    This exists because NOMINAL strength is the wrong yardstick for a
    super-linearity test. The weight is max(floor, 1 - strength*(1 - score)),
    and scores under a skewed partition cluster near 1, so halving the
    strength does NOT halve the distortion. Measured here: strength
    0.25/0.50/1.00 produce TVD 0.0123/0.0270/0.0697, i.e. 0.18/0.39/1.00 of
    full strength, not 0.25/0.50/1.00. Testing against the nominal
    prediction overstates the super-linearity gap by roughly 30%.
    """
    vals = []
    pat = str(root / "**" / f"{ARM}_seed*" / "pipeline_results.json")
    for p in sorted(glob.glob(pat, recursive=True)):
        fts = json.loads(Path(p).read_text()).get("final_trust_state") or {}
        for _r, ent in (fts.get("oracle_signal_history") or {}).items():
            rows = []
            for _c, e in ent.items():
                raw = e.get("pretrain_loss_var")
                n = e.get("num_examples") or 0
                if not e.get("is_verified") or raw is None or n <= 0:
                    continue
                s = 1.0 - max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))
                rows.append((s, float(n)))
            if len(rows) < 3:
                continue
            w = [max(floor, 1.0 - strength * (1.0 - s)) * n for s, n in rows]
            u = [n for _s, n in rows]
            sw, su = sum(w), sum(u)
            if sw <= 0 or su <= 0:
                continue
            vals.append(0.5 * sum(abs(a / sw - c / su) for a, c in zip(w, u)))
    return statistics.mean(vals) if vals else float("nan")


def batch_agreement(ref_seeds, arm_seeds, endpoint, split=5):
    """Paired effect computed separately for low and high seed numbers.

    Seeds added in a later session can differ systematically from the
    originals even when the config is identical: this pipeline re-seeds
    torch/numpy/random per run, yet its own source notes that Ray and
    DataLoader ordering still leave real movement, and our Phase-0 control
    shows a 0.0085 residual between two configurations that are
    mathematically identical.

    This caught a false finding. An "early-training harm" of -0.0024
    (t=-2.70, apparently significant at n=10) separated PERFECTLY by batch:
    seeds 1-5 gave -0.0002 and seeds 6-10 gave -0.0047. Perfect separation
    of 5 and 5 has probability 2/C(10,5) = 0.008 under one distribution, so
    the quantity was a batch artifact, not an effect. The endpoint
    comparisons passed the same check (final -0.0153 vs -0.0124, last10
    -0.0055 vs -0.0082), which is why those survived and the early-window
    claim was withdrawn.

    Returns (low_mean, high_mean, agree). `agree` is False when the two
    batches disagree in SIGN, or when one is more than 3x the other in
    magnitude -- a deliberately crude rule, because its job is to force a
    look, not to be a test.
    """
    out = []
    for keep in (lambda k: k <= split, lambda k: k > split):
        ks = [s for s in sorted(set(ref_seeds) & set(arm_seeds))
              if s.isdigit() and keep(int(s))]
        if not ks:
            return float("nan"), float("nan"), False
        out.append(statistics.mean(arm_seeds[s][endpoint] - ref_seeds[s][endpoint]
                                   for s in ks))
    a, b = out
    if (a < 0) != (b < 0):
        return a, b, False
    lo_m, hi_m = min(abs(a), abs(b)), max(abs(a), abs(b))
    if lo_m == 0.0:
        return a, b, hi_m == 0.0
    return a, b, (hi_m / lo_m) <= 3.0


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

    # --- ci_halfwidth: known value. [1,2,3] has sample stdev exactly 1 and
    # n=3, so SE = 1/sqrt(3). n=3 means 2 degrees of freedom, whose critical
    # value is 4.303 -- NOT 3.182, which is the dof=3 (n=4) entry. An
    # earlier version of this assertion used 3.182 and the self-test caught
    # it, which is the off-by-one this check exists to prevent.
    hw = ci_halfwidth([1.0, 2.0, 3.0])
    assert abs(hw - 4.303 / math.sqrt(3.0)) < 1e-9, hw
    assert ci_halfwidth([2.0] * 6) == 0.0, "zero spread -> zero width"
    assert math.isnan(ci_halfwidth([1.0])), "n<2 has no CI"
    # More seeds must NARROW the interval for the same spread.
    assert ci_halfwidth([1.0, 2.0, 3.0] * 4) < ci_halfwidth([1.0, 2.0, 3.0]), \
        "a larger sample must give a narrower interval"

    # --- batch_agreement. It must flag a SIGN split and a MAGNITUDE blowup,
    # and pass genuinely consistent batches. This is the guard that caught a
    # false early-training finding, so its failure modes are pinned.
    def _mk(vals):
        return {str(i + 1): {"final": v} for i, v in enumerate(vals)}
    zeros = _mk([0.0] * 10)
    # Agreeing batches: -0.01 either side.
    a, bb, ok = batch_agreement(zeros, _mk([-0.01] * 10), "final")
    assert ok and abs(a - bb) < 1e-12, (a, bb, ok)
    # Sign split: low batch positive, high batch negative.
    a, bb, ok = batch_agreement(zeros, _mk([0.01] * 5 + [-0.01] * 5), "final")
    assert not ok and a > 0 > bb, (a, bb, ok)
    # Magnitude blowup: same sign, 5x apart -> must NOT agree.
    a, bb, ok = batch_agreement(zeros, _mk([-0.001] * 5 + [-0.005] * 5), "final")
    assert not ok, (a, bb, ok)
    # Same sign, 2x apart -> within tolerance, agrees.
    _a, _b, ok = batch_agreement(zeros, _mk([-0.002] * 5 + [-0.004] * 5), "final")
    assert ok
    # The real early-window case that was withdrawn: -0.0002 vs -0.0047,
    # same sign but 23x apart, must be flagged.
    _a, _b, ok = batch_agreement(zeros, _mk([-0.0002] * 5 + [-0.0047] * 5), "final")
    assert not ok, "the withdrawn early-window split must be flagged"
    # And the endpoint case that survived: -0.0153 vs -0.0124, must pass.
    _a, _b, ok = batch_agreement(zeros, _mk([-0.0153] * 5 + [-0.0124] * 5), "final")
    assert ok, "the surviving endpoint split must not be flagged"
    # A missing batch must report disagreement, not silently pass.
    _a, _b, ok = batch_agreement(zeros, _mk([-0.01] * 5), "final")
    assert not ok, "only one batch present must not read as agreement"

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
    print("Positive = better than not weighting at all.")
    print("[k/4] = how many of the four endpoints clear p<0.05. The bar "
          "follows each ARM's")
    print("own paired sample size and is printed per row: arms can differ in "
          "seed count")
    print("(strength 0.00 is a control and is not re-run when seeds are "
          "added), so one")
    print("global threshold would be wrong for some rows.\n")

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
    # "Not significant" is not a result on its own. The CI half-width says
    # how large a real difference could still be hiding, which is what turns
    # a null into a bounded statement.
    print("  What those nulls can actually EXCLUDE (95% CI on the paired mean):")
    for label, p2 in SCHED[:2]:
        arm = load_seeds(p2)
        for ep in ("final", "last10"):
            shared = sorted(set(full) & set(arm))
            if len(shared) < 2:
                continue
            d = [arm[s][ep] - full[s][ep] for s in shared]
            m, _t, _n = paired_t(d)
            hw = ci_halfwidth(d)
            harm = statistics.mean(full[s][ep] - off[s][ep] for s in shared)
            frac = (hw / abs(harm)) if harm else float("nan")
            print(f"    {label:<16} {ep:<7} mean {m:+.4f}  CI [{m - hw:+.4f}, "
                  f"{m + hw:+.4f}]  half-width is {frac:.0%} of the "
                  f"{harm:+.4f} harm")
    print("    A half-width near or above the harm means the test could not have")
    print("    detected the schedule removing most of it -- underpowered, not null.\n")

    # Super-linearity, tested against the ACTUAL perturbation each strength
    # produces rather than its nominal value. See perturbation_tvd for why
    # nominal strength is the wrong yardstick.
    print("SUPER-LINEARITY (is the harm proportional to how much the weighting")
    print("actually distorts the client shares?)")
    # All strengths are evaluated on ONE common set of logged scores -- the
    # weighting-OFF arm's. Using each arm's own logs instead would divide a
    # TVD measured on one run's score distribution by a TVD measured on
    # another's, mixing the transform's distortion with run-to-run
    # differences in the scores themselves. It also matters numerically: the
    # own-logs version puts strength 0.25 at 0.147 of full distortion and
    # the common-reference version at 0.176, which moves one of the eight
    # cells across the significance line. OFF is the right reference because
    # it is the baseline every arm is compared against and its scores carry
    # no weighting feedback.
    full_tvd = perturbation_tvd(OFF, 1.0, 0.6, 1.5)
    print(f"  measured distortion (TVD vs unweighted), full strength = "
          f"{full_tvd:.4f}")
    print(f"  {'strength':>9} {'distortion':>11} {'observed':>10} "
          f"{'predicted':>10} {'gap':>9} {'t':>7} {'sig':>4}")
    nsig = ntot = 0
    for lab, p, st in [("0.25", b / "dose_a10_str025", 0.25),
                       ("0.50", b / "dose_a10_str050", 0.50)]:
        arm = load_seeds(p)
        frac = perturbation_tvd(OFF, st, 0.6, 1.5) / full_tvd if full_tvd else float("nan")
        for ep, _ in ENDPOINTS:
            shared = sorted(set(off) & set(arm) & set(full))
            if not shared:
                continue
            d = [(arm[s][ep] - off[s][ep]) - frac * (full[s][ep] - off[s][ep])
                 for s in shared]
            obs = statistics.mean(arm[s][ep] - off[s][ep] for s in shared)
            pred = frac * statistics.mean(full[s][ep] - off[s][ep] for s in shared)
            m, t, n = paired_t(d)
            ntot += 1
            ok = (not math.isnan(t)) and abs(t) > t_crit(n)
            nsig += ok
            print(f"  {lab:>9} {frac:>11.3f} {obs:>+10.4f} {pred:>+10.4f} "
                  f"{m:>+9.4f} {t:>+7.2f} {'YES' if ok else 'no':>4}")
    print(f"  => {nsig}/{ntot} significant. A positive gap means that strength "
          f"is LESS harmful\n     than its share of the distortion predicts.\n")

    # Batch agreement. Runs added in a later session can differ
    # systematically from the originals even at an identical config, so any
    # pooled claim has to survive being split by seed batch.
    print("BATCH AGREEMENT (do the first and second halves of the seeds give the")
    print("same answer? a NO means that quantity is not poolable across batches)")
    print(f"  {'arm / quantity':<26} {'endpoint':>8} {'seeds 1-5':>12} "
          f"{'seeds 6-10':>12} {'agree':>7}")
    for label, p in [("strength 1.00", FULL),
                     ("strength 0.50", b / "dose_a10_str050"),
                     ("start round 6", b / "sched_a10_start06"),
                     ("start round 15", b / "sched_a10_start15")]:
        arm = load_seeds(p)
        for ep in ("final", "last10"):
            lo_m, hi_m, ok = batch_agreement(off, arm, ep)
            if math.isnan(lo_m):
                continue
            print(f"  {label:<26} {ep:>8} {lo_m:>+12.4f} {hi_m:>+12.4f} "
                  f"{'yes' if ok else 'NO':>7}")
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
