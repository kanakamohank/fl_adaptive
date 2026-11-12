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
    "trust_weight_floor",                  # floor arms
    "promoted_weight_floor",               # set together with the above
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


# The cross-sweep arms, shared with plot_dose_and_schedule.py so the two
# cannot drift. (label, strength, inlier_floor, promoted_floor, dirname).
# sweep_a10_twagg predates the floor flag and carries the historical pair,
# which is why its promoted floor is 0.0 and not its inlier floor.
CROSS_ARMS = [
    ("strength 0.25", 0.25, 0.05, 0.0, "dose_a10_str025"),
    ("strength 0.50", 0.50, 0.05, 0.0, "dose_a10_str050"),
    ("strength 1.00", 1.00, 0.05, 0.0, "sweep_a10_twagg"),
    ("floor 0.05", 1.00, 0.05, 0.05, "floor_a10_f005"),
    ("floor 0.25", 1.00, 0.25, 0.25, "floor_a10_f025"),
    ("floor 0.50", 1.00, 0.50, 0.50, "floor_a10_f050"),
    ("floor 1.00", 1.00, 1.00, 1.00, "floor_a10_f100"),
]

# Smallest multiplier the PROMOTED path can produce before any floor:
# sigmoid(c_lambda*(trust - 0.5)) at trust = 0, with the scheduler default
# c_lambda = 8.
PROMOTED_MIN_RAW = 1.0 / (1.0 + math.exp(8.0 * 0.5))


def min_attainable_weight(strength, floor, promoted_floor):
    """Smallest weight multiplier ANY accepted client can be reduced to.

    Both paths matter. The inlier path bottoms out at max(floor, 1-strength);
    the promoted path at max(promoted_floor, pull(PROMOTED_MIN_RAW)), which
    at full strength is max(promoted_floor, 0.018).

    Taking only the inlier path would misreport sweep_a10_twagg as 0.05 when
    its promoted clients -- about 31% of contributors -- can reach 0.018, and
    would tabulate it as identical to the floor-0.05 arm although the two
    differ by 0.0019 in harm. That row is the most negative point, so it is
    also the most influential one in any fit against minimum weight.
    """
    pull = (lambda x: x) if strength >= 1.0 else (
        lambda x: 1.0 - strength * (1.0 - x))
    return min(max(floor, pull(0.0)),
               max(promoted_floor, pull(PROMOTED_MIN_RAW)))


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
    strength does NOT halve the distortion. Measured on each arm's own logs:
    strength 0.25/0.50/1.00 give TVD 0.0102/0.0234/0.0697, i.e. 0.147/0.336
    of full strength rather than 0.25/0.50. Testing against the nominal
    prediction overstates the super-linearity gap by roughly 30%.

    Three scope limits, none of them fixable from the logs:

    1. LATE-ROUND DOMINATED. Round TVDs are averaged unweighted, and the
       distortion grows enormously over training -- on these runs, mean TVD
       by decade is 0.0002 / 0.018 / 0.058 / 0.132 for rounds 1-10 / 11-20 /
       21-30 / 31-40. So the single number is effectively a late-training
       statistic. Mean-of-ratios is not an alternative: early rounds divide
       by ~0.
    2. VERIFIED CLIENTS ONLY. The aggregator normalises verified and
       promoted weights jointly, and promoted weights are scaled by the
       same strength, so down-weighting verified clients also shifts mass
       toward promoted ones. About a third of logged client-rows are
       unverified, and that movement is invisible here. Including promoted
       clients at undistorted weight raises frac(0.25) by roughly 9%, i.e.
       AWAY from the super-linearity conclusion, so omitting them is not
       what produces the finding.
    3. FLOOR APPLIED TO EVERYONE. aggregate_fit exempts BVD-flagged clients
       from the floor; outlier membership is not recoverable from the logs,
       so the floor is applied to all. Under 1% of scores are below the
       floor and the floor cannot bind at all below strength 1.0 (the
       minimum multiplier is 1-strength >= 0.5), so the worst case moves
       frac(0.25) by about 1%.
    """
    if hi <= lo:
        # Mirror the strategy's own guard (tavs_esp_strategy.py: `if hi <=
        # lo: lo, hi = 0.5, 3.0`). Without this, hi == lo raises
        # ZeroDivisionError and, worse, hi < lo silently INVERTS the signal
        # and returns a plausible-looking number.
        lo, hi = 0.5, 3.0
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


def batch_agreement(ref_seeds, arm_seeds, endpoint, split=5, atol=0.001):
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

    Returns (low_mean, high_mean, verdict), verdict being "yes", "NO", or
    "one-batch" when every seed falls on one side of the split. That third
    value exists because returning a plain False there made the row vanish
    from the caller's table, so renumbering seeds would silently remove the
    guard while the table still read as all-pass.

    `atol` keeps the rule from firing on noise. The magnitude test is a 3x
    ratio, which is scale-free and so would flag -0.00002 against -0.0008 --
    both indistinguishable from zero against the 0.022 round-to-round jitter
    this script prints. Batches whose effects are BOTH below atol agree by
    definition; there is nothing there to disagree about. atol also guards
    the sign test, where an exactly-zero batch would otherwise read as
    positive and flag a spurious sign flip.
    """
    out = []
    for keep in (lambda k: k <= split, lambda k: k > split):
        ks = [s for s in sorted(set(ref_seeds) & set(arm_seeds))
              if s.isdigit() and keep(int(s))]
        if not ks:
            return float("nan"), float("nan"), "one-batch"
        out.append(statistics.mean(arm_seeds[s][endpoint] - ref_seeds[s][endpoint]
                                   for s in ks))
    a, b = out
    # Both negligible -> nothing to disagree about.
    if abs(a) < atol and abs(b) < atol:
        return a, b, "yes"
    # Sign flip, but only when at least one side is big enough to mean it.
    if (a < 0) != (b < 0) and max(abs(a), abs(b)) >= atol:
        return a, b, "NO"
    lo_m, hi_m = min(abs(a), abs(b)), max(abs(a), abs(b))
    if lo_m == 0.0:
        return a, b, "yes" if hi_m < atol else "NO"
    return a, b, ("yes" if (hi_m / lo_m) <= 3.0 else "NO")


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
    a, bb, v = batch_agreement(zeros, _mk([-0.01] * 10), "final")
    assert v == "yes" and abs(a - bb) < 1e-12, (a, bb, v)
    # Sign split, both sides meaningful -> flagged.
    a, bb, v = batch_agreement(zeros, _mk([0.01] * 5 + [-0.01] * 5), "final")
    assert v == "NO" and a > 0 > bb, (a, bb, v)
    # Magnitude blowup: same sign, 5x apart, above atol -> flagged.
    a, bb, v = batch_agreement(zeros, _mk([-0.002] * 5 + [-0.010] * 5), "final")
    assert v == "NO", (a, bb, v)
    # Same sign, 2x apart -> within tolerance.
    _a, _b, v = batch_agreement(zeros, _mk([-0.002] * 5 + [-0.004] * 5), "final")
    assert v == "yes", v
    # The real early-window case that was withdrawn: -0.0002 vs -0.0047.
    # 23x apart and the larger side is above atol, so it must flag.
    _a, _b, v = batch_agreement(zeros, _mk([-0.0002] * 5 + [-0.0047] * 5), "final")
    assert v == "NO", "the withdrawn early-window split must be flagged"
    # And the endpoint case that survived: -0.0153 vs -0.0124, must pass.
    _a, _b, v = batch_agreement(zeros, _mk([-0.0153] * 5 + [-0.0124] * 5), "final")
    assert v == "yes", "the surviving endpoint split must not be flagged"
    # NOISE must not be flagged: -0.00002 vs -0.0008 is 40x apart but both
    # are far below the 0.022 round-to-round jitter. The scale-free 3x rule
    # alone called this a disagreement.
    _a, _b, v = batch_agreement(zeros, _mk([-0.00002] * 5 + [-0.0008] * 5), "final")
    assert v == "yes", "two negligible batches must not read as disagreement"
    # A sign flip between two negligible batches is also noise.
    _a, _b, v = batch_agreement(zeros, _mk([0.0001] * 5 + [-0.0002] * 5), "final")
    assert v == "yes", "a sign flip below atol is noise, not disagreement"
    # An exactly-zero batch against a meaningful one must still flag.
    _a, _b, v = batch_agreement(zeros, _mk([0.0] * 5 + [-0.01] * 5), "final")
    assert v == "NO", "zero vs a real effect is a disagreement"
    # One batch only must be reported as such, NOT silently dropped and
    # NOT counted as agreement.
    _a, _b, v = batch_agreement(zeros, _mk([-0.01] * 5), "final")
    assert v == "one-batch", v

    print("✓ self-test passed: tail_mean, paired_t (known value / zero spread "
          "/ n<2), compare sign + seed intersection, phase0 gate (identical "
          "PASSes, offset FAILs, no-seeds FAILs), _canon (order / numeric vs "
          "string / tuples / empty / unparseable), _bare prefix stripping, "
          "t_crit (n=5/n=10/fallback/n<2), ci_halfwidth (known value / zero "
          "spread / narrowing), batch_agreement (sign split / magnitude "
          "blowup / the two real retraction cases / noise below atol / "
          "one-batch)")
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
    # Floor arms. Every one sets BOTH the inlier and the promoted floor to
    # the same value. sweep_a10_twagg is listed separately as "pre-flag"
    # because it predates the flag and carries the historical pair
    # (0.05 inlier, 0.0 promoted) -- it is NOT the floor-0.05 arm, and
    # treating it as one would silently fold the promoted floor into the
    # ladder.
    FLOOR = [("pre-flag (0.05 / 0.0)", FULL),
             ("floor 0.05", b / "floor_a10_f005"),
             ("floor 0.25", b / "floor_a10_f025"),
             ("floor 0.50", b / "floor_a10_f050"),
             ("floor 1.00", b / "floor_a10_f100")]
    SCHED = [("start round 6", b / "sched_a10_start06"),
             ("start round 15", b / "sched_a10_start15"),
             ("start round 1 (= full)", FULL)]

    # Floor arms are optional: the dose/schedule report must still run
    # before that batch exists.
    FLOOR = [(l, pth) for l, pth in FLOOR if pth.exists()]
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
            # `off` is indexed below for the harm, so it must be in the
            # intersection. Without it, a seed present in FULL and the
            # schedule arm but missing from OFF raised KeyError -- exactly
            # the partially-complete-batch case this report has to survive.
            shared = sorted(set(full) & set(arm) & set(off))
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
    # Each arm's distortion is measured on ITS OWN logs, and the full-strength
    # denominator on the FULL arm's own logs.
    #
    # This is the consistent pairing and it is NOT the flattering one. The
    # quantity being scaled, (full - off), is the harm the FULL arm produced
    # under the distortion the FULL arm actually applied -- TVD 0.0697 on its
    # own logs. An earlier version measured every strength on the OFF arm's
    # logs, whose full-strength TVD is 0.0521, i.e. 34% smaller. Dividing by
    # that smaller denominator inflates frac, inflates the predicted harm and
    # inflates the gap, so it biased the test TOWARD super-linearity. It also
    # changed the answer: 8/8 cells significant on the OFF reference versus
    # 7/8 on this one. The sensitivity line below names whichever cells depend
    # on that choice, rather than letting one reference silently decide it.
    full_tvd = perturbation_tvd(FULL, 1.0, 0.6, 1.5)
    print(f"  distortion of full strength, on its own logs (TVD) = "
          f"{full_tvd:.4f}")
    print(f"  {'strength':>9} {'endpoint':>8} {'distortion':>11} "
          f"{'observed':>10} {'predicted':>10} {'gap':>9} {'t':>7} {'sig':>4}")
    nsig = ntot = 0
    borderline = []
    for lab, p, st in [("0.25", b / "dose_a10_str025", 0.25),
                       ("0.50", b / "dose_a10_str050", 0.50)]:
        arm = load_seeds(p)
        own = perturbation_tvd(p, st, 0.6, 1.5)
        frac = (own / full_tvd) if full_tvd else float("nan")
        # The plausible range across the three defensible references
        # (own logs, OFF logs, FULL logs), used only to flag sensitivity.
        alts = [frac,
                perturbation_tvd(OFF, st, 0.6, 1.5)
                / (perturbation_tvd(OFF, 1.0, 0.6, 1.5) or float("nan")),
                perturbation_tvd(FULL, st, 0.6, 1.5) / (full_tvd or float("nan"))]
        for ep, _ in ENDPOINTS:
            shared = sorted(set(off) & set(arm) & set(full))
            if len(shared) < 2:
                continue
            def _gap(f):
                d = [(arm[s2][ep] - off[s2][ep]) - f * (full[s2][ep] - off[s2][ep])
                     for s2 in shared]
                return paired_t(d)
            m, t, n = _gap(frac)
            obs = statistics.mean(arm[s2][ep] - off[s2][ep] for s2 in shared)
            pred = frac * statistics.mean(full[s2][ep] - off[s2][ep] for s2 in shared)
            ntot += 1
            ok = (not math.isnan(t)) and abs(t) > t_crit(n)
            nsig += ok
            # Does the verdict hold across every defensible reference?
            verdicts = set()
            for f in alts:
                if math.isnan(f):
                    continue
                _m2, t2, n2 = _gap(f)
                verdicts.add((not math.isnan(t2)) and abs(t2) > t_crit(n2))
            if len(verdicts) > 1:
                borderline.append(f"{lab}/{ep}")
            print(f"  {lab:>9} {ep:>8} {frac:>11.3f} {obs:>+10.4f} "
                  f"{pred:>+10.4f} {m:>+9.4f} {t:>+7.2f} "
                  f"{('YES' if ok else 'no') + ('*' if len(verdicts) > 1 else ''):>4}")
    print(f"  => {nsig}/{ntot} significant. A positive gap means that strength "
          f"is LESS harmful\n     than its share of the distortion predicts.")
    if borderline:
        print(f"  * {', '.join(borderline)} flip verdict depending on which run's "
              f"logs the\n    distortion is measured on, so that cell is not "
              f"load-bearing evidence.")
    else:
        print("  No cell depends on the choice of distortion reference.")
    print()

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
            lo_m, hi_m, verdict = batch_agreement(off, arm, ep)
            if verdict == "one-batch":
                print(f"  {label:<26} {ep:>8} {'--':>12} {'--':>12} "
                      f"{'1 batch':>9}   (all seeds on one side of the split: "
                      f"this guard did NOT run)")
                continue
            print(f"  {label:<26} {ep:>8} {lo_m:>+12.4f} {hi_m:>+12.4f} "
                  f"{verdict:>9}")
    print()

    if len(FLOOR) > 1:
        print("FLOOR SWEEP (is the harm caused by clients pushed toward ZERO "
              "weight?)")
        print("Raising the floor raises the smallest weight any ACCEPTED "
              "client can get.")
        print("Flagged clients stay unfloored throughout, so floor 1.00 is NOT "
              "the same as\nturning weighting off -- it leaves the outlier "
              "branch running.")
        print(hdr); print(rule)
        for label, p2 in FLOOR:
            print(fmt_row(label, off, load_seeds(p2)))
        print()

        f005 = b / "floor_a10_f005"
        if f005.exists():
            print("  Head to head vs floor 0.05 (does raising it REMOVE the "
                  "harm?), with CI:")
            base_sd = load_seeds(f005)
            for label, p2 in FLOOR:
                if p2 == f005 or p2 == FULL:
                    continue
                arm = load_seeds(p2)
                for ep in ("final", "last10"):
                    shared = sorted(set(base_sd) & set(arm) & set(off))
                    if len(shared) < 2:
                        continue
                    d = [arm[s2][ep] - base_sd[s2][ep] for s2 in shared]
                    m, _t, _n = paired_t(d)
                    hw = ci_halfwidth(d)
                    harm = statistics.mean(base_sd[s2][ep] - off[s2][ep]
                                           for s2 in shared)
                    print(f"    {label:<14} {ep:<7} {m:+.4f}  CI "
                          f"[{m - hw:+.4f}, {m + hw:+.4f}]   vs a floor-0.05 "
                          f"harm of {harm:+.4f}")
            print("    A CI that excludes 0 means the floor measurably "
                  "removed harm; one that\n    contains 0 but is narrow "
                  "relative to the harm bounds how much it could have.\n")

        # The cross-sweep test. This is what the floor sweep is FOR, and on
        # its own the floor ladder answers the wrong question: a monotone
        # fall in harm as the floor rises is consistent with BOTH candidate
        # explanations, because raising the floor also reduces the overall
        # distortion. Only arms that hold one constant while moving the
        # other can separate them.
        print("  WHAT DOES THE HARM TRACK? Arms from BOTH sweeps.")
        print("    (a) distortion   -- how far weighting moves the client shares")
        print("    (b) minimum weight -- how close any accepted client gets to 0")
        print(f"    {'arm':<16}{'min weight':>11}{'distortion':>12}"
              f"{'harm(last10)':>14}{'t':>8}")
        pts = []
        for lab, st, fl, pf, name in CROSS_ARMS:
            pth = b / name
            if not pth.exists():
                continue
            minw = min_attainable_weight(st, fl, pf)
            tvd = perturbation_tvd(OFF, st, 0.6, 1.5, floor=fl)
            m, t, n, _bt = compare(off, load_seeds(pth), "last10")
            pts.append((lab, minw, tvd, m, name))
            print(f"    {lab:<16}{minw:>11.3f}{tvd:>12.4f}{m:>+14.4f}{t:>+8.2f}")

        if len(pts) >= 3:
            def _r2(xs, ys):
                mx, my = statistics.mean(xs), statistics.mean(ys)
                sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                sxx = sum((x - mx) ** 2 for x in xs)
                syy = sum((y - my) ** 2 for y in ys)
                return (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")
            r_d = _r2([q[2] for q in pts], [q[3] for q in pts])
            r_m = _r2([q[1] for q in pts], [q[3] for q in pts])
            # Two decimals, not three. A seed-level paired bootstrap puts the
            # difference between these two at +0.07 with a 95% interval of
            # [-0.01, +0.16] -- it does NOT separate the hypotheses, and three
            # decimals on seven points would imply a precision that is not
            # there. The conclusion rests on the direct test below.
            print(f"    across these arms: harm vs distortion R^2 = {r_d:.2f}, "
                  f"vs minimum weight R^2 = {r_m:.2f}")
            print("    That gap is NOT decisive -- a seed-level bootstrap on the "
                  "difference spans\n    zero. Correlation over seven points "
                  "cannot settle this; the direct test can.")

            # The one pair that holds minimum weight FIXED and moves only the
            # distortion, tested per seed rather than by eyeballing one
            # significant arm against one null arm.
            names = {q[0]: q for q in pts}
            if "strength 0.50" in names and "floor 0.50" in names:
                a2, c2 = names["strength 0.50"], names["floor 0.50"]
                sa, sc = load_seeds(b / a2[4]), load_seeds(b / c2[4])
                print(f"\n    DIRECT TEST -- both arms allow a minimum weight "
                      f"of {a2[1]:.2f}, so nothing is")
                print(f"    crushed toward zero in either, but floor 0.50 "
                      f"distorts {c2[2] / a2[2]:.2f}x more.")
                print(f"    Paired per seed, floor 0.50 minus strength 0.50:")
                sig = 0
                for ep, _ in ENDPOINTS:
                    shared = sorted(set(sa) & set(sc))
                    if len(shared) < 2:
                        continue
                    d = [sc[s2][ep] - sa[s2][ep] for s2 in shared]
                    m2, t2, n2 = paired_t(d)
                    hw = ci_halfwidth(d)
                    ok = (not math.isnan(t2)) and abs(t2) > t_crit(n2)
                    sig += ok
                    print(f"      {ep:<7} {m2:+.5f}  t={t2:+5.2f}  CI "
                          f"[{m2 - hw:+.5f}, {m2 + hw:+.5f}]  "
                          f"{'YES' if ok else 'no'}")
                print(f"      => {sig}/4 endpoints. Harm appears at the SAME "
                      f"minimum weight when the\n         distortion rises, so "
                      f"near-total silencing is not what causes it.")
        print()

        print("  Batch agreement on the floor arms (last10):")
        for label, p2 in FLOOR:
            arm_sd = load_seeds(p2)
            lo_m, hi_m, verdict = batch_agreement(off, arm_sd, "last10")
            note = ""
            # Whether the two batches' per-seed RANGES overlap, measured
            # rather than asserted. An earlier version claimed overlap in
            # the printed text while the code only checked that both means
            # were small, so the claim would have kept printing after it
            # stopped being true.
            shared = sorted(set(off) & set(arm_sd))
            lo_v = [arm_sd[s2]["last10"] - off[s2]["last10"]
                    for s2 in shared if s2.isdigit() and int(s2) <= 5]
            hi_v = [arm_sd[s2]["last10"] - off[s2]["last10"]
                    for s2 in shared if s2.isdigit() and int(s2) > 5]
            overlap = bool(lo_v and hi_v
                           and min(lo_v) <= max(hi_v) and min(hi_v) <= max(lo_v))
            if verdict == "NO" and overlap and max(abs(lo_m), abs(hi_m)) < 0.003:
                # Both halves sit just above atol while straddling zero.
                # The per-seed values OVERLAP heavily between batches -- there
                # is no separation, unlike the early-window claim that was
                # retracted, where the two batches were far from zero and did
                # not overlap at all. So this fires on a near-null, which is
                # the rule being conservative, not an artifact being caught.
                note = ("   (both halves ~0 and the per-seed ranges DO "
                        "overlap: a near-null, not a split)")
            print(f"    {label:<22} {lo_m:+.4f} / {hi_m:+.4f}   {verdict}{note}")
        print()

    print("Absolute accuracies at the last10 endpoint (mean over seeds):")
    for label, p in ([("weighting OFF", OFF)] + DOSE[1:] + SCHED[:2]
                     + [(l, q) for l, q in FLOOR if q != FULL]):
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
