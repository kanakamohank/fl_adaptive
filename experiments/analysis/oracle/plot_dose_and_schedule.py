#!/usr/bin/env python3
"""Four figures for the dose / schedule / floor experiments.

Each panel answers one question and is readable on its own, because these
end up in slides and a reader will not have the console table beside them.

  fig 1  DOSE          effect vs weighting strength, with error bars
  fig 2  SCHEDULE      effect vs the round weighting switches on
  fig 3  TRAJECTORY    accuracy over rounds, weighted vs not
  fig 4  WEIGHT CURVE  the weight a client gets for a given score, per arm
                       -- this is the one that explains WHY the sweep cannot
                       separate dose from near-zero weights

Design choices that matter for honesty, not looks:

* Error bars are the 95% CI on the PAIRED mean, which is what the
  significance claim rests on. Plotting per-seed standard deviation instead
  would look larger and mean something else.
* Every point is annotated with how many of the four endpoints it clears,
  so a reader cannot take a single bar as the whole result.
* The round-to-round jitter of the baseline is drawn as a shaded band on
  fig 3. It is larger than every effect here, which is the single most
  important thing to see and the reason tail averages are quoted.
* fig 3 plots the MEAN over seeds and the per-seed spread, not one seed.

Usage:
  python3 experiments/analysis/oracle/plot_dose_and_schedule.py
  python3 experiments/analysis/oracle/plot_dose_and_schedule.py --out DIR
  python3 experiments/analysis/oracle/plot_dose_and_schedule.py --self-test
"""
import argparse
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.analysis.oracle.dose_and_schedule import (  # noqa: E402
    ENDPOINTS, ci_halfwidth, compare, load_seeds, paired_t, t_crit,
)

ARM = "tavs_skip"
QUOTE_EP = "last10"   # the endpoint quoted in the text; least jittery


def _mpl():
    import matplotlib
    matplotlib.use("Agg")       # no display in a container
    import matplotlib.pyplot as plt
    return plt


def effect(off, arm, endpoint):
    """(mean, ci_halfwidth, n_endpoints_significant) for one arm."""
    shared = sorted(set(off) & set(arm))
    if len(shared) < 2:
        return float("nan"), float("nan"), 0
    d = [arm[s][endpoint] - off[s][endpoint] for s in shared]
    m, _t, _n = paired_t(d)
    cleared = 0
    for ep, _ in ENDPOINTS:
        # compare() returns (mean, t, n, n_better) -- four values.
        _m2, t2, n2, _b2 = compare(off, arm, ep)
        if not math.isnan(t2) and abs(t2) > t_crit(n2):
            cleared += 1
    return m, ci_halfwidth(d), cleared


def series(root: Path):
    """{seed -> [accuracy per round]}"""
    out = {}
    pat = str(root / "**" / f"{ARM}_seed*" / "pipeline_results.json")
    for p in sorted(glob.glob(pat, recursive=True)):
        m = re.search(rf"{ARM}_seed(\d+)", p)
        if not m:
            continue
        a = json.loads(Path(p).read_text()).get("server_accuracies") or []
        if a:
            out[str(int(m.group(1)))] = a
    return out


def weight_curve(score, strength, floor, is_outlier=False):
    """The multiplier aggregate_fit gives a client with this score.

    Mirrors the verified_weights expression, including the short-circuit at
    full strength and the fact that a FLAGGED client is exempt from the
    floor. Kept here so fig 4 cannot drift from the strategy.
    """
    pull = score if strength >= 1.0 else 1.0 - strength * (1.0 - score)
    return pull if is_outlier else max(floor, pull)


def _annotate(ax, xs, ys, cleared, ns):
    """Label each point with endpoints-significant and its sample size.

    Offset to the RIGHT, not above: the error bar cap sits above the point
    and a centred label lands on top of it. The sample size is shown because
    arms can differ -- the strength 0.0 control is deliberately not re-run
    when seeds are added -- and a reader comparing two points with different
    n deserves to see that without opening the console table.
    """
    for x, y, c, n in zip(xs, ys, cleared, ns):
        if math.isnan(y):
            continue
        ax.annotate(f"{c}/4  n={n}", (x, y), textcoords="offset points",
                    xytext=(7, -3), ha="left", fontsize=8, color="#444444")


def fig_dose(plt, b, off, out):
    arms = [(0.0, b / "dose_a10_str000"), (0.25, b / "dose_a10_str025"),
            (0.5, b / "dose_a10_str050"), (1.0, b / "sweep_a10_twagg")]
    xs, ys, es, cs, ns = [], [], [], [], []
    for st, p in arms:
        if not p.exists():
            continue
        sd = load_seeds(p)
        m, hw, c = effect(off, sd, QUOTE_EP)
        xs.append(st); ys.append(m); es.append(hw); cs.append(c)
        ns.append(len(set(off) & set(sd)))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, lw=1.8,
                color="#1f4e79", label="measured")
    # The straight line a proportional harm would follow, anchored on the
    # full-strength point. Shows at a glance that the middle arms sit far
    # above it.
    if 1.0 in xs:
        full = ys[xs.index(1.0)]
        ax.plot([0, 1], [0, full], color="#c0504d", lw=1.4, ls=":",
                label="if harm were proportional to strength")
    _annotate(ax, xs, ys, cs, ns)
    ax.set_xlabel("trust_weight_strength  (0 = no weighting, 1 = full)")
    ax.set_ylabel(f"accuracy vs weighting OFF  ({QUOTE_EP})")
    ax.set_title("Dose: harm appears only at full strength\n"
                 "labels = endpoints significant out of 4, and seeds; "
                 "bars = 95% CI on the paired mean", fontsize=10)
    ax.set_xlim(-0.08, 1.16)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    f = out / "dose.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def fig_schedule(plt, b, off, out):
    arms = [(1, b / "sweep_a10_twagg"), (6, b / "sched_a10_start06"),
            (15, b / "sched_a10_start15")]
    xs, ys, es, cs, ns = [], [], [], [], []
    for r, p in arms:
        if not p.exists():
            continue
        sd = load_seeds(p)
        m, hw, c = effect(off, sd, QUOTE_EP)
        xs.append(r); ys.append(m); es.append(hw); cs.append(c)
        ns.append(len(set(off) & set(sd)))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.errorbar(xs, ys, yerr=es, marker="s", capsize=4, lw=1.8,
                color="#1f4e79")
    _annotate(ax, xs, ys, cs, ns)
    ax.set_xlabel("round at which trust weighting switches on  (of 40)")
    ax.set_ylabel(f"accuracy vs weighting OFF  ({QUOTE_EP})")
    ax.set_title("Schedule: delaying the switch does not help\n"
                 "FedCNI switches at 15% of training; here that changes "
                 "nothing", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 1.5, max(xs) + 3.5)
    fig.tight_layout()
    f = out / "schedule.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def fig_trajectory(plt, b, out):
    offs = series(b / "sweep_a10_notwagg")
    ons = series(b / "sweep_a10_twagg")
    shared = sorted(set(offs) & set(ons))
    if not shared:
        return None
    n = min(min(len(offs[s]) for s in shared), min(len(ons[s]) for s in shared))
    rounds = list(range(1, n + 1))
    mo = [statistics.mean(offs[s][r] for s in shared) for r in range(n)]
    mn = [statistics.mean(ons[s][r] for s in shared) for r in range(n)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(rounds, mo, color="#1f4e79", lw=1.8, label="weighting OFF")
    ax.plot(rounds, mn, color="#c0504d", lw=1.8, label="weighting ON (full)")
    # Per-seed spread, so nobody reads the mean as a tight curve.
    for s in shared:
        ax.plot(rounds, offs[s][:n], color="#1f4e79", lw=0.4, alpha=0.25)
        ax.plot(rounds, ons[s][:n], color="#c0504d", lw=0.4, alpha=0.25)
    # The jitter band: how much the baseline moves between adjacent rounds.
    half = n // 2
    jit = statistics.mean(abs(offs[s][r] - offs[s][r - 1])
                          for s in shared for r in range(half, n))
    ax.axhspan(mo[-1] - jit / 2, mo[-1] + jit / 2, color="#999999", alpha=0.18,
               label=f"round-to-round jitter of the baseline ({jit:.3f})")
    ax.set_xlabel("communication round")
    ax.set_ylabel("global model accuracy")
    ax.set_title("Trajectory: the gap is small next to the jitter\n"
                 "which is why tail averages are quoted, not the final round",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    f = out / "trajectory.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def fig_weight_curve(plt, out):
    xs = [i / 400.0 for i in range(401)]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for lab, st, fl, col in [
            ("strength 0.25", 0.25, 0.05, "#7f9db9"),
            ("strength 0.50", 0.50, 0.05, "#4472c4"),
            ("strength 1.00, floor 0.05  (current)", 1.0, 0.05, "#c0504d"),
            ("strength 1.00, floor 0.50  (proposed)", 1.0, 0.50, "#548235")]:
        ax.plot(xs, [weight_curve(x, st, fl) for x in xs], lw=1.9,
                label=lab, color=col)
    ax.axhline(0.05, color="#999999", lw=0.9, ls=":")
    ax.annotate("floor 0.05", (0.02, 0.07), fontsize=8, color="#666666")
    ax.set_xlabel("trust score  (0 = looks noisiest, 1 = looks cleanest)")
    ax.set_ylabel("weight multiplier applied to the client")
    ax.set_title("Why the dose sweep cannot separate two explanations\n"
                 "only strength 1.00 lets any accepted client approach zero",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    f = out / "weight_curve.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def self_test():
    # weight_curve must mirror the strategy exactly, including the two
    # asymmetries that caused real errors earlier in this work.
    assert weight_curve(0.0, 1.0, 0.05) == 0.05, "floor binds at full strength"
    assert weight_curve(0.0, 0.5, 0.05) == 0.5, \
        "at strength 0.5 the minimum is 1-strength, NOT the floor"
    assert weight_curve(0.0, 0.25, 0.05) == 0.75
    assert weight_curve(0.0, 1.0, 0.05, is_outlier=True) == 0.0, \
        "a flagged client is exempt from the floor and must reach zero"
    assert weight_curve(0.0, 1.0, 0.50, is_outlier=True) == 0.0, \
        "raising the floor must not lift a flagged client off zero"
    # Full strength must be the exact identity on accepted clients above the
    # floor -- the short-circuit, not 1-1*(1-s), which is not bit-identical.
    for i in range(1, 1000):
        s = i / 1000.0
        if s > 0.05:
            assert weight_curve(s, 1.0, 0.05) == s, (s, weight_curve(s, 1.0, 0.05))
    # Monotone in score, and never outside [0, 1].
    prev = -1.0
    for i in range(401):
        v = weight_curve(i / 400.0, 0.5, 0.05)
        assert 0.0 <= v <= 1.0 and v >= prev
        prev = v
    # effect() must return nan rather than raise when seeds do not overlap.
    m, hw, c = effect({"1": {e: 0.3 for e, _ in ENDPOINTS}},
                      {"9": {e: 0.3 for e, _ in ENDPOINTS}}, QUOTE_EP)
    assert math.isnan(m) and math.isnan(hw) and c == 0
    # And a real paired effect must come out with the right sign.
    off = {s: {e: 0.30 for e, _ in ENDPOINTS} for s in "12345"}
    arm = {s: {e: 0.29 for e, _ in ENDPOINTS} for s in "12345"}
    m, hw, _c = effect(off, arm, QUOTE_EP)
    assert abs(m + 0.01) < 1e-12 and hw == 0.0, (m, hw)
    print("✓ self-test passed: weight_curve mirrors the strategy (floor binds "
          "only at full strength, flagged clients exempt, identity above the "
          "floor, monotone, bounded); effect() handles disjoint seeds and "
          "signs correctly")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/_figures/dose_and_schedule"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return 0 if self_test() else 1

    plt = _mpl()
    args.out.mkdir(parents=True, exist_ok=True)
    off = load_seeds(args.base / "sweep_a10_notwagg")
    if not off:
        print(f"no weighting-OFF runs under {args.base / 'sweep_a10_notwagg'}")
        return 1

    made = []
    for fn in (lambda: fig_dose(plt, args.base, off, args.out),
               lambda: fig_schedule(plt, args.base, off, args.out),
               lambda: fig_trajectory(plt, args.base, args.out),
               lambda: fig_weight_curve(plt, args.out)):
        f = fn()
        if f:
            made.append(f)
    for f in made:
        print(f"wrote {f}")
    if not made:
        print("no figures produced -- check the run directories exist")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
