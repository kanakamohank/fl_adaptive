#!/usr/bin/env python3
"""The cost story: verification work saved, and what it costs in accuracy.

Three panels, because the cost claim has three parts and one chart cannot
carry them:

  fig 1  SAVED VS SPENT   accuracy against verification work, per arm. The
                          point is that TAVS sits level with verify-everything
                          while doing a third less work, and random skipping
                          at the same work does not.
  fig 2  CEILING          what was saved against what the WEIGHT BUDGET
                          allows. An earlier version of this panel drew the
                          reference line at "an oracle skips every clean
                          client" = (1 - p_noisy) = 60%. That line is
                          unreachable by construction: gamma_budget caps the
                          share of aggregate weight unverified clients may
                          hold, which caps the skip share at
                          g / ((1-g)*p_i + g) ~= 41% at the trust levels these
                          runs reach. The panel now draws BOTH, because the
                          gap between them is the finding.
  fig 3  WHAT BLOCKS IT   per 10-client cohort, how many clients are skipped
                          versus held back by each constraint, replayed from
                          the scheduler's own branch order.

                          The previous version of this panel was WRONG. Its
                          subtitle read "the weight budget binds in under 6%
                          of rounds and is not shown -- it is not the
                          limiter", and it omitted the budget entirely. That
                          6% was measured AFTER the demotion loop, which runs
                          until gamma <= gamma_budget and so is under the cap
                          by construction. Pre-demotion gamma is 0.815 and
                          the budget demotes ~200 of 400 client-rounds per
                          seed: it is the limiter, by a wide margin.

Everything is read from pilot_results.json (exact counts; total_verified +
total_promoted == clients_per_round * rounds in every arm) and from
pipeline_results.json for the constraint decomposition.

Usage:
  python3 experiments/analysis/oracle/plot_cost.py
  python3 experiments/analysis/oracle/plot_cost.py --self-test
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
    ci_halfwidth, paired_t,
)
from experiments.analysis.oracle.staleness import (  # noqa: E402
    load_runs, replay_schedule, tavs_config,
)

ARM_LABEL = {"full_verify": "verify everything",
             "tavs_skip": "TAVS (trust-chosen skips)",
             "random_skip": "random skips, same rate"}
ARM_COLOR = {"full_verify": "#1f4e79", "tavs_skip": "#548235",
             "random_skip": "#c0504d"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pilot_rows(base: Path, fam: str):
    """{arm -> {seed -> row}} from a run family's pilot_results.json files."""
    out = {}
    for p in sorted(glob.glob(str(base / fam / "**" / "pilot_results.json"),
                              recursive=True)):
        for r in json.loads(Path(p).read_text()).get("runs", []):
            prev = out.setdefault(r["arm"], {})
            if r["seed"] in prev:
                # A second pilot file in the family (e.g. a re-run that
                # landed at a different matched skip rate) would otherwise
                # silently replace seeds and the figure would mix configs.
                raise RuntimeError(
                    f"duplicate {r['arm']} seed {r['seed']} in {fam}: more "
                    f"than one pilot_results.json. Point --base at one.")
            prev[r["seed"]] = r
    return out


def arm_stats(rows, arm, ref="full_verify"):
    """(mean accuracy, mean verifications, paired delta vs ref, ci, n).

    The mean and the error bar are computed over the SAME seeds -- those
    shared with the reference arm. An earlier version averaged accuracy over
    all of the arm's seeds while pairing the error bar over the shared ones,
    so with unequal seed sets the point and its interval would have
    described different samples.
    """
    if arm not in rows:
        return None
    use = sorted(set(rows[arm]) & set(rows[ref])) if ref in rows else sorted(rows[arm])
    if not use:
        use = sorted(rows[arm])
    acc = [rows[arm][s]["late_accuracy"] for s in use]
    ver = [rows[arm][s]["total_verified"] for s in use]
    d = hw = float("nan")
    n = len(use)
    if ref in rows and len(use) >= 2 and set(use) <= set(rows[ref]):
        diffs = [rows[arm][s]["late_accuracy"] - rows[ref][s]["late_accuracy"]
                 for s in use]
        d, _t, n = paired_t(diffs)
        hw = ci_halfwidth(diffs)
    return statistics.mean(acc), statistics.mean(ver), d, hw, n


def fig_saved_vs_spent(plt, base, fams, out):
    fig, axes = plt.subplots(1, len(fams), figsize=(5.4 * len(fams), 4.6),
                             squeeze=False)
    drew = False
    for ax, (fam, title) in zip(axes[0], fams):
        rows = pilot_rows(base, fam)
        if "tavs_skip" not in rows:
            ax.set_visible(False)
            continue
        drew = True
        for arm in ("full_verify", "tavs_skip", "random_skip"):
            s = arm_stats(rows, arm)
            if not s:
                continue
            acc, ver, d, hw, _n = s
            yerr = [[hw], [hw]] if not math.isnan(hw) and arm != "full_verify" else None
            ax.errorbar([ver], [acc], yerr=yerr, marker="o", markersize=9,
                        capsize=4, color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                        linestyle="none")
            ax.annotate(f"{ver:.0f} checks", (ver, acc),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, color="#444444")
        fv = arm_stats(rows, "full_verify")
        if fv:
            ax.axhline(fv[0], color="#1f4e79", lw=0.9, ls=":")
        ax.set_xlabel("server-side verifications performed (of 400)")
        ax.set_title(title, fontsize=10)
    if not drew:
        plt.close(fig)
        return None
    axes[0][0].set_ylabel("global model accuracy (last-10 rounds)")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Fewer checks, same accuracy -- but only if the skips are "
                 "chosen by trust\nerror bars: 95% CI on the paired "
                 "difference from verify-everything", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    f = out / "cost_saved_vs_spent.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return f


def budget_ceiling(run):
    """Max share of a cohort that can skip, given the weight budget.

    gamma = sum_p / (|V| + sum_p) <= g, with each promoted client carrying
    p_i = sigmoid(c_lambda*(t_eff - 0.5)) and each verified client carrying 1.
    Solving for the promoted share m/n:

        p*m*(1-g) = g*(n-m)   ->   m/n = g / ((1-g)*p + g)

    Note the direction: a HIGHER p_i (more trusted clients) gives a LOWER
    ceiling, because each promoted client eats more of the budget. Raising
    theta_high therefore cannot buy more skipping.

    Returns (ceiling, mean_p_i) from the run's own promoted clients, or None.
    """
    tc = tavs_config(run)
    tbr = run.get("trust_by_round") or {}
    if not tbr:
        return None
    g = tc["gamma_budget"]
    cl, tau = tc.get("c_lambda", 8.0), tc.get("tau_ramp", 5.0)
    ra = run["round_assignments"]
    join = {}
    for rk in sorted(ra, key=int):
        for c in (ra[rk].get("sampled") or []):
            join.setdefault(c, int(rk))
    ws = []
    for rk in sorted(ra, key=int):
        if rk not in tbr:
            continue
        r = int(rk)
        for c in (ra[rk].get("promoted") or []):
            t = min(tbr[rk].get(c, 0.0),
                    1.0 - math.exp(-(r - join.get(c, r)) / tau))
            ws.append(1.0 / (1.0 + math.exp(-cl * (t - 0.5))))
    if not ws:
        return None
    p_i = statistics.mean(ws)
    return g / ((1.0 - g) * p_i + g), p_i


# The ceiling needs each promoted client's weight p_i, which needs
# trust_by_round. sweep_a10_twagg and the IID family both predate that
# logging, so the ceiling is measured on a DONOR family that has it and
# shares the config. stale_a10_sa40_sr40 is sweep_a10_twagg with the
# staleness caps lifted -- its skip rate is identical to 0.1 points
# (32.1% both), so its p_i distribution stands in. The IID family has no
# such donor, so its ceiling is left unmarked rather than guessed.
CEILING_DONOR = {"sweep_a10_twagg": "stale_a10_sa40_sr40"}


def fig_ceiling(plt, base, fams, out, p_noisy=0.4, budget=400):
    """Achieved saving against BOTH reference lines: the oracle bound that
    motivated the sweeps, and the ceiling the weight budget actually imposes.
    """
    labels, got, ceil, pis = [], [], [], []
    for fam, title in fams:
        rows = pilot_rows(base, fam)
        if "tavs_skip" not in rows:
            continue
        v = statistics.mean(x["total_verified"]
                            for x in rows["tavs_skip"].values())
        donor = CEILING_DONOR.get(fam, fam)
        bc = ([budget_ceiling(r) for r in load_runs(base / donor).values()]
              if (base / donor).exists() else [])
        bc = [x for x in bc if x]
        labels.append(title)
        got.append(budget - v)
        ceil.append(statistics.mean(c for c, _ in bc) * budget if bc else float("nan"))
        pis.append(statistics.mean(p for _, p in bc) if bc else float("nan"))
    if not labels:
        return None
    oracle = budget * (1.0 - p_noisy)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = list(range(len(labels)))
    ax.barh(y, got, color="#548235", label="skipped by TAVS")
    for i, (gv, cv) in enumerate(zip(got, ceil)):
        if not math.isnan(cv) and cv > gv:
            ax.barh([i], [cv - gv], left=[gv], color="#ffd966",
                    label="reachable under the budget" if i == 0 else None)
        if not math.isnan(cv) and oracle > cv:
            ax.barh([i], [oracle - cv], left=[cv], color="#d9d9d9",
                    label="blocked BY the budget" if i == 0 else None)
    ax.axvline(oracle, color="#c0504d", lw=1.6, ls="--")
    ax.annotate(f"oracle: skip every clean client ({oracle:.0f})",
                (oracle - 4, len(labels) - 0.38), fontsize=8, color="#c0504d",
                ha="right", va="bottom")
    for i, (gv, cv, pv) in enumerate(zip(got, ceil, pis)):
        ax.annotate(f"{gv:.0f}  ({gv / budget:.0%})", (gv / 2, i), ha="center",
                    va="center", fontsize=9, color="white")
        if math.isnan(cv):
            ax.annotate("budget ceiling not measurable\n(no trust log for "
                        "this family)", (gv + 6, i), fontsize=7.5,
                        color="#888888", va="center")
            continue
        ax.plot([cv, cv], [i - 0.34, i + 0.34], color="#7f6000", lw=2.0)
        ax.annotate(f"budget ceiling {cv / budget:.0%}\n(mean p_i={pv:.2f})",
                    (cv + 3, i), fontsize=8, color="#7f6000", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"verifications skipped, of {budget} opportunities")
    ax.set_xlim(0, budget * 0.78)
    ax.set_title("How much of the saving is captured, and what caps it\n"
                 "the oracle bound was never reachable: gamma_budget caps "
                 "the skip share long before it", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    f = out / "cost_ceiling.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return f


# Families carrying round-keyed trust logs. The branch replay needs
# trust_by_round, which only runs made after that logging fix have:
# sweep_a10_twagg and the IID family both predate it and return None, which
# is why this panel takes its own family list rather than the shared `fams`.
# stale_a10_sa40_sr40 is sweep_a10_twagg's config with the staleness caps
# lifted -- the arm where the caps provably never fire, so what remains is
# every OTHER constraint, which is exactly what this panel is about.
BLOCK_FAMS = [("stale_a10_sa40_sr40",
               "Dirichlet 1.0, staleness caps lifted"),
              ("stale_a10_sa08_sr20",
               "Dirichlet 1.0, caps doubled (8/20)")]


def fig_what_blocks(plt, base, fams, out):
    """Where each client-round goes, from the scheduler's own branch order.

    Unlike the version this replaces, these categories ARE a partition:
    every sampled client-round lands in exactly one, so the bars sum to the
    cohort size and can be stacked honestly.
    """
    fams = [(f, t) for f, t in BLOCK_FAMS if (base / f).exists()]
    rows, labels = [], []
    for fam, title in fams:
        gs = [g for g in (replay_schedule(r)
                          for r in load_runs(base / fam).values()) if g]
        if not gs:
            continue
        n = len(gs)
        tot = sum(g["sampled"] for g in gs) / n
        rows.append({k: sum(g[k] for g in gs) / n / tot * 10.0
                     for k in ("replay_promoted", "demoted_by_budget",
                               "stale", "tier1", "decoy")})
        labels.append(title)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    keys = [("replay_promoted", "#548235", "skipped"),
            ("demoted_by_budget", "#c0504d",
             "demoted by the weight budget"),
            ("stale", "#f4b183", "forced back by the staleness cap"),
            ("tier1", "#7f7f7f", "Tier 1: trust below theta_low"),
            ("decoy", "#8faadc", "drawn as a decoy")]
    left = [0.0] * len(rows)
    for k, col, lab in keys:
        vals = [r[k] for r in rows]
        ax.barh(range(len(rows)), vals, left=left, color=col, label=lab,
                height=0.5)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v > 0.35:
                ax.annotate(f"{v:.1f}", (l0 + v / 2, i), ha="center",
                            va="center", fontsize=8,
                            color="white" if col != "#f4b183" else "#333333")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("clients per 10-client cohort (a partition: these sum to 10)")
    ax.set_xlim(0, 10.4)
    ax.set_title("What stops more being skipped\n"
                 "the weight budget, not trust and not the staleness cap",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, framealpha=0.95)
    fig.tight_layout()
    f = out / "cost_what_blocks.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f

def self_test():
    # arm_stats: paired delta direction and the exact-accounting invariant.
    rows = {"full_verify": {s: {"late_accuracy": 0.30, "total_verified": 400}
                            for s in range(1, 6)},
            "tavs_skip": {s: {"late_accuracy": 0.30, "total_verified": 272}
                          for s in range(1, 6)}}
    acc, ver, d, hw, n = arm_stats(rows, "tavs_skip")
    assert acc == 0.30 and ver == 272 and n == 5
    assert abs(d) < 1e-12 and hw == 0.0, (d, hw)
    rows["tavs_skip"] = {s: {"late_accuracy": 0.29, "total_verified": 272}
                         for s in range(1, 6)}
    _a, _v, d2, _hw, _n = arm_stats(rows, "tavs_skip")
    assert abs(d2 + 0.01) < 1e-12, "a worse arm must give a NEGATIVE delta"
    # A missing reference must not raise.
    assert math.isnan(arm_stats({"tavs_skip": rows["tavs_skip"]}, "tavs_skip")[2])
    # A missing arm returns None rather than inventing a point.
    assert arm_stats(rows, "nonexistent") is None
    # The budget ceiling, hand-checked. gamma = p*m/((n-m)+p*m) <= g
    # rearranges to m/n = g / ((1-g)*p + g). Verify the formula by running
    # the constraint forward from the share it predicts.
    def ceiling(g, p):
        return g / ((1.0 - g) * p + g)
    for g, p in ((0.35, 0.982), (0.35, 0.774), (0.35, 0.50), (0.20, 0.90)):
        m = ceiling(g, p)                      # promoted share, n = 1
        gamma = p * m / ((1.0 - m) + p * m)    # feed it back in
        assert abs(gamma - g) < 1e-12, (g, p, m, gamma)
    # Direction, which is the counterintuitive part and the reason this is
    # pinned: MORE trusted clients give a LOWER ceiling, because each
    # promoted client consumes more of the weight budget. If this ever
    # inverts, "raise theta_high to skip more" starts looking correct.
    assert ceiling(0.35, 0.982) < ceiling(0.35, 0.774) < ceiling(0.35, 0.50)
    # And the measured figures: at p_i = 0.774 the ceiling is 41%, which is
    # far below the (1 - p_noisy) = 60% oracle bound the sweeps assumed.
    assert abs(ceiling(0.35, 0.774) - 0.4103) < 5e-4, ceiling(0.35, 0.774)
    assert ceiling(0.35, 0.774) < (1 - 0.4), (
        "the oracle bound must sit ABOVE the budget ceiling, or fig_ceiling's "
        "two reference lines are the wrong way round"
    )
    print("✓ self-test passed: arm_stats pairing/sign/missing-arm, "
          "budget-ceiling formula + direction")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/_figures/cost"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return 0 if self_test() else 1

    plt = _mpl()
    args.out.mkdir(parents=True, exist_ok=True)
    fams = [("sweep_a10_twagg", "Dirichlet 1.0 (skewed)"),
            ("tavs_integration_v2_iid_worst_twagg", "IID")]
    fams = [(f, t) for f, t in fams if (args.base / f).exists()]
    if not fams:
        print(f"no run families found under {args.base}")
        return 1

    made = [f for f in (fig_saved_vs_spent(plt, args.base, fams, args.out),
                        fig_ceiling(plt, args.base, fams, args.out),
                        fig_what_blocks(plt, args.base, fams, args.out)) if f]
    for f in made:
        print(f"wrote {f}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
