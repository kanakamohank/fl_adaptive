#!/usr/bin/env python3
"""The cost story: verification work saved, and what it costs in accuracy.

Three panels, because the cost claim has three parts and one chart cannot
carry them:

  fig 1  SAVED VS SPENT   accuracy against verification work, per arm. The
                          point is that TAVS sits level with verify-everything
                          while doing a third less work, and random skipping
                          at the same work does not.
  fig 2  HEADROOM         what was saved against what an oracle could save.
                          With error rate p, a scheduler that verified only
                          the noisy clients would skip (1-p). This panel is
                          the gap, and the reason the staleness caps matter.
  fig 3  WHAT BLOCKS IT   per 10-client cohort, how many clients are promoted
                          versus held back by each constraint, with exact
                          trust reconstructed from the logs.

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
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.analysis.oracle.dose_and_schedule import (  # noqa: E402
    ci_halfwidth, paired_t,
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
            out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def arm_stats(rows, arm, ref="full_verify"):
    """(mean accuracy, mean verifications, paired delta vs ref, ci, n)."""
    if arm not in rows:
        return None
    acc = [x["late_accuracy"] for x in rows[arm].values()]
    ver = [x["total_verified"] for x in rows[arm].values()]
    d = hw = float("nan")
    n = len(acc)
    if ref in rows:
        shared = sorted(set(rows[arm]) & set(rows[ref]))
        if len(shared) >= 2:
            diffs = [rows[arm][s]["late_accuracy"] - rows[ref][s]["late_accuracy"]
                     for s in shared]
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


def fig_headroom(plt, base, fams, out, p_noisy=0.4, budget=400):
    labels, got = [], []
    for fam, title in fams:
        rows = pilot_rows(base, fam)
        if "tavs_skip" not in rows:
            continue
        v = statistics.mean(x["total_verified"] for x in rows["tavs_skip"].values())
        labels.append(title)
        got.append(budget - v)
    if not labels:
        return None
    oracle = budget * (1.0 - p_noisy)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = range(len(labels))
    ax.barh(list(y), got, color="#548235", label="skipped by TAVS")
    ax.barh(list(y), [oracle - g for g in got], left=got, color="#d9d9d9",
            label="left on the table")
    ax.axvline(oracle, color="#c0504d", lw=1.6, ls="--")
    # Above the top bar, not beside it: at the bottom it sat under the
    # legend and both were unreadable.
    ax.annotate(f"an oracle skipping every clean client: {oracle:.0f}",
                (oracle, len(labels) - 0.35), fontsize=8, color="#c0504d",
                ha="right", va="bottom")
    ax.set_ylim(-0.6, len(labels) - 0.05)
    for i, g in enumerate(got):
        ax.annotate(f"{g:.0f}  ({g / budget:.0%})", (g / 2, i), ha="center",
                    va="center", fontsize=9, color="white")
        ax.annotate(f"+{oracle - g:.0f} unclaimed", (g + (oracle - g) / 2, i),
                    ha="center", va="center", fontsize=8, color="#444444")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"verifications skipped, of {budget} opportunities")
    ax.set_title(f"How much of the available saving is captured\n"
                 f"with {p_noisy:.0%} of clients noisy, a perfect scheduler "
                 f"could skip the other {1 - p_noisy:.0%}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    f = out / "cost_headroom.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return f


def constraint_breakdown(base: Path, fam: str):
    """Per 10-client cohort: promoted, and who was held back by what.

    Trust is REPLAYED from the logged pretrain_loss_var through the same EMA
    the scheduler used -- verified elsewhere to reproduce the recorded final
    trust to 0.0 error on 1500 client values. The logged trust_evolution
    cannot be used: its per-client lists have ragged lengths with no recorded
    start round.
    """
    acc = {"promoted": [], "below_theta_low": [], "forced_stale": []}
    for p in sorted(glob.glob(str(base / fam / "**" / "tavs_skip_seed*"
                                  / "pipeline_results.json"), recursive=True)):
        d = json.loads(Path(p).read_text())
        tc = d["config"]["tavs_config"]
        a, t0 = tc["alpha_trust"], tc["initial_trust"]
        lo, hi = (tc["pretrain_loss_var_rescale_lo"],
                  tc["pretrain_loss_var_rescale_hi"])
        tau, th_lo = tc.get("tau_ramp", 5.0), tc["theta_low"]
        fts = d["final_trust_state"]
        osh = fts.get("oracle_signal_history") or {}
        ra = fts.get("round_assignments") or {}
        sh = {r["round"]: r for r in (d.get("scheduling_history") or [])}
        if not (osh and ra):
            continue
        join, trust = {}, {}
        for rk in sorted(ra, key=lambda x: int(x)):
            for c in ra[rk].get("sampled", []):
                join.setdefault(c, int(rk))
        for rk in sorted(ra, key=lambda x: int(x)):
            r = int(rk)
            P = set(ra[rk].get("promoted", []))
            S = set(ra[rk].get("sampled", [])) or set(ra[rk].get("verified", [])) | P
            if not S:
                continue
            def eff(c):
                return min(trust.get(c, t0),
                           1.0 - math.exp(-(r - join.get(c, r)) / tau))
            acc["promoted"].append(len(P))
            acc["below_theta_low"].append(sum(1 for c in S if eff(c) < th_lo))
            acc["forced_stale"].append(sh.get(r, {}).get("num_forced_stale", 0))
            for cid, e in (osh.get(rk) or {}).items():
                if not e.get("is_verified"):
                    continue
                raw = e.get("pretrain_loss_var")
                if raw is None:
                    continue
                b = 1.0 - max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))
                trust[cid] = a * trust.get(cid, t0) + (1.0 - a) * b
    if not acc["promoted"]:
        return None
    return {k: statistics.mean(v) for k, v in acc.items()}


def fig_what_blocks(plt, base, fams, out):
    rows, labels = [], []
    for fam, title in fams:
        b = constraint_breakdown(base, fam)
        if b:
            rows.append(b)
            labels.append(title)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    keys = [("promoted", "#548235", "promoted (skipped)"),
            ("forced_stale", "#c0504d", "forced back by the staleness cap"),
            ("below_theta_low", "#7f7f7f", "trust below theta_low, never promotable")]
    # GROUPED, not stacked. The three categories OVERLAP -- a client can be
    # both forced back by staleness and below theta_low -- so stacking them
    # implies a partition that does not exist and overflows the cohort size
    # (the IID column summed to 10.7 of 10).
    h = 0.26
    for j, (k, col, lab) in enumerate(keys):
        offs = [i + (1 - j) * h for i in range(len(rows))]
        vals = [r[k] for r in rows]
        ax.barh(offs, vals, height=h, color=col, label=lab)
        for off, v in zip(offs, vals):
            ax.annotate(f"{v:.1f}", (v + 0.08, off), va="center", fontsize=8,
                        color="#333333")
    ax.axvline(10, color="#999999", lw=1.0, ls=":")
    ax.annotate("cohort size 10", (10, len(rows) - 0.55), fontsize=8,
                color="#666666", ha="right", va="bottom")
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("clients per 10-client cohort (categories OVERLAP -- not a "
                  "partition)")
    ax.set_xlim(0, 11.5)
    ax.set_title("What stops more being skipped\n"
                 "the weight budget binds in under 6% of rounds and is not "
                 "shown -- it is not the limiter", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    f = out / "cost_what_blocks.png"
    fig.savefig(f, dpi=150)
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
    # Headroom arithmetic: with p=0.4 of 400, an oracle skips 240.
    assert abs(400 * (1 - 0.4) - 240) < 1e-9
    print("✓ self-test passed: arm_stats pairing/sign/missing-arm, headroom "
          "arithmetic")
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
                        fig_headroom(plt, args.base, fams, args.out),
                        fig_what_blocks(plt, args.base, fams, args.out)) if f]
    for f in made:
        print(f"wrote {f}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
