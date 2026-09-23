#!/usr/bin/env python3
"""
Priority 1a: existing-data forensics for the noise-slippage question.

The mentor's question: "how many corrupted updates slip through under
random_skip vs TAVS?" -- and the reviewer's addition: "aggregate mass isn't
sufficient; also look at variance, upper-bound accuracy-impact, trust
distributions, trust trajectories, skip-decision correlation, and P-set
gradient overlap."

This script reads whatever pilot_results.json + experiment_summary.json
files already exist on disk and computes those diagnostics per arm per
seed. It does NOT run any new pipelines. Its job is to see whether the
+0.13pp TAVS-vs-Random accuracy gap has a mechanism-level story or is
just noise.

Load a pilot's results directory and get:

  Slippage table (per arm, seed-averaged):
    - fraction of noisy clients seen that were verified vs promoted
    - poison_budget_r = sum over promoted noisy clients of their trust weight
    - cumulative_poison = sum_r poison_budget_r
    - per-round std of poison_budget (bursty vs smooth)
    - upper-bound accuracy gap explained by slippage difference (rough)

  Trust distributions (tavs_skip only, arm-relevant):
    - final trust of noisy vs clean clients (histogram)
    - bimodality index (is there separation?)
    - trust trajectory over rounds (does separation grow?)

  Skip-decision alignment (existing data only):
    - per-round overlap of promoted set between arms (Jaccard)
    - if TAVS's promoted set ~= Random's promoted set on most rounds,
      the two arms are effectively the same policy on this data

  P-set direction consistency:
    - proxy: within each arm, per-round fraction of promoted clients that
      are actually the noisy ones. That is the arm's "how much noise
      slipped through unchecked" measure -- the mentor's exact ask.

Usage:
    python -m experiments.analyze_pilot_forensics \\
        results/pilot_skip_comparison/r20/split-iid_noiseC40_R30/skip46

Output: forensics.json + forensics.png in the same dir.
"""
import argparse
import json
import math
import statistics
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_run(run_dir: Path) -> Optional[Dict]:
    """Return pipeline_results.json for one arm+seed run, or None if missing."""
    p = run_dir / "pipeline_results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _noisy_cids(pipeline_results: Dict) -> set:
    """Flower proxy.cids of the noisy clients for this run."""
    # noisy_client_config_ids is the "honest_XX" list; cid_to_client_config_id
    # bridges from Flower's opaque cid. Invert the bridge, keep only entries
    # whose config_id is in the noisy set.
    noisy_cfg = set(pipeline_results.get("noisy_client_config_ids") or [])
    if not noisy_cfg:
        return set()
    bridge = pipeline_results.get("cid_to_client_config_id") or {}
    return {cid for cid, cfg in bridge.items() if cfg in noisy_cfg}


def _per_round_assignments(pipeline_results: Dict) -> Tuple[List[set], List[set]]:
    """Return (per_round_verified, per_round_promoted) as lists of cid sets.

    Prefers the ground-truth `round_assignments` (populated by newer runs
    that persist the strategy's own V/P record) over `tier_evolution` (which
    defaults absent clients to Tier 1 and is thus wrong for RandomSkip and
    FullVerification).
    """
    ra = pipeline_results.get("round_assignments")
    if ra:
        rounds = sorted(int(r) for r in ra.keys())
        V = [set(ra[str(r)].get("verified", []) or ra[str(r)].get("Verified", []))
             for r in rounds]
        P = [set(ra[str(r)].get("promoted", []) or ra[str(r)].get("Promoted", []))
             for r in rounds]
        return V, P

    # Legacy fallback for runs predating round_assignments persistence.
    # Works for TavsEspStrategy arms; unreliable for RandomSkip/FullVerify.
    tier_evo = pipeline_results.get("tier_evolution") or {}
    if not tier_evo:
        return [], []
    max_len = max(len(v) for v in tier_evo.values())
    V, P = [], []
    for r in range(max_len):
        v, p = set(), set()
        for cid, tiers in tier_evo.items():
            if r >= len(tiers):
                continue
            (p if tiers[r] >= 2 else v).add(cid)
        V.append(v); P.append(p)
    return V, P


def _per_round_promoted_by_cid(pipeline_results: Dict) -> List[set]:
    return _per_round_assignments(pipeline_results)[1]


def _per_round_verified_by_cid(pipeline_results: Dict) -> List[set]:
    return _per_round_assignments(pipeline_results)[0]


def slippage_stats(pipeline_results: Dict) -> Dict:
    """Per-run slippage summary — the mentor's exact ask.

    Returns a dict with:
      noisy_seen_verified   : total (client, round) pairs where a noisy client
                              landed in V (BVD ran on it)
      noisy_seen_promoted   : same but landed in P (BVD skipped)
      per_round_promoted_noisy: list of counts, one per round -- burstiness
      poison_share_per_round: list of fractions "of promoted set that are
                              noisy" (0.0 to 1.0), one per round
    """
    noisy = _noisy_cids(pipeline_results)
    if not noisy:
        return {"noise_injected": False}

    promoted_rounds = _per_round_promoted_by_cid(pipeline_results)
    verified_rounds = _per_round_verified_by_cid(pipeline_results)

    per_round_promoted_noisy = []
    per_round_verified_noisy = []
    per_round_promoted_size = []
    for r, (P, V) in enumerate(zip(promoted_rounds, verified_rounds)):
        pn = len(P & noisy)
        vn = len(V & noisy)
        per_round_promoted_noisy.append(pn)
        per_round_verified_noisy.append(vn)
        per_round_promoted_size.append(len(P))

    def _safe_mean(xs): return float(np.mean(xs)) if xs else 0.0
    def _safe_std(xs): return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

    return {
        "noise_injected": True,
        "num_rounds_observed": len(promoted_rounds),
        "noisy_client_count": len(noisy),
        "total_noisy_client_rounds_promoted": int(sum(per_round_promoted_noisy)),
        "total_noisy_client_rounds_verified": int(sum(per_round_verified_noisy)),
        # Burstiness metric: how variable is per-round poison?
        "promoted_noisy_per_round_mean": _safe_mean(per_round_promoted_noisy),
        "promoted_noisy_per_round_std":  _safe_std(per_round_promoted_noisy),
        # Full traces so the plotter and any manual re-analysis have them.
        "per_round_promoted_noisy": per_round_promoted_noisy,
        "per_round_verified_noisy": per_round_verified_noisy,
        "per_round_promoted_size": per_round_promoted_size,
    }


def trust_bimodality(experiment_summary: Dict) -> Dict:
    """Read noise_diagnostic and compute a simple bimodality index.

    bimodality index (BC) = (mean_clean - mean_noisy) / sqrt(var_clean + var_noisy)
    -- essentially a two-group Cohen's-d flavor. >0 means clean trust higher
    than noisy trust (mechanism working); ~0 means the two groups overlap.
    """
    diag = experiment_summary.get("noise_diagnostic") or {}
    if not diag.get("noise_injected"):
        return {"noise_injected": False}
    n = diag.get("sorted_trust_noisy") or []
    c = diag.get("sorted_trust_clean") or []
    if len(n) < 2 or len(c) < 2:
        return {"noise_injected": True, "n_noisy": len(n), "n_clean": len(c)}
    mn, mc = float(np.mean(n)), float(np.mean(c))
    sn, sc = float(np.var(n, ddof=1)), float(np.var(c, ddof=1))
    denom = math.sqrt(sn + sc) if (sn + sc) > 1e-12 else float("nan")
    d = (mc - mn) / denom if denom == denom else float("nan")
    return {
        "noise_injected": True,
        "n_noisy": len(n), "n_clean": len(c),
        "mean_trust_noisy": mn, "mean_trust_clean": mc,
        "var_trust_noisy": sn, "var_trust_clean": sc,
        # Positive = mechanism working (clean trusted more than noisy).
        "cohens_d_flavor": d,
        # Reviewer's ask: rank-biserial from the diagnostic if present.
        "bottom_k_overlap_pct": diag.get("bottom_k_overlap_pct"),
        "bottom_k_guarded_overlap_pct": diag.get("bottom_k_guarded_overlap_pct"),
    }


def skip_decision_overlap(pr_a: Dict, pr_b: Dict) -> Dict:
    """Per-round Jaccard between the promoted sets of two arms (same seed).

    If Jaccard is high (>0.7) across most rounds, the two policies pick
    nearly the same clients to skip -- they are effectively the same policy
    on THIS data, and any accuracy difference is coming from something other
    than "who to skip."
    """
    Pa = _per_round_promoted_by_cid(pr_a)
    Pb = _per_round_promoted_by_cid(pr_b)
    rounds = min(len(Pa), len(Pb))
    jacs = []
    for r in range(rounds):
        u = Pa[r] | Pb[r]
        if not u:
            continue
        jacs.append(len(Pa[r] & Pb[r]) / len(u))
    if not jacs:
        return {"per_round_jaccard": [], "mean": None}
    return {
        "per_round_jaccard": jacs,
        "mean": float(np.mean(jacs)),
        "std":  float(np.std(jacs, ddof=1)) if len(jacs) > 1 else 0.0,
    }


def process_pilot(pilot_results_json: Path) -> Dict:
    """Walk a pilot's directory tree and compute forensics for every run
    reachable from its pilot_results.json."""
    with open(pilot_results_json) as f:
        pilot = json.load(f)

    pilot_dir = pilot_results_json.parent
    # Rows in the pilot include arm, seed, and output_dir (implicit via
    # sibling dir naming convention). Re-derive: skip{XX} for the matched
    # rate we're in, and the sibling skip00 for the arms that ran there.
    # Just walk both.
    matched = pilot.get("matched_skip_rate", 0.0)
    matched_skip = f"skip{int(round(matched * 100)):02d}"
    root = pilot_dir.parent   # r20/{config_tag}/
    arm_dirs = {}
    # Look for both the matched-rate directory (random_skip lives there) and
    # the skip00 sibling (full_verify + tavs_skip + ablation arms live there).
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for d in sub.iterdir():
            if not d.is_dir():
                continue
            name = d.name   # e.g. "tavs_skip_seed3"
            # Parse "arm_seed{n}" from the trailing "_seed{n}".
            if "_seed" not in name:
                continue
            arm, _, seed_s = name.rpartition("_seed")
            try:
                seed = int(seed_s)
            except ValueError:
                continue
            arm_dirs.setdefault(arm, {})[seed] = d

    print(f"discovered arms: {sorted(arm_dirs.keys())}")

    # Per-arm slippage aggregated across seeds.
    forensics: Dict = {"pilot": str(pilot_results_json), "arms": {}, "matched_skip_rate": matched}
    for arm, seed_map in sorted(arm_dirs.items()):
        per_seed = {}
        for seed, d in sorted(seed_map.items()):
            pr = _load_run(d)
            if not pr:
                continue
            summary_p = d / "experiment_summary.json"
            summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
            per_seed[seed] = {
                "slippage": slippage_stats(pr),
                "trust_bimodality": trust_bimodality(summary),
            }
        # Aggregate slippage across seeds.
        agg = {}
        slip_seeds = [s for s in per_seed.values() if s["slippage"].get("noise_injected")]
        if slip_seeds:
            agg["mean_promoted_noisy_per_run"] = float(np.mean(
                [s["slippage"]["total_noisy_client_rounds_promoted"] for s in slip_seeds]
            ))
            agg["mean_verified_noisy_per_run"] = float(np.mean(
                [s["slippage"]["total_noisy_client_rounds_verified"] for s in slip_seeds]
            ))
            agg["mean_burstiness_std"] = float(np.mean(
                [s["slippage"]["promoted_noisy_per_round_std"] for s in slip_seeds]
            ))
        bim_seeds = [s["trust_bimodality"] for s in per_seed.values()
                     if s["trust_bimodality"].get("noise_injected") and
                        s["trust_bimodality"].get("cohens_d_flavor") is not None]
        if bim_seeds:
            cd = [b["cohens_d_flavor"] for b in bim_seeds
                  if b["cohens_d_flavor"] == b["cohens_d_flavor"]]   # drop NaN
            if cd:
                agg["mean_cohens_d_trust"] = float(np.mean(cd))
                agg["frac_seeds_positive_d"] = float(np.mean([1.0 if x > 0 else 0.0 for x in cd]))
        forensics["arms"][arm] = {
            "n_seeds": len(per_seed),
            "aggregate": agg,
            "per_seed": per_seed,
        }

    # Skip-decision overlap between tavs and random per seed.
    if "tavs_skip" in arm_dirs and "random_skip" in arm_dirs:
        common_seeds = sorted(set(arm_dirs["tavs_skip"]) & set(arm_dirs["random_skip"]))
        overlaps = []
        for s in common_seeds:
            pr_t = _load_run(arm_dirs["tavs_skip"][s])
            pr_r = _load_run(arm_dirs["random_skip"][s])
            if not pr_t or not pr_r:
                continue
            ov = skip_decision_overlap(pr_t, pr_r)
            if ov["mean"] is not None:
                overlaps.append((s, ov["mean"]))
        if overlaps:
            forensics["skip_overlap_tavs_random"] = {
                "per_seed_mean_jaccard": overlaps,
                "grand_mean": float(np.mean([o[1] for o in overlaps])),
            }
    return forensics


def make_plot(forensics: Dict, out_path: Path):
    """One panel per diagnostic. Kept minimal -- this is a triage plot, not a paper figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Pilot forensics: slippage, trust separation, skip-set overlap",
                 fontsize=13, fontweight="bold")

    # Panel 1: mean noisy-client-rounds promoted per arm.
    ax = axes[0]
    arms = sorted(forensics["arms"].keys())
    proms = [forensics["arms"][a]["aggregate"].get("mean_promoted_noisy_per_run", 0.0) for a in arms]
    verifs = [forensics["arms"][a]["aggregate"].get("mean_verified_noisy_per_run", 0.0) for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, proms, 0.4, label="promoted (skipped BVD)", color="#d95f02")
    ax.bar(x + 0.2, verifs, 0.4, label="verified (BVD ran)", color="#1b9e77")
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("noisy client-rounds (avg per run)")
    ax.set_title("How much noise slipped through each arm")
    ax.legend(); ax.grid(alpha=.3, axis="y")

    # Panel 2: Cohen's-d flavor -- trust separation.
    ax = axes[1]
    ds = [forensics["arms"][a]["aggregate"].get("mean_cohens_d_trust", float("nan")) for a in arms]
    ax.bar(x, ds, 0.55, color="#7570b3")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("mean Cohen's-d flavor (clean vs noisy trust)")
    ax.set_title("Trust separation (positive = mechanism working)")
    ax.grid(alpha=.3, axis="y")

    # Panel 3: per-seed skip-set jaccard tavs vs random.
    ax = axes[2]
    ov = forensics.get("skip_overlap_tavs_random")
    if ov:
        seeds, jacs = zip(*ov["per_seed_mean_jaccard"])
        ax.scatter(seeds, jacs, s=45, color="#e6ab02")
        ax.axhline(ov["grand_mean"], ls="--", color="grey",
                   label=f"mean = {ov['grand_mean']:.3f}")
        ax.axhline(0.7, ls=":", color="red",
                   label="0.7 (\"same policy\" threshold)")
        ax.set_xlabel("seed"); ax.set_ylabel("mean Jaccard(TAVS-P, Random-P)")
        ax.set_title("Skip-set overlap between TAVS and Random")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=.3)
    else:
        ax.text(0.5, 0.5, "tavs vs random skip-set\noverlap not computable\n(missing arm)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pilot_dir", type=str,
                        help="Directory containing pilot_results.json (e.g. "
                             "results/pilot_skip_comparison/r20/split-iid_noiseC40_R30/skip46)")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    pr_json = pilot_dir / "pilot_results.json"
    if not pr_json.exists():
        raise SystemExit(f"no pilot_results.json in {pilot_dir}")

    forensics = process_pilot(pr_json)
    out_json = pilot_dir / "forensics.json"
    out_png  = pilot_dir / "forensics.png"
    out_json.write_text(json.dumps(forensics, indent=2, default=str))
    make_plot(forensics, out_png)

    # Console report -- headline numbers only.
    print(f"\n{'=' * 80}")
    print(f"FORENSICS SUMMARY  ({pilot_dir.name})")
    print(f"{'=' * 80}")
    print(f"{'arm':>22} {'n':>4} {'noisy_promoted':>16} {'noisy_verified':>16} "
          f"{'burst_std':>10} {'d_trust':>9} {'d_pos%':>7}")
    for arm, info in sorted(forensics["arms"].items()):
        agg = info["aggregate"]
        n = info["n_seeds"]
        p = agg.get("mean_promoted_noisy_per_run", float("nan"))
        v = agg.get("mean_verified_noisy_per_run", float("nan"))
        b = agg.get("mean_burstiness_std", float("nan"))
        d = agg.get("mean_cohens_d_trust", float("nan"))
        f = agg.get("frac_seeds_positive_d")
        f_str = f"{f*100:.0f}%" if f is not None else "-"
        print(f"{arm:>22} {n:>4} {p:>16.2f} {v:>16.2f} {b:>10.2f} {d:>+9.3f} {f_str:>7}")

    ov = forensics.get("skip_overlap_tavs_random")
    if ov:
        print(f"\n  TAVS vs Random skip-set overlap: mean Jaccard = "
              f"{ov['grand_mean']:.3f} (>=0.7 => effectively same policy)")

    print(f"\nWrote: {out_json}\n       {out_png}")


if __name__ == "__main__":
    main()
