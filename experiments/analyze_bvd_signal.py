#!/usr/bin/env python3
"""
BVD signal diagnostic: are noisy clients distinguishable at the detector?

Reads pipeline_results.json (which the strategy's scheduling_history now
carries det_per_client per-client raw_distance / sigma_sq / max_z stats
alongside the already-logged det_max_z_* and det_sigma_sq_* aggregates).
Joins each round's per-client stats against the ground-truth noisy-client
identity via cid_to_client_config_id and noisy_client_config_ids, then
reports per-round noisy-vs-clean distributions on three quantities.

Interpretation table:
  * raw_distance separates noisy from clean, Z does NOT   -> sigma eats
      signal (slow EMA; try alpha_sigma=0.5 or lower)
  * raw_distance overlaps, Z overlaps                     -> SNR too low
      (noise regime is wrong, not detector)
  * both separate but max(Z) < tau_z                      -> calibration
      issue (lower tau_z)
  * both separate AND Z crosses tau_z for noisy clients   -> detector is
      working; something else explains the uniform slippage we saw

Usage:
    python -m experiments.analyze_bvd_signal \\
        path/to/pipeline_results.json \\
        [--rounds 1,5,10,15,20]
"""
import argparse
import json
import os
import sys
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(pr_path: Path) -> dict:
    with pr_path.open() as f:
        return json.load(f)


def noisy_cids(pr: dict) -> set:
    """Flower proxy.cids for this run's noisy clients.

    noisy_client_config_ids is a list of 'honest_XX' strings; the bridge is
    cid_to_client_config_id. Invert the bridge, keep entries whose config_id
    is in the noisy set. Returns empty set if either side is missing.
    """
    noisy = set(pr.get("noisy_client_config_ids") or [])
    if not noisy:
        return set()
    bridge = pr.get("cid_to_client_config_id") or {}
    return {cid for cid, cfg in bridge.items() if cfg in noisy}


def collect(pr: dict):
    """Return list of per-(round, cid) rows with raw_distance, sigma_sq, z,
    and a noisy flag. Prefers the entry's own `round` field so filtered or
    reordered histories still carry the correct label."""
    noisy = noisy_cids(pr)
    rows = []
    for i, s in enumerate(pr.get("scheduling_history", []), start=1):
        per_client = s.get("det_per_client") or {}
        rd = int(s.get("round", i))
        for cid, stats in per_client.items():
            rows.append({
                "round": rd,
                "cid": cid,
                "noisy": cid in noisy,
                "max_z": float(stats.get("max_z", 0.0)),
                "raw_dist": float(stats.get("raw_dist_at_argmax", 0.0)),
                "sigma_sq": float(stats.get("sigma_sq_at_argmax", 0.0)),
                "rank_raw": int(stats.get("rank_raw_at_argmax", 0)),
                "cohort_size": int(stats.get("cohort_size_at_argmax", 0)),
            })
    return rows, noisy


def _summary(xs):
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": xs[len(xs) // 2],
        "p25": xs[len(xs) // 4],
        "p75": xs[(3 * len(xs)) // 4],
    }


def _mannwhitney_u_p(xs, ys):
    """Two-sided Mann-Whitney U p-value via scipy if available, else None."""
    try:
        from scipy import stats
    except Exception:
        return None
    if not xs or not ys:
        return None
    try:
        res = stats.mannwhitneyu(xs, ys, alternative="greater")
        return float(res.pvalue)
    except Exception:
        return None


def _ratio(num, den):
    if den is None or den <= 0 or num is None:
        return float("nan")
    return num / den


def report(rows, noisy, tau_z: float = 5.0):
    """Per-round noisy-vs-clean distributions on raw_distance, sigma_sq, Z."""
    rounds = sorted({r["round"] for r in rows})
    print(f"\n{'=' * 100}")
    print(f"BVD SIGNAL DIAGNOSTIC")
    print(f"{'=' * 100}")
    print(f"n_noisy_cids_in_run = {len(noisy)}  "
          f"n_total_cids_in_run = {len({r['cid'] for r in rows})}  "
          f"tau_z (threshold) = {tau_z}")
    print()
    print(f"{'rd':>3}  {'raw_C':>10}  {'raw_N':>10}  {'ratio N/C':>10}  "
          f"{'sigma_med':>12}  {'Z_C':>8}  {'Z_N':>8}  "
          f"{'rank_N':>8}  {'noisy>tau':>12}")
    cross_count = 0
    noisy_rows_total = 0
    sep_rounds_rawdist = 0
    sep_rounds_z = 0
    all_noisy_raw, all_clean_raw = [], []
    all_noisy_z,   all_clean_z   = [], []
    all_noisy_rank, cohort_sizes = [], []
    for rd in rounds:
        rd_rows = [r for r in rows if r["round"] == rd]
        clean_rd = [r for r in rd_rows if not r["noisy"]]
        noisy_rd = [r for r in rd_rows if r["noisy"]]
        c_rd = _summary([r["raw_dist"] for r in clean_rd])
        n_rd = _summary([r["raw_dist"] for r in noisy_rd])
        c_z = _summary([r["max_z"] for r in clean_rd])
        n_z = _summary([r["max_z"] for r in noisy_rd])
        sigma_med = _summary([r["sigma_sq"] for r in rd_rows])["median"]
        noisy_cross = sum(1 for r in noisy_rd if r["max_z"] > tau_z)
        noisy_rows_total += len(noisy_rd)
        cross_count += noisy_cross
        if c_rd["median"] is not None and n_rd["median"] is not None:
            if n_rd["median"] > c_rd["median"]:
                sep_rounds_rawdist += 1
            if n_z["median"] is not None and c_z["median"] is not None and n_z["median"] > c_z["median"]:
                sep_rounds_z += 1
        rank_med = _summary([r["rank_raw"] for r in noisy_rd])["median"]
        cohort_med = _summary([r["cohort_size"] for r in noisy_rd])["median"]
        all_noisy_raw.extend(r["raw_dist"] for r in noisy_rd)
        all_clean_raw.extend(r["raw_dist"] for r in clean_rd)
        all_noisy_z  .extend(r["max_z"]    for r in noisy_rd)
        all_clean_z  .extend(r["max_z"]    for r in clean_rd)
        all_noisy_rank.extend(r["rank_raw"] for r in noisy_rd)
        cohort_sizes.extend(r["cohort_size"] for r in rd_rows)
        print(f"{rd:>3}  "
              f"{(c_rd['median'] if c_rd['median'] is not None else float('nan')):>10.4f}  "
              f"{(n_rd['median'] if n_rd['median'] is not None else float('nan')):>10.4f}  "
              f"{_ratio(n_rd['median'], c_rd['median']):>10.2f}  "
              f"{(sigma_med if sigma_med is not None else float('nan')):>12.6f}  "
              f"{(c_z['median'] if c_z['median'] is not None else float('nan')):>8.2f}  "
              f"{(n_z['median'] if n_z['median'] is not None else float('nan')):>8.2f}  "
              f"{(f'{rank_med}/{cohort_med}' if rank_med is not None else '-'):>8}  "
              f"{noisy_cross}/{len(noisy_rd):<10}")

    # Overall effect sizes: pooled noisy-vs-clean distributions, Mann-Whitney
    # U one-sided (noisy > clean), plus a bare median ratio so a 1.0001x
    # "separation" cannot be hidden behind a boolean count.
    print()
    print(f"POOLED ACROSS {len(rounds)} rounds "
          f"(n_noisy_rows = {len(all_noisy_raw)}, n_clean_rows = {len(all_clean_raw)}):")
    if all_noisy_raw and all_clean_raw:
        med_n_raw = sorted(all_noisy_raw)[len(all_noisy_raw)//2]
        med_c_raw = sorted(all_clean_raw)[len(all_clean_raw)//2]
        med_n_z   = sorted(all_noisy_z)[len(all_noisy_z)//2]
        med_c_z   = sorted(all_clean_z)[len(all_clean_z)//2]
        p_raw = _mannwhitney_u_p(all_noisy_raw, all_clean_raw)
        p_z   = _mannwhitney_u_p(all_noisy_z,   all_clean_z)
        p_raw_s = f"MWU_p={p_raw:.4f}" if p_raw is not None else "(scipy unavailable)"
        p_z_s   = f"MWU_p={p_z:.4f}"   if p_z   is not None else "(scipy unavailable)"
        print(f"  raw_dist: median noisy={med_n_raw:.4f}  clean={med_c_raw:.4f}  "
              f"ratio={_ratio(med_n_raw, med_c_raw):.2f}x  {p_raw_s}")
        print(f"  max_z:    median noisy={med_n_z:.2f}  clean={med_c_z:.2f}  "
              f"ratio={_ratio(med_n_z, med_c_z):.2f}x  {p_z_s}")
    if all_noisy_rank and cohort_sizes:
        med_rank = sorted(all_noisy_rank)[len(all_noisy_rank)//2]
        med_cs = sorted(cohort_sizes)[len(cohort_sizes)//2]
        print(f"  noisy clients' median LOO-rank at argmax block: "
              f"{med_rank}/{med_cs} (rank=cohort_size would mean 'always top-raw')")

    print()
    print(f"COUNTS")
    print(f"  rounds where noisy raw_dist median > clean raw_dist median: {sep_rounds_rawdist}/{len(rounds)}")
    print(f"  rounds where noisy Z        median > clean Z        median: {sep_rounds_z}/{len(rounds)}")
    print(f"  noisy-client-rounds with max_z > tau_z={tau_z}:              {cross_count}/{noisy_rows_total}  "
          f"({(100.0*cross_count/noisy_rows_total) if noisy_rows_total else 0.0:.1f}%)")

    verdict = _verdict(sep_rounds_rawdist, sep_rounds_z, cross_count,
                       noisy_rows_total, len(rounds))
    print()
    print(f"VERDICT (single-run; aggregate across seeds before claiming the mechanism works):")
    for line in verdict.split("\n"):
        print(f"  {line}")
    return verdict


def _verdict(sep_raw: int, sep_z: int, cross: int, noisy_total: int, rounds: int) -> str:
    """Classify this single run.

    Deliberately covers all four quadrants of (raw_strong, z_strong), plus
    the two detector-fires-anyway cases. Thresholds are intentionally loose
    on a single run (>=60% of rounds for separation; >=25% noisy-rows across
    tau_z for detector-fires). Aggregating across seeds should tighten these.
    """
    raw_strong = sep_raw >= 0.6 * rounds
    z_strong   = sep_z   >= 0.6 * rounds
    detector_fires = (cross / noisy_total) >= 0.25 if noisy_total else False

    if detector_fires and z_strong and raw_strong:
        return ("detector fires on noisy clients in THIS run (>=25% cross tau_z) "
                "and medians separate. Aggregate across seeds before claiming the "
                "mechanism works; also check why slippage was still uniform despite "
                "firing (verified-set composition? trust-EMA lag?).")
    if detector_fires and not z_strong:
        return ("Z crosses tau_z for some noisy clients despite medians overlapping "
                "-- long-tail detection, likely a few lucky-high-Z assignments per "
                "round. Mechanism not confirmed; try more seeds.")
    if z_strong and not raw_strong:
        return ("sigma_sq may have collapsed: Z separates noisy from clean, but "
                "noisy clients' raw distances do NOT exceed clean ones. Investigate "
                "sigma seeding and the per-block denominator -- the ratio is "
                "ambiguous without the numerator going the same direction.")
    if raw_strong and not z_strong:
        return ("sigma_sq eats signal: noisy clients ARE farther from the cohort "
                "(raw medians separate) but Z does NOT. Try lowering alpha_sigma "
                "(0.9 -> 0.5) or weighting the fresh median more vs the EMA.")
    if raw_strong and z_strong and not detector_fires:
        return ("calibration miss: both raw_dist and Z separate noisy from clean, "
                "but Z stays below tau_z={}. Lower tau_z or widen the behaviour "
                "score tail.".format(5.0))
    # not raw_strong and not z_strong -- SNR too low. The honest default.
    return ("SNR too low in THIS run: noisy clients do not show larger raw "
            "distances than clean ones. Mechanism cannot work at this noise "
            "regime; either raise label_noise_rate or reconsider what BVD is "
            "being asked to detect.")


def plot(rows, out_path: Path, tau_z: float = 5.0):
    """Three panels: raw_distance, sigma_sq, Z, each as noisy-vs-clean overlay."""
    rounds = sorted({r["round"] for r in rows})
    noisy_dist = [[r["raw_dist"] for r in rows if r["round"] == rd and r["noisy"]] for rd in rounds]
    clean_dist = [[r["raw_dist"] for r in rows if r["round"] == rd and not r["noisy"]] for rd in rounds]
    noisy_z    = [[r["max_z"]    for r in rows if r["round"] == rd and r["noisy"]] for rd in rounds]
    clean_z    = [[r["max_z"]    for r in rows if r["round"] == rd and not r["noisy"]] for rd in rounds]
    sigma_all  = [[r["sigma_sq"] for r in rows if r["round"] == rd] for rd in rounds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, data_n, data_c, title, ylabel, log_y in [
        (axes[0], noisy_dist, clean_dist, "raw_distance (argmax block)",
         "raw distance (log)", True),
        (axes[2], noisy_z,    clean_z,    "max_z per client", "Z", False),
    ]:
        for rd, nvals, cvals in zip(rounds, data_n, data_c):
            # Filter non-positive values when log scale; log(0) is -inf and
            # linear-plot-with-tiny-cvals is what the reviewer flagged as
            # visually crushing all clean points onto the x-axis.
            if cvals:
                cv = [v for v in cvals if (v > 0 or not log_y)]
                if cv:
                    ax.scatter([rd] * len(cv), cv, s=16, color="#1b9e77",
                               alpha=0.4, label="clean" if rd == rounds[0] else None)
            if nvals:
                nv = [v for v in nvals if (v > 0 or not log_y)]
                if nv:
                    ax.scatter([rd] * len(nv), nv, s=22, color="#d95f02",
                               alpha=0.6, marker="x",
                               label="noisy" if rd == rounds[0] else None)
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
        else:
            ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[2].axhline(tau_z, color="red", ls=":", label=f"tau_z = {tau_z}")
    axes[2].legend(loc="best")

    ax = axes[1]
    for rd, sv in zip(rounds, sigma_all):
        if sv:
            ax.scatter([rd] * len(sv), sv, s=16, color="#7570b3", alpha=0.4)
    ax.set_xlabel("round")
    ax.set_ylabel("sigma_sq (per block, log scale)")
    ax.set_yscale("log")
    ax.set_title("sigma_sq EMA across blocks")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pipeline_results", type=str,
                   help="path to a run's pipeline_results.json")
    p.add_argument("--tau-z", type=float, default=5.0,
                   help="BVD anomaly threshold used in the run (for cross-rate stat)")
    args = p.parse_args()

    pr_path = Path(args.pipeline_results)
    if not pr_path.exists():
        raise SystemExit(f"no file at {pr_path}")
    pr = load(pr_path)
    rows, noisy = collect(pr)
    if not rows:
        raise SystemExit("no det_per_client stats in scheduling_history -- "
                         "run was probably produced before the BVD logging hook. "
                         "Rerun with the current code.")

    report(rows, noisy, tau_z=args.tau_z)
    out_png = pr_path.parent / "bvd_signal.png"
    plot(rows, out_png, tau_z=args.tau_z)
    print(f"\nWrote: {out_png}")
    out_json = pr_path.parent / "bvd_signal.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
