#!/usr/bin/env python3
"""Rank-of-noisy plot — the honest detector-quality view at any sparsity.

For each seed, rank all clients by per-client detrended mean of the
chosen signal (highest first = most suspicious). Record where each true
noisy client lands in that ranking. Pool across seeds and compare
against a uniform null via Mann-Whitney U.

Also produces the catch-at-K curve: fraction of true noisy caught when
flagging the top-K ranks.

Usage:
  python experiments/analysis/oracle/plot_rank_of_noisy.py \
      --results-dir results/oracle/r40/<cfg_tag>/skip00 \
      --arm full_verify \
      --signal pretrain_loss_var \
      --out plots/rank_of_noisy.png
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from experiments.analysis.oracle import _signals as sig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arm", default="full_verify",
                    choices=("full_verify", "tavs_skip", "random_skip"))
    ap.add_argument("--signal", default="pretrain_loss_var")
    ap.add_argument("--descending", action="store_true", default=True,
                    help="Rank by DESCENDING signal (highest = most suspicious). "
                         "Default True because loss-family signals are high on "
                         "noisy clients. Pass --no-descending for cosine-style "
                         "signals that are LOW on noisy.")
    ap.add_argument("--no-descending", dest="descending", action="store_false")
    ap.add_argument("--out", default="plots/rank_of_noisy.png")
    args = ap.parse_args()

    paths = sig.discover_seed_files(args.results_dir, args.arm)
    if not paths:
        raise SystemExit(f"no {args.arm}_seed*/pipeline_results.json under {args.results_dir}")

    all_noisy_ranks, all_clean_ranks = [], []
    per_seed_noisy = []
    total_clients = None
    for p in paths:
        run = sig.load_run(p)
        osh = sig.oracle_history(run)
        if not osh:
            continue
        is_noisy = sig.noisy_predicate(run)
        pcm = sig.per_cid_detrended_mean(osh, args.signal)
        if not pcm:
            continue
        key = (lambda kv: -kv[1]) if args.descending else (lambda kv: kv[1])
        ranked = sorted(pcm.items(), key=key)
        ranks = {cid: i + 1 for i, (cid, _) in enumerate(ranked)}
        noisy = sorted([ranks[c] for c in pcm if is_noisy(c)])
        clean = sorted([ranks[c] for c in pcm if not is_noisy(c)])
        if not noisy or not clean:
            continue
        total_clients = len(pcm) if total_clients is None else total_clients
        all_noisy_ranks.extend(noisy)
        all_clean_ranks.extend(clean)
        per_seed_noisy.append(noisy)
        print(f"seed {sig.seed_number(p)}: {len(noisy)} noisy ranks "
              f"min={noisy[0]} med={noisy[len(noisy)//2]} max={noisy[-1]}")

    if not all_noisy_ranks:
        raise SystemExit("no ranks computed; check --signal and --arm")

    U, z, p = sig.mann_whitney_u(all_noisy_ranks, all_clean_ranks)
    print(f"\nMann-Whitney U = {U:.0f}, z = {z:.2f}, two-tailed p ≈ {p:.2e}")
    print(f"noisy ranks: mean={st.mean(all_noisy_ranks):.1f} median={st.median(all_noisy_ranks)}")
    print(f"clean ranks: mean={st.mean(all_clean_ranks):.1f} median={st.median(all_clean_ranks)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    N = total_clients or 50
    bins = list(range(1, N + 2))
    ax1.hist(all_noisy_ranks, bins=bins, color="#c0392b", alpha=0.7,
             label=f"noisy (n={len(all_noisy_ranks)})", edgecolor="white")
    ax1.hist(all_clean_ranks, bins=bins, color="#2b6cb0", alpha=0.4,
             label=f"clean (n={len(all_clean_ranks)})", edgecolor="white")
    ax1.axhline(len(all_clean_ranks) / N, color="gray", linestyle=":",
                 alpha=0.6, label="uniform null")
    ax1.set_xlabel(f"Rank (1 = most suspicious by {args.signal})")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Rank distribution — {args.signal}\n"
                   f"Mann-Whitney p ≈ {p:.1e}")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Catch-at-K
    K_range = list(range(1, N + 1))
    catch_mean, catch_std = [], []
    for K in K_range:
        caught = [sum(1 for r in s if r <= K) / len(s) for s in per_seed_noisy if s]
        catch_mean.append(st.mean(caught) if caught else float("nan"))
        catch_std.append(st.pstdev(caught) if len(caught) > 1 else 0.0)
    ax2.plot(K_range, catch_mean, color="#1f77b4", linewidth=2, label=args.signal)
    ax2.fill_between(K_range,
                      [m - s for m, s in zip(catch_mean, catch_std)],
                      [m + s for m, s in zip(catch_mean, catch_std)],
                      color="#1f77b4", alpha=0.2)
    ax2.plot(K_range, [K / N for K in K_range], color="gray", linestyle="--",
             label="random (uniform null)")
    # Mark |noisy| as reference K (if roughly constant across seeds)
    if per_seed_noisy:
        avg_num_noisy = int(round(st.mean(len(s) for s in per_seed_noisy)))
        ax2.axvline(avg_num_noisy, color="red", linestyle=":", alpha=0.5,
                     label=f"K=|noisy|≈{avg_num_noisy}")
    ax2.set_xlabel(f"K (top-K most-suspicious clients flagged by {args.signal})")
    ax2.set_ylabel("Fraction of true noisy caught")
    ax2.set_title(f"Catch-at-K — mean ± std across {len(per_seed_noisy)} seeds")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
