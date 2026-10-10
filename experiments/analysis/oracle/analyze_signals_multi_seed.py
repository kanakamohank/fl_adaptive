#!/usr/bin/env python3
"""Per-signal clean-vs-noisy classification table across all seeds of
one oracle arm. Reports error counts and mean accuracy ± std.

Usage:
  python experiments/analysis/oracle/analyze_signals_multi_seed.py \
      --results-dir results/oracle/r40/split-iid_N50_C10_noiseC40_R30_type-cifar10n-random1_ep5_oracle/skip00 \
      --arm full_verify

The script auto-discovers `<arm>_seed*/pipeline_results.json` under the
results-dir and computes, for each scalar oracle signal:

  (a) a per-client mean of the detrended signal (subtract round mean)
  (b) a midpoint-threshold classifier predicting clean/noisy
  (c) classification errors and accuracy per seed, aggregated across seeds

The error count is the honest metric: "AUC 1.000" from Gaussian
projection consistently overstated real-world performance. See
test_tavs_esp_strategy.py's expected_trust_after_rounds for how the
earlier projection errors arose.
"""
import argparse
import os
import statistics as st
import sys

# Make `experiments.analysis.oracle._signals` importable when invoked by file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from experiments.analysis.oracle import _signals as sig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Path to an oracle run's skip** directory, e.g. "
                         "results/oracle/r40/<cfg_tag>/skip00/")
    ap.add_argument("--arm", default="full_verify",
                    choices=("full_verify", "tavs_skip", "random_skip"),
                    help="Which arm's seed files to pull. Default full_verify.")
    args = ap.parse_args()

    paths = sig.discover_seed_files(args.results_dir, args.arm)
    if not paths:
        raise SystemExit(f"no {args.arm}_seed*/pipeline_results.json under {args.results_dir}")

    print(f"Arm: {args.arm}  seeds found: {len(paths)}")
    print(f"{'signal':<28} " + "  ".join(f"{sig.seed_number(p):>8}" for p in paths) +
          f"  {'mean_acc':>10} {'std_acc':>8}")

    for signal in sig.SCALAR_SIGNALS:
        per_seed_err = []
        per_seed_acc = []
        for p in paths:
            run = sig.load_run(p)
            osh = sig.oracle_history(run)
            is_noisy = sig.noisy_predicate(run)
            pcm = sig.per_cid_detrended_mean(osh, signal)
            r = sig.midpoint_threshold_errors(pcm, is_noisy)
            if r is None:
                per_seed_err.append(None)
                per_seed_acc.append(None)
            else:
                per_seed_err.append(r["total_errors"])
                per_seed_acc.append(r["accuracy"])
        # Format row
        cells = []
        for e in per_seed_err:
            if e is None:
                cells.append("     N/A")
            else:
                cells.append(f"{e:>3}/err")
        accs = [a for a in per_seed_acc if a is not None]
        mean_acc = st.mean(accs) if accs else float("nan")
        std_acc = st.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{signal:<28} " + "  ".join(f"{c:>8}" for c in cells) +
              f"  {mean_acc:>10.3f} {std_acc:>8.3f}")

    print("\nLegend: 'err/50' is classification errors out of 50 clients using a "
          "midpoint-threshold classifier on per-client detrended means. "
          "Lower = better separation. Chance baseline for 20 noisy of 50 is "
          "60% (always-clean).")


if __name__ == "__main__":
    main()
