# Oracle-signal analysis scripts

These scripts analyze the `oracle_signal_history` payload inside each
oracle run's `pipeline_results.json`. All scripts are parameterised by
`--results-dir` (or `--pipeline-results` for the per-sample mechanism
plot) so they work against any arm directory produced by
`experiments/pilot_skip_comparison.py --log-oracle-signals`.

## Scripts

| script | purpose | produces |
|---|---|---|
| `analyze_signals_multi_seed.py` | per-signal clean/noisy classification errors + mean/std accuracy across seeds | text table on stdout |
| `plot_per_round_accuracy.py` | per-round signal strength curve (which signals mature when) | PNG |
| `plot_rank_of_noisy.py` | rank-of-noisy histogram + catch-at-K curve + Mann-Whitney U | PNG |
| `plot_mechanism_histograms.py` | per-sample pretrain-loss histograms for a handful of noisy and clean clients at a specific round | PNG |

Shared helpers live in `_signals.py` (loading, detrending,
midpoint-threshold classifier, rank, Mann-Whitney U). Unit tests at
`src/tests/test_oracle_analysis_helpers.py` cover the computation
functions.

## Example usage

```bash
# Full comparison across all 11 scalar signals on the CIFAR-10N 5-seed oracle.
python experiments/analysis/oracle/analyze_signals_multi_seed.py \
    --results-dir results/oracle/r40/split-iid_N50_C10_noiseC40_R30_type-cifar10n-random1_ep5_oracle/skip00 \
    --arm full_verify

# Per-round accuracy chart.
python experiments/analysis/oracle/plot_per_round_accuracy.py \
    --results-dir results/oracle/r40/split-iid_N50_C10_noiseC40_R30_type-cifar10n-random1_ep5_oracle/skip00 \
    --arm full_verify \
    --out plots/c10n_per_round.png

# Rank-of-noisy + catch-at-K.
python experiments/analysis/oracle/plot_rank_of_noisy.py \
    --results-dir results/oracle/r40/split-iid_N50_C10_noiseC40_R30_type-cifar10n-random1_ep5_oracle/skip00 \
    --arm full_verify --signal pretrain_loss_var \
    --out plots/c10n_rank_of_noisy.png

# Mechanism plot (requires the pilot to have been run with
# --log-per-sample-losses-at-round R).
python experiments/analysis/oracle/plot_mechanism_histograms.py \
    --pipeline-results results/oracle_mech/.../full_verify_seed1/pipeline_results.json \
    --round 20 \
    --out plots/mech_r20.png
```

## PNG convention

Scripts write PNGs under a `--out <path>` the caller specifies. They are
NOT committed to the repo; regenerate on demand. The `plots/` directory
is in `.gitignore` to prevent accidental commits.

## What the signals are

See `src/tavs_v2/tavs_esp_strategy.py` (TavsEspConfig) and
`src/clients/tavs_flower_client.py` (`_oracle_signals_on_train`) for
how each signal is computed. The project's current best signal on
CIFAR-10N random1 at oracle conditions is `pretrain_loss_var` (97% ± 1%
classification accuracy across 5 seeds; see
`analyze_signals_multi_seed.py` output).

## Caveats

- Classification is a midpoint-threshold on per-client detrended means;
  it is the "honest" readout. Gaussian AUC projections previously
  overstated detector quality — see `plot_rank_of_noisy.py` for the
  rank-based alternative.
- Signals are logged only for VERIFIED clients each round. Promoted
  clients have no BVD readouts by construction (BVD doesn't run on them).
- `pretrain_loss_var` on a mid-training round is a mixture of (a) the
  mixture of right- and wrong-labeled samples that noisy clients hold,
  and (b) the model's partial memorization of noisy labels. The exact
  mechanism (bimodal vs fat-tailed) is not resolvable from any single
  round; see `plot_mechanism_histograms.py` and run with an early round
  (R=3-5) and a mid round (R=15-20) to compare.
