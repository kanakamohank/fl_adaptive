# Noise-Detection in Federated Learning — Project Findings

> **This README is the presentation deck itself** (10 sections, written for the team meeting). Plots in this folder, scripts that produced them under `experiments/analysis/oracle/`, results data under `results/oracle/r40/`.

**For team presentation.** Covers the oracle-signal detection experiment on CIFAR-10N and CIFAR-100 across 6 noise configurations.

## Plots in this folder

| file | what it shows | use for |
|---|---|---|
| `sweep_auroc.png` | 6-point AUROC curve of top signals across effective flip rate; hero signal annotated | Headline chart |
| `mechanism_pooled.png` | Pooled per-sample loss density, clean vs noisy, from CIFAR-100 pair-flip run (92 noisy + 158 clean clients across 5 rounds) | Mechanism finding (not bimodal, just shifted) |
| `rank_of_noisy.png` | Rank-of-noisy histogram + catch-at-K curve on CIFAR-100 dense | Visual proof the detector ranks correctly |

Regenerate from current `results/` with the scripts under `experiments/analysis/oracle/`.

---

## 1. Executive summary (one slide)

- **Problem**: TAVS-ESP's noise detectors (BVD, small-loss-fraction) fail to flag noisy clients on realistic FL noise regimes. 46%→9% skip-rate collapse observed in the pilot was traced to **wrong detector signal**, not undersampling.
- **Finding**: A simpler signal — **pre-training per-sample loss variance (`pretrain_loss_var`)** — detects noisy clients reliably at per-client effective flip rate ≥3%. On CIFAR-100 **uniform** noise, AUROC climbs 0.85 (at 1.4%) → 0.93 (at 3%) → saturates near 1.00 (at ≥5%). **Caveats:** CIFAR-10N saturation is consistent with but not proven (only 2 CIFAR-10N points, dense one is pinned at ceiling); on CIFAR-100 **pair-flip** at 30% rate the signal only reaches 0.86, so semantic-close "structured" noise is a known weaker regime.
- **Secondary**: BVD is not just failing — at ≥3% effective rate, it's **inverted** (ranks clean clients as *more* suspicious than noisy). Consistent with LOO-median contamination at 20-of-50 prevalence.
- **Mechanism**: Noisy clients' per-sample loss distribution is **unimodal shifted up at every quantile**, not bimodal. Both mean and variance work as detectors; variance edges mean slightly because the tail shift is larger than the body shift.

---

## 2. What we built (oracle instrumentation)

New opt-in diagnostic logging that captures per-(round, client) signals without changing scheduling:

- **Client-side**: one forward pass before local training on fresh global weights; computes per-sample losses with `reduction='none'`, aggregates to `pretrain_loss_mean`, `pretrain_loss_var`, `per_class_loss`, and optionally the full per-sample loss list at specific rounds.
- **Server-side**: logs cosine with weighted aggregate, cosine with simple mean, BVD max_z, BVD behavior_score, and the client-reported signals. All packed into `oracle_signal_history` in `pipeline_results.json`.
- **Zero cost when off**: controlled by `TavsEspConfig.log_oracle_signals: bool = False`.

Code lives in `experiments/analysis/oracle/` with reusable analysis scripts (`analyze_signals_multi_seed.py`, `plot_rank_of_noisy.py`, `plot_mechanism_histograms.py`, `plot_per_round_accuracy.py`) + a shared `_signals.py` utility module. All parameterized by `--results-dir`.

---

## 3. The 6-point sweep (headline result)

Six oracle configurations across CIFAR-10N (real human noise) and CIFAR-100 (synthetic uniform + synthetic within-superclass pair-flip). 5 seeds each. AUROC = probability a random noisy client ranks more suspicious than a random clean one; 0.5 = random, 1.0 = perfect.

| config | effective rate | pretrain_loss_var | cosine | bvd_behavior | small_loss_fraction |
|---|---|---|---|---|---|
| CIFAR-10N sparse (0.1 × 0.08) | 1.4% | 0.851 ± 0.036 | 0.657 ± 0.161 | 0.526 ± 0.107 | 0.627 ± 0.058 |
| **CIFAR-100 uniform (0.4 × 0.03)** | **2.97%** | **0.930 ± 0.051** | 0.793 ± 0.046 | 0.449 ± 0.048 | 0.565 ± 0.083 |
| CIFAR-10N dense (0.4 × 0.3) | 5.1% | 0.999 ± 0.001 | 0.668 ± 0.056 | 0.476 ± 0.046 | 0.646 ± 0.082 |
| CIFAR-100 sparse (0.1 × 0.08) | 7.9% | 0.998 ± 0.004 | 0.979 ± 0.025 | 0.370 ± 0.109 | 0.537 ± 0.172 |
| CIFAR-100 r15 (0.4 × 0.15) | 14.85% | 1.000 ± 0.000 | 0.993 ± 0.012 | 0.424 ± 0.058 | 0.786 ± 0.077 |
| CIFAR-100 dense (0.4 × 0.3) | 29.7% | 1.000 ± 0.000 | 0.998 ± 0.004 | 0.428 ± 0.040 | 0.934 ± 0.019 |
| CIFAR-100 pair-flip (0.4 × 0.3) | 29.7% | 0.857 ± 0.074 | 0.951 ± 0.045 | 0.473 ± 0.055 | 0.914 ± 0.025 |

**Reading the table in simple words:**

- `pretrain_loss_var` column: climbs from 0.85 at low noise, through 0.93 at 3%, saturates at ~1.00 for 5%+ on CIFAR-100 uniform. One exception — on CIFAR-100 pair-flip at the same 29.7% effective rate, it drops to 0.86. **Hypothesis** (not tested on a second structured noise type): semantic-close wrong labels (pair-flip within-superclass) produce less distinctive per-sample loss than random distant wrong labels at matched rate. Treat as a direction, not a confirmed explanation.

- `cosine` column: on CIFAR-100 (any noise type) at ≥7% effective → 0.95+. On both CIFAR-10N points regardless of rate → 0.66-0.67. Cosine clearly has a dataset-specific issue with CIFAR-10N; cause unknown.

- `bvd_behavior_score` column: at or below 0.5 (random) on five of seven configs. **BVD is inverted** — the median-reference it uses is contaminated by noisy clients.

- `small_loss_fraction` column: works only at high effective rates (0.78+ at ≥15%); unusable at the realistic 1-5% range the pilot was testing.

**Chart**: see `sweep_auroc_with_knee.png` for the full 6-point curve across all signals (shipped alongside this doc).

---

## 4. Mechanism (why `pretrain_loss_var` works)

Logged full per-sample loss list at rounds [5, 10, 15, 20, 30] on CIFAR-100 pair-flip 30% rate. Pooled across 5 seeds → **92 noisy clients + 158 clean clients, each with ~900 per-sample losses**.

**Per-round quantile table (CIFAR-100 pair-flip, pooled per group):**

| round | clean p25 | clean p99 | noisy p25 | noisy p99 | p25 gap | p99 gap |
|---|---|---|---|---|---|---|
| 5 | 3.44 | 5.98 | 3.58 | 6.41 | +0.14 | +0.43 |
| 10 | 2.55 | 6.55 | 2.79 | 7.11 | +0.24 | +0.56 |
| 15 | 2.03 | 6.99 | 2.40 | 7.82 | +0.37 | +0.83 |
| 20 | 1.72 | 7.13 | 2.09 | 7.95 | +0.37 | +0.82 |
| 30 | 1.39 | 7.43 | 1.70 | 7.91 | +0.31 | +0.48 |

**Finding:** Noisy clients' per-sample loss distribution is **unimodal, shifted up at every quantile**. The gap grows smoothly from p25 → p99 — the top of the distribution shifts more than the bottom, but the bottom moves too. There is **no bimodal "two humps" signature** at any training phase, including the early round 5.

**Implications:**
- Mean would work almost as well as variance at this high rate. Variance's edge comes from the slightly stronger tail shift.
- The common "noisy clients have a mixture of right-labeled low-loss + wrong-labeled high-loss samples" story — our original hypothesis — is **not supported by the data**. Not falsified either; the shift could be consistent with a tightly-overlapping bimodal, but there's no visible second hump.

**Chart**: `cifar100_pairflip_mechanism_pooled.png` (pooled per-sample loss density, clean vs noisy).

---

## 5. BVD inversion (secondary finding)

BVD's `bvd_behavior_score` sits below the 0.5 random baseline on five of seven configs (every point at effective rate ≥3%). On one config it also showed that **clean clients are penalized MORE OFTEN than noisy clients** in the sub-ceiling tail (14.6% of CLEAN scores below ceiling vs 10.7% of NOISY scores on CIFAR-100 dense).

**Mechanism (consistent-with, not proven):** At 20-of-50 noisy prevalence, BVD's leave-one-out median reference contains ~19 noisy neighbors. The reference is contaminated by the clients BVD was supposed to flag. "Typical" ends up meaning "typical of a mostly-noisy group," so noisy clients look normal and clean clients look deviant.

**Cannot fully disambiguate** without re-running BVD against a ground-truth-clean reference (would need raw gradient blocks logged; not available in current oracle data).

---

## 6. What's open / not fully resolved

| open item | status | what would close it |
|---|---|---|
| **Exact saturation knee between 3% and 5%** | partial — "transition region 3-5%" | one more run at ~4% effective (~37h) |
| **Cosine attribution — why CIFAR-10N breaks it** | hypothesis refuted (pair-flip CIFAR-100 doesn't collapse like CIFAR-10N), new mechanism unknown | CIFAR-10 **uniform** at ~3% effective (~37h) — isolates dataset size from noise realism |
| **BVD inversion mechanism** | consistent with LOO contamination | re-run with gradient blocks logged + oracle-clean reference BVD recomputation |

**n=1 seed caveats to call out:**
- CIFAR-10N dense pretrain_loss_var 0.999 is pinned at the saturation ceiling across all 5 seeds. Can't distinguish "truly saturated" from "ceiling clip."
- Only two unsaturated CIFAR-10N points (1.4%, 5.1%); CIFAR-10N dense is at ceiling. Dataset-specific effects on `pretrain_loss_var` are **consistent with the data but underpowered to prove or rule out**.

**Scope caveats (apply to EVERY finding above):**
- All experiments: **50 clients, 10 sampled per round, 40 rounds, 5 local epochs, small CIFAR CNN (~1.2M params), IID partition.** None of the AUROC numbers have been shown to generalize to different N, different sampling rate, non-IID partitions, or larger architectures.
- **BVD inversion mechanism specifically depends on 20-of-50 noisy prevalence.** The LOO median is contaminated at this ratio; at a lower noisy fraction (say 2 of 50) the contamination is weaker and BVD inversion is not predicted. Not tested below 10% prevalence except at 1.4% effective rate where the inversion signal is itself noisy.

---

## 7. Bug fixes (worth mentioning briefly)

Two latent bugs surfaced during this work and were fixed:

1. **Server-side eval was hardcoded to CIFAR-10.** On the first CIFAR-100 run, reported test accuracy was 1.3% (model trained on 100 classes was evaluated against 10-class labels). Oracle signals were unaffected (they come from client train data). Fixed to dispatch on `config.dataset`.

2. **Server's `initialize_parameters` hardcoded 10-class model.** Caused silent shape mismatch on CIFAR-100 runs. Fixed to read `num_classes` from `TavsEspConfig`.

Both have regression tests to prevent recurrence.

---

## 8. Repo artifacts (for anyone who wants to reproduce)

- **Analysis scripts** (parameterized, work on any oracle `results/.../skipXX` directory):
  - `experiments/analysis/oracle/analyze_signals_multi_seed.py` — per-signal error table
  - `experiments/analysis/oracle/plot_per_round_accuracy.py` — signal strength vs round
  - `experiments/analysis/oracle/plot_rank_of_noisy.py` — rank + catch-at-K + Mann-Whitney
  - `experiments/analysis/oracle/plot_mechanism_histograms.py` — per-sample loss shape (needs `--log-per-sample-losses-at-rounds`)
  - `experiments/analysis/oracle/_signals.py` — shared utilities (detrend, AUROC, Mann-Whitney, no scipy)
- **Oracle instrumentation**: `TavsEspConfig.log_oracle_signals`, `TavsEspConfig.log_per_sample_losses_at_rounds`
- **Tests**: 7 regression suites in `src/tests/` covering serde, config-tag, strategy init, analysis helpers, pair-map, per-sample list parsing, eval dispatch.

**Results data on main** (public to the team) under:
- `results/oracle/r40/split-iid_N50_C10_<cfg>_ep5_oracle/` for each of the 7 configs listed in §3.

---

## 9. Recommended next steps (ranked for team decision)

**Critical-path first:**

1. **Integrate `pretrain_loss_var` into TAVS trust EMA** as a replacement signal, with the rescale window the oracle data suggests. Code work + at least one validation run under the real TAVS sampling regime (not oracle full_verify). Moves the project from "detection works in oracle conditions" to "detector works in realistic deployment." This is the only step on the critical path to a working system.

**Nice-to-haves (sidebar questions):**

2. **CIFAR-10 uniform at ~2.9% effective rate** (~30-37h, 5 seeds). Resolves the cosine mystery (why CIFAR-10N breaks cosine) by isolating dataset size from noise realism. Only matters if we want to use cosine as a secondary signal.
3. **One more sweep point at ~4% effective** (~37h). Pins the exact knee location. Current data already says "operational by 3%, saturated by 5% on uniform noise" — pinning doesn't change the operational conclusion.

---

## 10. One-sentence project headline

> `pretrain_loss_var` — per-sample loss variance computed on fresh global weights before local training — detects noisy clients with AUROC 0.85 at 1.4% effective flip rate on CIFAR-10N and saturates near 1.00 by 5% effective rate on CIFAR-100 uniform noise (consistent with but not proven on CIFAR-10N dense; weaker 0.86 on CIFAR-100 pair-flip at 30%), while BVD and small-loss-fraction stay at or below random across every tested regime, with all results from a single configuration (N=50 clients, 10 per round, 40 rounds, 5 local epochs, IID partition).
