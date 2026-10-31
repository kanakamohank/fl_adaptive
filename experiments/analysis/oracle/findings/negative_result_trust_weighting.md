# Client re-weighting by a reliability signal helps under IID and hurts under non-IID

**Status:** complete. Six configurations × 5 seeds. Mechanism open.

---

## 1. Claim

> Down-weighting clients by a loss-based reliability signal **improves**
> accuracy only under a **perfectly IID** partition. At Dirichlet α = 1.0 —
> mild heterogeneity, the gentlest non-IID setting commonly tested — the
> effect has already **reversed**, and it deepens as skew increases. The sign
> is set by data heterogeneity, not by the quality of the signal: under
> non-IID, **improving the signal makes the harm worse**.

Three things make this worth reporting. First, the sign flip is total: across
three arms × 5 seeds at each of three partition settings, **15/15 comparisons
improve under IID and 30/30 degrade under non-IID**, with no exceptions.
Second, the flip occurs at the *first* departure from uniformity, not only
under severe skew. Third, the obvious objection to the negative half — "your
detector was too weak" — is ruled out: we raised detection from near-chance
to 17/20 and the harm **doubled**.

---

## 2. Setup

| | |
|---|---|
| Dataset | CIFAR-10, labels from CIFAR-10N `worst` (~40% annotator error) |
| Partition | **two settings compared**: IID, and Dirichlet α = 0.3 |
| Noise | 40% of clients noisy, `label_noise_rate = 1.0` → ~16% of all labels wrong |
| Federation | 50 clients, 10 sampled per round, 40 rounds, 5 local epochs |
| Model | small CIFAR CNN (~1.2M params) |
| Metric | mean test accuracy over the last 10 rounds ("late10"), 5 seeds, paired |
| Arms | full verification / TAVS skip / random skip at matched skip rate |

The weighting under test: when enabled, a verified client contributes
`behaviour_score × num_examples` to the aggregate instead of `num_examples`
alone. `behaviour_score` is the reliability signal, inverted so a suspected
noisy client contributes less.

---

## 3. Main result

### 3a. The sign flip

Effect of enabling the weighting, paired by seed, all three arms:

| partition | arm | weighting OFF | weighting ON | change | t | seeds improved |
|---|---|---|---|---|---|---|
| **IID** | full verify | 0.7295 | 0.7406 | **+0.0110** | +9.53 | **5/5** |
| **IID** | TAVS | 0.7290 | **0.7415** | **+0.0125** | +11.42 | **5/5** |
| **IID** | random | 0.7278 | 0.7326 | **+0.0048** | +3.37 | **5/5** |
| α = 1.0 | full verify | 0.3646 | 0.3600 | −0.0046 | −3.22 | **0/5** |
| α = 1.0 | TAVS | 0.3645 | 0.3591 | −0.0055 | −4.28 | **0/5** |
| α = 1.0 | random | 0.3649 | 0.3395 | −0.0254 | −11.01 | **0/5** |
| α = 0.3 | full verify | 0.3073 | 0.2994 | −0.0079 | −3.47 | **0/5** |
| α = 0.3 | TAVS | 0.3076 | 0.3013 | −0.0063 | −8.58 | **0/5** |
| α = 0.3 | random | 0.3073 | 0.2666 | −0.0407 | −6.10 | **0/5** |

Each OFF/ON pair differs in **exactly one** configuration field
(`disable_trust_weighted_aggregation`), verified programmatically against the
saved configs. Paired seeds were confirmed to hold the **identical** set of 20
noisy clients on both sides, so the pairing is valid.

**Where the sign flips.** α = 1.0 is mild heterogeneity — the gentlest non-IID
setting in common use, with α = 0.3 the aggressive one. The benefit is already
gone at α = 1.0, so the crossover lies between IID and α = 1.0, not between
α = 1.0 and α = 0.3. The harm then deepens with skew, most visibly on random
skip (+0.0048 → −0.0254 → −0.0407).

**One measured caveat.** Random skip's rate is matched to TAVS's *observed*
rate within each run, so a shift in TAVS's rate moves random's too. Only the
IID pair shifted (TAVS 34.5% → 36.5%, random 34.7% → 36.4%); α = 1.0 and
α = 0.3 shifted by ≤ 0.3 points. Scale of the contamination: in the
weighting-OFF data a **34-point** skip difference (full verify at 0% versus
TAVS at 34.5%) moved accuracy by **0.0005**, so a 1.6-point shift is worth
roughly **0.00002** — about 200× smaller than the +0.0048 random delta it
could affect, and it does not touch the two non-IID settings that carry the
central finding.

Under IID with weighting enabled, TAVS reaches **0.7415** — above full
verification (0.7406), above random skip (0.7326), and **+0.0125 above the
unweighted baseline**. This is the one configuration in which the system
delivers what it was designed to.

### 3b. Under non-IID, a better detector makes it worse

| config | detector AUROC | detection (of 20 noisy) | TAVS late10 | vs baseline |
|---|---|---|---|---|
| **weighting OFF** | — | — | **0.3076** | — |
| weighting ON, pooled signal | 0.738 | 9.8 / 20 | 0.3013 | **−0.0063** (t = −8.58, 0/5) |
| weighting ON, per-class signal | **0.895** | **17.0 / 20** | **0.2937** | **−0.0139** (t = −5.35, 0/5) |

Chance detection is 8/20. **Zero seeds out of five beat the baseline in either
weighted configuration**, and the better detector produced the larger loss.

With the weighting off, all three arms are indistinguishable: full 0.3073,
TAVS 0.3076, random 0.3073.

---

## 4. Ruling out "the detector was too weak"

Every intermediate mechanism improved between the two weighted runs:

| | pooled signal | per-class signal |
|---|---|---|
| detector AUROC | 0.738 | **0.895** |
| trust gap (clean − noisy) | +0.032 | **+0.129** |
| rank-of-noisy (chance 8/20) | 9.8 | **17.0** |
| routing: noisy pushed to verified | −3.4% | **−17.2%** |

The machinery worked. Detection, trust separation, and scheduling all improved
substantially. Accuracy still fell, and fell further.

**Independent check on the detector.** On the same logged data, FedNoRo's own
statistic (per-class mean loss, IJCAI 2023 §4.1) scores **AUROC 0.965**. This
is not a broken reimplementation of a known method — their detection
reproduces, and it is the *downstream use* that fails.

---

## 5. A metric that inverts the conclusion

In the strongest-detector run, `TAVS − random` was **+0.0451**, the largest
margin measured anywhere in this project.

| | late10 |
|---|---|
| do nothing (weighting off) | **0.3076** |
| full verification | 0.2941 |
| TAVS | 0.2937 |
| random skip | 0.2486 |

TAVS beats random by a wide margin **and is 0.0139 below doing nothing**. All
arms were harmed; random was harmed most.

Judged on "beats random", a better detector looks like a bigger success while
the model gets worse. The comparison against an unweighted baseline was fixed
in advance for this reason, and it is the only comparison that tracks the
actual outcome.

---

## 6. Mechanism: tested, not established

**Hypothesis.** Under a skewed partition a client may supply most of the
cohort's samples for some class. Silencing it costs the aggregate that class
entirely, and wrong labels may cost less than missing coverage. A sharper
detector would remove such clients more efficiently — which would explain the
inverse relationship.

**Measured** (200 rounds, 1076 client-rows, weights reconstructed from logs
using the same scoring the strategy ran):

*Coverage damage — direction supports the hypothesis:*

| | |
|---|---|
| class-distribution entropy, unweighted | 2.1544 |
| class-distribution entropy, weighted | 2.1109 |
| change | **−0.0435** (coverage concentrated) |
| share change, rare classes | **−0.00384** |
| share change, common classes | +0.00401 |

Weighting does concentrate coverage, and rare classes do lose more.

*Who gets silenced — confounded, then controlled:*

A naive correlation gives `rho(behaviour, irreplaceability) = +0.254`, which
appears to *refute* the hypothesis. It is confounded: CIFAR-10N `worst`
randomises a noisy client's labels, which flattens its class distribution and
mechanically lowers its measured irreplaceability. Splitting by true status:

| group | n | rho(behaviour, irreplaceability) | behaviour: irreplaceable vs replaceable |
|---|---|---|---|
| all | 1076 | +0.254 | +0.0687 |
| **clean only** | 546 | +0.084 | +0.0119 |
| **noisy only** | 530 | **−0.154** | **−0.0288** |

Among noisy clients, the more irreplaceable ones are penalised slightly
harder — the harmful case. Clean irreplaceable clients are not penalised.

**Verdict: inconclusive at the level of direct measurement.** Every sign
points the predicted way, but the magnitudes (entropy −0.04, rare-class share
−0.004, behaviour −0.03) are too small to obviously account for a −0.0139
accuracy loss.

**However, the IID/non-IID contrast in §3a is the hypothesis's central
prediction, and it holds exactly.** The coverage account predicts harm only
where client data stops overlapping — and the effect is positive in every IID
arm and negative in every non-IID arm, 15/15 each way. That is strong
circumstantial support even though the direct proxies were too blunt to
confirm it. **The mechanism is consistent with the evidence but not
established.**

---

## 7. Is the harm just a weak baseline?

The strongest objection to §3a is that heterogeneity does not only flip the
weighting's sign — it also wrecks the model. Holding rounds, noise and
weighting fixed and changing **only** the partition:

| partition | TAVS accuracy, weighting OFF |
|---|---|
| IID | **0.7290** |
| α = 1.0 | **0.3645** |

Mild heterogeneity alone halves accuracy before noise or weighting enter. So
a reviewer can reasonably ask whether the weighting hurts *because data is
heterogeneous*, or merely *because the model is bad* — these move together.

### 7a. Trying to fix the baseline: client drift is not the bottleneck

Local drift between averaging rounds is the textbook cause of FedAvg
degrading under skew, and the textbook fix is to average more often. Tested
at α = 1.0, holding total training work constant (learning rate unchanged, so
the comparison is not confounded by step size):

| config | rounds × local epochs | total passes | TAVS late10 |
|---|---|---|---|
| original | 40 × 5 | 200 | 0.3645 |
| more frequent averaging | 100 × 2 | 200 | **0.3337** |

Averaging 2.5× more often made it slightly **worse**, and the curve had
flattened by the end (+0.0005 over the final ten rounds). **Client drift is
not what limits this model under heterogeneity.** The remaining candidates —
a larger backbone, more clients per round, or an algorithmic fix such as
FedProx or SCAFFOLD — are each either a separate project or a reimplementation
of published work, and were not pursued.

### 7b. The harm does not fade as the model improves

If a weak model were the cause, the harm should shrink as training lifts
accuracy. It does not. Weighting effect measured at four points during
training:

| partition | window | baseline accuracy | weighting effect |
|---|---|---|---|
| IID | r1–10 | 0.4743 | +0.0010 |
| IID | r11–20 | 0.6347 | **+0.0122** |
| IID | r21–30 | 0.7007 | **+0.0126** |
| IID | r31–40 | 0.7290 | **+0.0125** |
| α = 0.3 | r1–10 | 0.1596 | **−0.0067** |
| α = 0.3 | r11–20 | 0.2297 | −0.0022 |
| α = 0.3 | r21–30 | 0.2761 | −0.0026 |
| α = 0.3 | r31–40 | 0.3076 | **−0.0063** |

Under α = 0.3 the baseline nearly **doubles** (0.16 → 0.31) while the effect
stays negative in all four windows, with no trend toward zero. The simple
form of the objection — "too weak to benefit" — is not supported.

The two settings also differ in *timing*. Under IID the benefit takes 10–20
rounds to appear (+0.0010 → +0.0122) as the trust average accumulates; under
non-IID the harm is present in the first ten rounds. Different signatures,
suggesting the two are not the same mechanism with opposite sign.

**What remains unresolved.** Every non-IID measurement sits between 0.16 and
0.37 accuracy. We have never observed a *healthy* non-IID model, so the
confound between heterogeneity and baseline quality is **weakened, not
broken**. Separating them cleanly needs a federated setup that trains well
under skew, which this model does not.

---

## 8. Scope limits

- **One dataset, one partition setting, one noise type.** CIFAR-10 / CIFAR-10N
  `worst` / Dirichlet α = 0.3. No claim beyond this.
- **Three partition points, not a curve.** IID, α = 1.0 and α = 0.3. The sign
  flips between IID and α = 1.0; exactly where in that gap, and whether the
  decay is gradual or abrupt, is unmeasured.
- **N = 50 clients, 10 sampled.** Not tested at larger scale.
- **Absolute accuracy under non-IID is low (~0.30 against 0.10 chance).**
  Examined in §7: the harm persists while accuracy doubles, so a weak model
  does not explain it, but no healthy non-IID measurement exists either.
- **Honest-but-noisy threat model only.** No adversarial clients were present
  in any run (`byzantine_fraction = 0.0` throughout).

---

## 9. What would strengthen this

1. **A federated setup that trains well under skew** (larger backbone, more
   clients per round, or FedProx/SCAFFOLD). This is the one change that would
   fully break the confound in §7. More frequent averaging has already been
   ruled out.
2. **Close the IID → α = 1.0 gap** (α = 3.0 or 5.0, i.e. very mild skew).
   The sign is known to flip inside that interval. Showing whether the benefit
   decays gradually or vanishes at the first departure from uniformity decides
   between two quite different claims, and the latter is the more striking.
3. **A second noise model.** Everything here uses CIFAR-10N `worst` (real
   human annotation errors). Repeating the contrast with uniform or pair-flip
   synthetic noise at a matched effective rate would show whether the
   interaction depends on the noise process. No new code is required.
4. **Direct mechanism test.** Hold out a known-rare class and measure
   per-class accuracy with and without weighting. The aggregate-level proxies
   used here were too blunt.
5. **Milder noise.** At ~16% corruption the model is near collapse. The same
   comparison at 3–7% would show whether the harm is specific to severe noise.
6. **A second dataset.** CIFAR-100 is supported in the codebase but did not
   train under Dirichlet(0.3) at 16% noise with this model: accuracy stayed at
   chance (0.0121 against 0.0100) for all 40 rounds while loss rose from
   `ln(100)` = 4.606 to 6.43. An earlier CIFAR-100 run under IID at 15% noise
   reached 0.32, so the dataset path works; the non-IID high-noise corner is
   out of reach for a ~1.2M-parameter CNN. Reaching it would need a larger
   backbone or many more rounds.

---

## 10. Reproduction

```bash
# baseline: weighting off
python3 experiments/pilot_skip_comparison.py --rounds 40 --local-epochs 5 \
  --num-clients 50 --clients-per-round 10 --data-split dirichlet --data-alpha 0.3 \
  --dataset cifar10 --label-noise-num-classes 10 \
  --noisy-client-fraction 0.4 --label-noise-rate 1.0 \
  --label-noise-type cifar10n --cifar10n-label-set worst \
  --trust-signal pretrain_loss_var --pretrain-loss-var-rescale 0.6,1.5 \
  --alpha-trust 0.7 --log-oracle-signals --seeds 1,2,3,4,5 \
  --results-dir results/tavs_integration_v4_noniid_notwagg

# weighted, strong signal: add --trust-signal per_class_loss_var --trust-weighted-aggregation
```

Analysis:

```bash
python3 experiments/analysis/oracle/compare_per_class_signal.py <run>/r40/*
python3 experiments/analysis/oracle/trust_vs_imbalance.py
python3 experiments/analysis/oracle/mechanism_coverage_loss.py
```

Results directories: `tavs_integration_v2_iid_worst_twagg` (IID, weighted),
`v6_iid_notwagg` (IID, unweighted), `v3_noniid_perclass` (non-IID, pooled
signal, weighted), `v4_noniid_notwagg` (non-IID, unweighted baseline),
`v5_noniid_perclass_trust` (non-IID, per-class signal, weighted).

---

## 11. Secondary observation: which statistic, when

A side finding from the same logs. Comparing per-class **mean** loss
(FedNoRo's statistic) against per-class **variance** (ours):

| run | noise | per-class mean | per-class variance |
|---|---|---|---|
| IID, random1 | mild | 0.819 | **0.903** |
| IID, worst | heavy | 0.985 | 0.987 |
| non-IID, worst | heavy | **0.965** | 0.895 |

Variance wins at mild noise, mean wins at heavy noise. A plausible reason:
within one class, loss variance scales like `p(1−p)` in the corrupted
fraction `p` — zero when nothing is wrong *and* zero when everything is
wrong. Variance detects *mixed* labels; it loses sensitivity when a client's
labels are uniformly corrupted. Mean rises monotonically with `p` and does
not double back.

Consistent with the data but **not directly measured** — confirming it needs
the true-versus-noisy split logged per class, which is not currently
recorded.

A further observation: a **single** class discriminates poorly (within-class
AUROC ≈ 0.56–0.62). Combining ten classes reaches 0.895–0.965. The detection
power comes from aggregating many weak per-class signals, not from any class
being individually informative.
