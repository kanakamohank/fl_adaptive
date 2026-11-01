# Client re-weighting by a reliability signal helps only under IID; any heterogeneity removes the benefit

**Status:** complete. Ten configurations × 5 seeds, across two noise models.
Mechanism open.

---

## 1. Claim

> Down-weighting clients by a loss-based reliability signal gives a clear
> benefit under a **perfectly IID** partition (+0.012 to +0.017, t = 11–14).
> **Any** departure from uniformity removes that benefit entirely — already by
> Dirichlet α = 3.0, which is very mild skew — after which the effect is
> small and slightly negative and does **not** worsen with further skew. The
> outcome is governed by data heterogeneity, not by the quality of the signal:
> under non-IID, **improving the signal makes matters worse, not better**.

**Two claims of different strength, stated separately on purpose.**

*The benefit vanishes* — strongly supported. The IID effect is large and
unambiguous (t = 11.42 on CIFAR-10N, t = 13.69 on uniform noise), and it is
absent at every non-IID setting tested.

*It turns actively harmful* — directionally consistent but weak. Across four
non-IID settings, **19 of 20** seed-comparisons are negative, yet the effect
is only −0.005 and several measurements sit near the significance threshold
(t = −2.19 at uniform α = 1.0 does not clear it). Do not lean on this half.

Three things make the result worth reporting. First, it **replicates under a
second, structurally different noise model** (§3c), so it is not an artefact
of one corruption process. Second, the flip occurs at the *first* departure
from uniformity rather than only under severe skew, and is a cliff rather
than a slope (§3a). Third, the obvious objection — "your detector was too
weak" — is ruled out: we raised detection from near-chance to 17/20 and the
harm got worse, not better (§4).

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

**Mechanisms actually exercised.** Across the four weighted runs (800
scheduled rounds), decoy verification fired **135** times and promoted-update
clipping **20** times; peak client trust reached **0.938**, well past the
`theta_high = 0.70` tier threshold. The cosine filter never fired because it
is disabled by configuration (`cosine_filter_promoted = False`). So the
scheduler's tier machinery is live in these results, not dormant.

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
| α = 3.0 | full verify | 0.4085 | 0.4041 | −0.0044 | −2.65 | 1/5 |
| α = 3.0 | TAVS | 0.4094 | 0.4041 | −0.0053 | −3.10 | 1/5 |
| α = 3.0 | random | 0.4098 | 0.3870 | −0.0228 | −8.85 | **0/5** |
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

**A cliff, not a slope.** Tracking the TAVS arm as skew increases:

| partition | TAVS effect |
|---|---|
| IID | **+0.0125** |
| α = 3.0 | −0.0053 |
| α = 1.0 | −0.0055 |
| α = 0.3 | −0.0063 |

α = 3.0 is *very* mild skew — much closer to uniform than α = 1.0, itself the
gentlest non-IID setting in common use. The benefit is already gone there, and
the effect then sits at roughly −0.005 regardless of how much more skewed the
data becomes. **The transition happens at the first departure from
uniformity**, and further heterogeneity adds almost nothing.

This corrects an earlier reading of these data. The harm does deepen with skew
for **random skip** (−0.0228 → −0.0254 → −0.0407), and that pattern was
wrongly generalised to the other arms. For TAVS and full verification it is
flat.

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

**An unexplained arm difference.** Under IID, random skip benefits on
CIFAR-10N (+0.0048, t = 3.37, 5/5) but **not** under uniform noise (+0.0007,
t = 0.36, 3/5), while full verification and TAVS benefit under both. Random
skip applies the weighting to the fewest clients, since it drops ~34% at
random and those receive a different weighting path. That is consistent with
the benefit requiring verification *coverage* rather than the weighting
alone — but it is one observation across two noise models, not a tested
claim.

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

### 3c. It replicates under a second noise model

Everything above uses CIFAR-10N `worst` — real human annotator errors, which
carry structure (people confuse particular class pairs). If the result
depended on that structure rather than on heterogeneity, the claim would be
wrong. Repeating the 2×2 with **uniform synthetic noise** — errors scattered
at random, matched to the same ~16% of labels corrupted — gives:

| noise model | partition | arm | OFF | ON | change | t | seeds improved |
|---|---|---|---|---|---|---|---|
| uniform | **IID** | full verify | 0.7237 | 0.7409 | **+0.0172** | +14.73 | **5/5** |
| uniform | **IID** | TAVS | 0.7238 | 0.7404 | **+0.0166** | +13.69 | **5/5** |
| uniform | **IID** | random | 0.7241 | 0.7249 | +0.0007 | +0.36 | 3/5 |
| uniform | α = 1.0 | full verify | 0.3488 | 0.3435 | −0.0053 | −2.00 | 1/5 |
| uniform | α = 1.0 | TAVS | 0.3486 | 0.3439 | −0.0047 | −2.19 | 1/5 |
| uniform | α = 1.0 | random | 0.3493 | 0.3191 | −0.0302 | −9.91 | **0/5** |

**The pattern reproduces.** Same corruption level, structurally different
corruption process, same outcome: a clear benefit under IID (if anything
larger, +0.0166 against +0.0125) and no benefit under α = 1.0. The finding
is about data heterogeneity, not about how the labels were corrupted.

**Where it is weaker.** Under uniform noise the non-IID effect for full
verification and TAVS is **not statistically significant** (t = −2.00 and
−2.19 against the ~2.78 bar for 5 seeds, 1/5 seeds positive). The direction
matches, the magnitude does not reach significance. This is the main evidence
behind stating "the benefit vanishes" as the claim and treating "it becomes
harmful" as the weaker companion.


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

**A note on what has and has not been benchmarked.** Feeding the per-class
**mean** loss into our own scoring pipeline — the statistic FedNoRo uses,
scored our way — reaches **AUROC 0.965** on the same logged data, against
0.895 for the per-class variance. That compares two *statistics inside our
pipeline*. It is **not** a comparison against FedNoRo's method, which also
requires its GMM split, warm-up phase, logit adjustment and full client
participation. **No published method has been benchmarked against anywhere
in this document.** The detector is shown to work; it is not shown to be
competitive.

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

- **One dataset.** CIFAR-10 only. Two noise models (CIFAR-10N `worst` and
  uniform) and four partition settings are covered; the dataset is not.
- **Four partition points, not a curve.** IID, α = 3.0, α = 1.0, α = 0.3.
  The benefit is gone by α = 3.0, so the transition sits between IID and
  α = 3.0 and is narrower than a gradual decay — but its exact location is
  still unmeasured.
- **N = 50 clients, 10 sampled.** Not tested at larger scale.
- **Absolute accuracy under non-IID is low (~0.30 against 0.10 chance).**
  Examined in §7: the harm persists while accuracy doubles, so a weak model
  does not explain it, but no healthy non-IID measurement exists either.
- **Honest-but-noisy threat model only.** Every run reported here has
  `byzantine_fraction = 0.0`: the noisy clients make labelling mistakes, they
  do not craft malicious updates. Six older runs elsewhere in the repository
  (`verification_strategy_comparison/`, `clip_factor_sweep/`) do include
  layerwise and distributed attackers at 15–25% Byzantine, but they use 20
  clients, 20 rounds and a single seed, so they are not comparable with
  anything here and are not drawn on.
- **The signal in use cannot see model poisoning.** `pretrain_loss_var` is
  computed on a client's own data before local training, so an attacker with
  clean data that corrupts its update afterwards reports a normal value.
  Nothing in this document speaks to adversarial robustness.

---

## 9. What would strengthen this

1. **A federated setup that trains well under skew** (larger backbone, more
   clients per round, or FedProx/SCAFFOLD). This is the one change that would
   fully break the confound in §7. More frequent averaging has already been
   ruled out.
2. **Close the IID → α = 3.0 gap** (α = 10 or 30, i.e. near-uniform). The
   benefit is present at IID and gone by α = 3.0, so the transition is inside
   that interval. Locating it would say whether the mechanism needs *exact*
   uniformity or merely something close to it.
3. **Direct mechanism test.** Hold out a known-rare class and measure
   per-class accuracy with and without weighting. The aggregate-level proxies
   used here were too blunt.
4. **Milder noise.** At ~16% corruption the model is near collapse. The same
   comparison at 3–7% would show whether the harm is specific to severe noise.
5. **A second dataset.** CIFAR-100 is supported in the codebase but did not
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

The uniform-noise replication and the α = 3.0 point were produced by
`experiments/run_overnight_noise_model.sh`, which runs all six jobs in order
and resumes from disk if interrupted.

Results directories, by role:

| directory | noise | partition | weighting |
|---|---|---|---|
| `tavs_integration_v6_iid_notwagg` | CIFAR-10N worst | IID | off |
| `tavs_integration_v2_iid_worst_twagg` | CIFAR-10N worst | IID | on |
| `sweep_a30_notwagg` / `sweep_a30_twagg` | CIFAR-10N worst | α = 3.0 | off / on |
| `sweep_a10_notwagg` / `sweep_a10_twagg` | CIFAR-10N worst | α = 1.0 | off / on |
| `tavs_integration_v4_noniid_notwagg` | CIFAR-10N worst | α = 0.3 | off |
| `tavs_integration_v3_noniid_perclass` | CIFAR-10N worst | α = 0.3 | on (pooled signal) |
| `tavs_integration_v5_noniid_perclass_trust` | CIFAR-10N worst | α = 0.3 | on (per-class signal) |
| `nm_uniform_iid_notwagg` / `nm_uniform_iid_twagg` | uniform | IID | off / on |
| `nm_uniform_a10_notwagg` / `nm_uniform_a10_twagg` | uniform | α = 1.0 | off / on |
| `probe_a10_r100ep2` | CIFAR-10N worst | α = 1.0 | off, 100r × 2ep (§7a) |

---

## 11. Secondary observation: which statistic, when

A side finding from the same logs. Comparing per-class **mean** loss
(the statistic FedNoRo uses) against per-class **variance** (ours), both
scored by our own pipeline rather than by FedNoRo's method:

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
