# Pre-registered prediction: the staleness sweep will move the skip rate only a few points

**Written before the results existed.** Batch started 2026-11-12 21:42 UTC
(3 arms x 5 seeds); at the time of writing, 0 of 15 runs had finished. This
file exists so the prediction can be checked against the outcome rather than
reconstructed after it.

## The question

TAVS skips 32% of server-side verification checks at no measurable accuracy
cost (6 of 6 configs, CI +-0.3-1.1% of accuracy). With 40% of clients noisy,
an oracle verifying only the noisy ones could skip 60%. The gap is ~113 of
400 verification opportunities.

Two candidate limiters were tested against the logs:

- **weight budget** (`gamma_budget = 0.35`, a cap on the share of aggregate
  weight unverified clients may hold). Reconstructing the scheduler's own
  gamma from exact replayed trust puts it at 0.24-0.27, touching the cap in
  under 6% of rounds. *Not currently binding.*
- **staleness caps** (`s_max_appearances = 4`, `s_max_rounds = 10`), which
  force re-verification regardless of trust. These fire 2.8-3.2 times per
  10-client cohort -- about as many clients as the scheduler manages to
  skip. *This is the active limiter*, which is why the sweep targets it.

## The prediction

**The skip rate will rise by a few points at most -- not to anything near
60% -- because the weight budget takes over as soon as staleness stops
binding.**

The budget permits a promoted set only while
`sum_p / (|V| + sum_p) <= 0.35`, where each promoted client's weight is
`sigmoid(8 * (trust - 0.5))`. For a 10-client cohort of confidently-trusted
clients:

| trust | weight | max promotable | skip rate |
|---|---|---|---|
| 0.70 | 0.832 | 3 | 30% |
| 0.90 | 0.961 | 3 | 30% |
| 0.99 | 0.981 | 3 | 30% |

The scheduler already sits at 3.2 of 10. So for high-trust clients it is
**already at the ceiling the budget allows**, and the expected mechanism is:
remove the staleness cap -> the scheduler tries to promote more -> gamma
exceeds 0.35 -> the demotion loop pushes them back.

This also explains an earlier oddity: gamma binds in only 4-6% of rounds
*because staleness holds promotion below the gamma ceiling anyway*. Remove
staleness and gamma starts doing the work.

## What would falsify it

- A skip rate at or above ~50% in the no-expiry (40/40) arm. The table above
  would then be wrong and the budget is not the second ceiling.
- A skip rate essentially unchanged from 32%, with gamma still not binding.
  That would mean neither cap was the limiter and the real constraint is
  something else again -- most likely trust never reaching `theta_high`.

## Known weakness in the prediction

The table assumes every promoted client shares one trust value. Real cohorts
are mixed, and a low-trust client weighs less and so packs in more -- which
is why splits of 6 promoted do occur in the logs. The true ceiling is
therefore **above 3 but well short of 6**. Treat "a few points" as the
claim, not the exact figure.

## What this sweep cannot answer

`byzantine_fraction = 0` in every arm. The staleness cap exists to stop a
client that once earned trust from coasting indefinitely without being
checked again. A finding that relaxing it costs no accuracy is measured
**without attackers** and says nothing about its value under attack. The
writeup must not blur the two.

5 seeds, not 10. Every other effect in this project needed 10 seeds before
it was conclusive.

## If the prediction holds

Neither cap can move the ceiling alone, so the informative follow-up is
raising `gamma_budget` **and** the staleness caps together.

---

## Outcome of the first (and only) completed seed: the prediction's DIRECTION held, its MECHANISM did not

The container was stopped before the sweep finished, but seed 1 of the
no-expiry arm completed. Measured against the same seed of the 4/10
baseline:

| caps | skip rate | accuracy (last10) |
|---|---|---|
| 4 / 10 | 30.8% | 0.3684 |
| 40 / 40 (no expiry) | 31.2% | 0.3642 |

**+0.5 points.** Removing the staleness expiry entirely barely moved the
skip rate -- less even than the "few points" predicted.

The predicted mechanism was that the weight budget would take over. It did
not:

| caps | mean gamma | % of rounds at the cap | clients below theta_high |
|---|---|---|---|
| 4 / 10 | 0.262 | 0% | 5.7 of 10 |
| 40 / 40 | 0.264 | 0% | 5.8 of 10 |

Gamma is unchanged and never binds. This is **falsifier #2** as written
above: *"a skip rate essentially unchanged from 32%, with gamma still not
binding ... would mean neither cap was the limiter and the real constraint
is something else again -- most likely trust never reaching theta_high."*

That is what the last column shows: about 5.8 of every 10 sampled clients
sit below theta_high = 0.7 and are not eligible for promotion at all,
regardless of either cap.

**So the limiter is trust eligibility, not the staleness cap and not the
weight budget.** The knob worth sweeping next is theta_high, or the trust
dynamics that feed it (alpha_trust, k_trust, and the behaviour score itself,
which under IID scores roughly half of all clients at 0).

**n = 1.** Every effect in this project has needed 10 seeds before it was
conclusive, and the two accuracy figures above differ by 0.0042, which is
well inside this setup's round-to-round jitter of ~0.022. Treat this as a
strong hint about the skip rate -- a structural count, far less noisy than
accuracy -- and as no evidence at all about the accuracy column.

The run that produced it was computed in the cloud container, not on the
machine every other result in this project came from. It was deleted rather
than kept: --skip-completed would have silently reused it and made a
10-seed mean out of 1 container seed and 9 Mac seeds. This project already
retracted a finding caused by two batches differing systematically on the
SAME machine; mixing machines is strictly worse. The numbers above are the
record.

---

# CONFIRMED at 10 seeds, all three arms (2026-11-13)

The full sweep ran on the user's machine. Three arms (8/20, 12/30, 40/40),
10 seeds each, against `results/sweep_a10_twagg` (caps 4/10) already on
disk. Reproduce with:

    python3 experiments/analysis/oracle/staleness.py

## The arms are comparable, and the cap really was switched off

Every arm differs from the baseline in exactly two config fields,
`s_max_appearances` and `s_max_rounds`, and in nothing else. All four
families have the same 10 seeds in a single config subdirectory.

The cap's own firing rate is logged, so this is measured rather than
assumed. One subtlety had to be handled first: a client that has NEVER been
verified is stale by definition, so its first-ever appearance is counted as
forced-stale at any cap setting. That floor is exactly one event per
distinct client -- 50 of 400 client-rounds, 12.5% -- and it must be
subtracted before the number means anything. (A first pass did not subtract
it, read the residual 12.5% as "the cap is still firing", and wrongly
declared the whole sweep uninterpretable.)

| caps | skip | verified | forced-stale (raw) | forced-stale (net of floor) |
|---|---|---|---|---|
| 4 / 10 (baseline) | 32.1% | 67.9% | 27.9% | **15.4%** |
| 8 / 20 | 32.1% | 67.9% | 13.9% | 1.4% |
| 12 / 30 | 31.9% | 68.1% | 12.7% | 0.1% |
| 40 / 40 (no expiry) | 32.1% | 67.9% | 12.5% | **0.0%** |

Clean monotone to zero. At the default the cap forced re-verification on
15.4% of client-rounds; in the no-expiry arm it never fires at all. The arms
test what they claim.

## The skip rate did not move

Paired per seed, n = 10, significance bar |t| > 2.26:

| caps | change in skip rate | 95% CI | t | seeds up |
|---|---|---|---|---|
| 8 / 20 | +0.0 points | ±0.3 | +0.19 | 5/10 |
| 12 / 30 | −0.2 points | ±0.4 | −1.05 | 2/10 |
| 40 / 40 | +0.0 points | ±0.4 | +0.00 | 3/10 |

Not a weak effect -- no effect, with a CI tight enough to exclude anything
above half a point.

## Both clauses of falsifier #2 fire

F2 required the skip rate to be unchanged **and** gamma to still not bind.
Gamma is replayed from the logs using the scheduler's own formula,
`sum_p / (|V| + sum_p)` with `p_i = sigmoid(c_lambda·(t_eff − 0.5))`:

| caps | mean gamma | max | rounds within 0.01 of the 0.35 cap |
|---|---|---|---|
| 8 / 20 | 0.269 | 0.349 | 23/400 (5.8%) |
| 12 / 30 | 0.267 | 0.349 | 14/400 (3.5%) |
| 40 / 40 | 0.268 | 0.349 | 19/400 (4.8%) |

Unchanged across arms, and nowhere near the cap. **The prediction in this
document is falsified: neither staleness cap was the limiter, and the weight
budget is not the next ceiling either.** The n=1 container result above was
right.

## What the limiter actually is -- and the n=1 note was half right

The n=1 note named trust eligibility. That is the largest single gate but
NOT the whole story. Tier 3 requires **two** conditions,
`t_eff >= theta_high` AND `clean_streak >= k_trust`, and the second blocks
almost as many client-rounds as the first. Per seed, out of 400 sampled
client-rounds in the no-expiry arm:

| outcome | client-rounds | share |
|---|---|---|
| promoted (skipped) | 128.4 | 32.1% |
| blocked: raw trust < theta_high = 0.7 | 139.6 | 34.9% |
| blocked: `clean_streak` < k_trust = 3 | 81.0 | 20.3% |
| blocked: the `tau_ramp` cap on effective trust | 6.3 | 1.6% |
| eligible but drawn as a decoy | 13.0 | 3.3% |
| cleared every gate, still not promoted (budget etc.) | 31.7 | 7.9% |

The streak gate is the finding this sweep was not looking for. It resets
whenever the detector flags the client **or** the behaviour score is <= 0.8,
and under the 0.6-1.5 rescale window a score above 0.8 requires
`pretrain_loss_var < 0.780`. Measured over the no-expiry arm, the streak is
reset on **33.5% of all verifications** (19.9% by the score rule, 13.6% by
the detector), and only **24.4%** of verifications leave a client standing
at streak >= 3.

That 33.5% is close to the 40% noisy-client fraction, which is the
reassuring reading: the gate is mostly resetting the clients it exists to
catch. But it also means a clean client needs three consecutive clean checks
to become skippable, and the EMA at `alpha_trust = 0.7` has to climb to 0.7
in parallel. Those two clocks, not the staleness cap, set the 32% ceiling.

Read with care: trust and the streak update ONLY on verification, so a
client held below either bar is partly held there *by* being skipped. This
sizes the gates; it does not settle the direction of causation.

## Accuracy: nothing moves, in either direction

Four endpoints, paired, n = 10, against a round-to-round jitter of 0.0239:

| caps | final | last3 | last5 | last10 | endpoints cleared |
|---|---|---|---|---|---|
| 8 / 20 | −0.0001 | −0.0020 | −0.0019 | −0.0014 | 0/4 |
| 12 / 30 | +0.0021 | +0.0016 | +0.0008 | +0.0015 | 0/4 |
| 40 / 40 | +0.0019 | +0.0004 | −0.0009 | +0.0012 | 0/4 |

CIs on the last10 endpoint are ±0.0018 to ±0.0029 -- tight enough to call
these real nulls rather than underpowered ones. Two of the three arms fail
the seeds-1-5 / 6-10 batch-agreement check, which is further reason to read
them as noise around zero rather than as small positive effects.

## How much authority did the cap have? (counterfactual, upper bound)

Replaying the 4/10 clocks over the no-expiry arm's realised schedule: of the
66.7 client-rounds per seed the cap would have forced to verify, **50.2
(75.3%) were blocked by the trust-or-streak gate anyway**. The cap had sole
authority over 16.5 client-rounds, 4.1% of 400.

Even that 4.1% did not materialise -- the measured change is 0.0 points with
a CI of ±0.4. The replay cannot account for decoy draws, the weight budget,
or the feedback by which extra skipping starves the very trust updates that
make a client eligible later. Treat 4.1% as a ceiling on the cap's
influence, not a prediction.

## What this does and does not license

It does **not** license removing the staleness cap. `byzantine_fraction` is
0 in every run here. The cap exists to stop a client that once earned trust
from coasting forever unchecked -- an anti-drift / anti-Sybil guard. Finding
it costs nothing *without* an attacker says nothing about its value *with*
one. What the sweep does establish is that the cap is not what is holding
the skip rate at 32%, so relaxing it is not the route to skipping more.

## The next experiment

`theta_high` and `k_trust`, together, since they are the two gates that
actually bind. `k_trust` is the cheaper and more interesting of the two: it
is an integer currently at 3, the behaviour-score reset rule at 0.8 is doing
more work than its one-line justification suggests, and dropping it to 1 or
2 is a one-field change. `theta_high` is the bigger lever but moves the
Byzantine-robustness story with it, so it needs an attacker arm to be
meaningful.
