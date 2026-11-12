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
