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
