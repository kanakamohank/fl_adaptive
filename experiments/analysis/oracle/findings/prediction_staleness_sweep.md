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
# SETTLED at 10 seeds, all three arms (2026-11-13)

The full sweep ran on the user's machine: three arms (8/20, 12/30, 40/40),
10 seeds each, against `results/sweep_a10_twagg` (caps 4/10). Reproduce:

    python3 experiments/analysis/oracle/staleness.py

**Correction notice.** A first pass at this analysis reported that the
limiter was "trust eligibility" -- `theta_high` plus the `clean_streak`
gate -- and that the weight budget did not bind. Both claims were wrong,
for two independent reasons, and the section they appeared in has been
removed rather than amended. The errors are documented at the end of this
file, because each is a reasoning trap that will recur.

## The arms are comparable, and the cap really was switched off

Every arm differs from the baseline in exactly two config fields,
`s_max_appearances` and `s_max_rounds`. All four families have the same 10
seeds in a single config subdirectory.

The cap's firing rate is logged, so this is measured. One subtlety first: a
client that has never been verified is stale **by definition**
(`is_stale` returns True when `last_verified_round` is unset), so its
first-ever appearance counts as forced-stale at any cap setting. That floor
is exactly one event per distinct client -- 50 of 400 client-rounds, 12.5%.

| caps | skip | verified | forced-stale (raw) | net of the floor |
|---|---|---|---|---|
| 4 / 10 (baseline) | 32.1% | 67.9% | 27.9% | **15.4%** |
| 8 / 20 | 32.1% | 67.9% | 13.9% | 1.4% |
| 12 / 30 | 31.9% | 68.1% | 12.7% | 0.1% |
| 40 / 40 (no expiry) | 32.1% | 67.9% | 12.5% | **0.0%** |

Monotone to zero. At the default the cap forced re-verification on 15.4% of
client-rounds; in the no-expiry arm it never fires. The arms did their job.

## The skip rate did not move

Paired per seed, n = 10, bar |t| > 2.26:

| caps | change in skip rate | 95% CI | t | seeds up |
|---|---|---|---|---|
| 8 / 20 | +0.0 points | ±0.3 | +0.19 | 5/10 |
| 12 / 30 | −0.2 points | ±0.4 | −1.05 | 2/10 |
| 40 / 40 | +0.0 points | ±0.4 | +0.00 | 3/10 |

No effect, with a CI tight enough to exclude half a point.

## Neither falsifier fires as written -- and the answer is the budget

F1 needed a skip rate >= 50%: it is 32.1%. F2 needed the skip rate
unchanged **and** gamma not binding. The skip rate is unchanged, but gamma
binds hard. Measured **before** the demotion loop:

| caps | gamma pre | max | post | over the cap | demoted/seed |
|---|---|---|---|---|---|
| 8 / 20 | 0.801 | 1.000 | 0.281 | 345/380 (91%) | 196.3 |
| 12 / 30 | 0.809 | 1.000 | 0.281 | 344/380 (91%) | 199.2 |
| 40 / 40 | 0.815 | 1.000 | 0.282 | 343/380 (90%) | 200.4 |

Against a `gamma_budget` of 0.35, and exceeding it in **90% of rounds**.
Rounds whose tentative pool is empty are excluded: `sum_p = 0` makes their
gamma a structural 0, a round where the budget cannot bind, and averaging
them in understated the figure (0.775 against 0.815).

**The weight budget is the binding limiter.** The prediction's "second
ceiling" turns out to be the first and only one; the staleness caps never
got near it.

## What actually decides each client-round

Phase 1 of `schedule_verifications` is, in order:

    if is_stale(c)         -> V    (hard expiry; never-verified counts)
    elif t_eff < theta_low -> V    (Tier 1)
    elif t_eff >= theta_high and streak >= k_trust:
                              V if the decoy roll hits else P   (Tier 3)
    else                   -> P    (Tier 2, TENTATIVE)

**Tier 2 is the final `else`, and it promotes.** A client under
`theta_high`, or with a short streak, is skipped anyway. Phase 2 then
demotes from the tentative pool, lowest effective trust first, until
`gamma <= gamma_budget`. Per seed, out of 400 sampled client-rounds (40/40
arm):

| branch | client-rounds |
|---|---|
| Tier 3, promoted | 74.8 |
| Tier 2, tentatively promoted | 251.2 |
| stale -> verify (all first appearances) | 50.0 |
| Tier 1, trust below theta_low = 0.3 | **10.7** |
| drawn as a decoy | 13.3 |
| **demoted by the weight budget** | **200.4** |
| = actually skipped | 125.6 |

Tier 1 blocks only 10.7 client-rounds. The budget blocks 200.4. That is the
whole story, and it is the reverse of what the removed section claimed.

Replay fidelity: the replay reconstructs trust from the round-keyed log but
replays the streak and both staleness clocks, so it reproduces the promoted
**count** to within 2.2% (125.6 vs 128.4 logged) while matching the exact
promoted **set** in 101 of 400 rounds. Counts are what the table reports.

## Why the null, as a bookkeeping identity

If the budget binds, relaxing the staleness cap *cannot* raise the skip
rate: clients the cap stops forcing into V enter the tentative pool, and
the demotion loop sends the surplus straight back. From the 8/20 arm to the
40/40 arm:

| quantity | change |
|---|---|
| stale -> verify | −5.5 |
| tentatively promoted | +4.6 |
| demoted by the budget | +4.1 |
| **actually skipped** | **+0.5** |

The cap released 5.5 client-rounds; the budget reclaimed 4.1 of them. That
is a structural count, not a noisy accuracy difference, and it explains the
null in TEST 1 completely.

## The streak gate: real, but it gates TIER, not skipping

`clean_streak` resets whenever the detector flags the client **or** the
behaviour score is <= 0.8, and under the 0.6-1.5 rescale window a score
above 0.8 needs `pretrain_loss_var < 0.780`. It is reset on **33.5% of all
verifications** (19.9% by the score rule, 13.6% by the detector) and only
24.4% of checks leave a client standing at streak >= 3.

That 33.5% is close to the 40% noisy fraction, which is the reassuring
reading: the gate mostly resets the clients it exists to catch. But a short
streak demotes a client from Tier 3 to Tier 2, and **Tier 2 still
promotes** -- so the cost is losing the decoy exemption and going first in
line for budget demotion, not being blocked. Calling it a skip-rate gate
was the error.

## Accuracy: nothing moves

Four endpoints, paired, n = 10, against round-to-round jitter of 0.0239:

| caps | final | last3 | last5 | last10 | cleared |
|---|---|---|---|---|---|
| 8 / 20 | −0.0001 | −0.0020 | −0.0019 | −0.0014 | 0/4 |
| 12 / 30 | +0.0021 | +0.0016 | +0.0008 | +0.0015 | 0/4 |
| 40 / 40 | +0.0019 | +0.0004 | −0.0009 | +0.0012 | 0/4 |

CIs on last10 are ±0.0018 to ±0.0029 -- real nulls, not underpowered ones.
Two of three arms fail the seeds-1-5 / 6-10 batch-agreement check, further
reason to read these as noise around zero.

## What this does and does not license

It does **not** license removing the staleness cap. `byzantine_fraction` is
0 in every run. The cap stops a client that once earned trust from coasting
unchecked -- an anti-drift / anti-Sybil guard. Costing nothing *without* an
attacker says nothing about its value *with* one. What the sweep does
establish is that the cap is not what holds the skip rate at 32%, so
relaxing it is not the route to skipping more.

## The next experiment

**`gamma_budget`.** It is the binding constraint, it is a single float, and
pre-demotion gamma of 0.815 against a 0.35 cap says there is a lot of room.
Sweep it at 0.45 / 0.55 / 0.70 and the skip rate should move for the first
time in this project. That is also the experiment with a real safety
story to tell, because the budget is exactly the knob that bounds how much
aggregate weight unverified clients may hold -- so it needs a Byzantine arm
alongside it, not just a clean one.

`theta_low` is the only other gate with any force, and at 10.7 client-rounds
it has little to give. `theta_high` and `k_trust` move clients between Tier
2 and Tier 3, which changes decoy exposure and demotion order but not
eligibility to skip.

## The two errors in the removed section, kept as warnings

**1. Measuring a constraint by its own output.** The first pass computed
gamma from the *logged* promoted and verified sets -- that is, after the
demotion loop. The loop runs until `gamma <= gamma_budget`, so the post
value is under the cap by construction, however hard the budget bit. It
read 0.268 against a 0.35 cap and concluded "gamma does not bind". The
pre-demotion value is 0.815, over the cap in 90% of rounds. The lesson: to
ask whether a constraint binds, measure its input, never its output.

Why gamma lands at 0.268 rather than just under 0.35: the loop demotes in
ascending trust order, so the last client removed is the highest-trust one
still in the pool and carries a large posterior weight. The final step
overshoots well past the cap.

**2. Using the Tier-3 condition as the condition for skipping.** The first
pass treated `t_eff >= theta_high AND clean_streak >= k_trust` as what a
client needs to be skipped, and reported everything failing it as
"blocked": 140 client-rounds on the trust bar, 81 on the streak. Those were
Tier-2 clients the scheduler was skipping happily. `test_trust_by_round_
logging.py`'s own docstring already said Tier 2 is tentatively promoted,
and the attribution contradicted it. The lesson: replay the branch order
from the source, and check that the replayed outcome reproduces the logged
one -- which is now reported as `set_match` on every run.
