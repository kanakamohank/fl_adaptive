#!/usr/bin/env bash
# How much verification work can TAVS actually save, and when does it start
# costing accuracy?
#
# WHY THIS KNOB AND NOT THE WEIGHT BUDGET
# With 40% of clients noisy, a perfect scheduler would verify only those and
# skip the other 60%. TAVS skips 32-37%, leaving 23-28 percentage points
# unclaimed. The obvious suspect was gamma_budget (the cap on how much
# aggregate weight unverified clients may hold). It is not: reconstructing the
# scheduler's own gamma exactly from the logs puts it at 0.24-0.27 against a
# 0.35 cap, touching the cap in under 6% of rounds in every regime. Raising it
# would buy almost nothing.
#
# The staleness caps are what bind. s_max_appearances=4 and s_max_rounds=10
# force a client back to verification after 4 sampled-but-skipped rounds or 10
# rounds since its last check, REGARDLESS of how trustworthy it is. Measured
# over the same runs, that fires 2.8-3.2 times per 10-client cohort -- about
# as many clients as the scheduler manages to skip. It is the largest single
# limiter.
#
# BOTH CAPS MUST MOVE TOGETHER. They are deliberately independent: a
# rarely-sampled client trips the wall-clock cap (s_max_rounds) long before it
# accumulates appearances, and a heavily-sampled one trips the appearance cap
# first. Raising only one leaves the other binding, so each arm raises both.
#
# WHAT THE SWEEP IS FOR. Skipping a third already costs nothing measurable
# (6 of 6 configs, CIs +-0.3-1.1% of accuracy). This asks how far that goes:
# the useful result is the point at which accuracy starts to move. A flat
# curve all the way to "no expiry" is also a result -- it would say the hard
# expiry is pure cost with no measured benefit, at least with no attacker
# present.
#
# ARMS (defaults 4/10 are already on disk as results/sweep_a10_twagg and are
# NOT re-run; the weighting-OFF reference is sweep_a10_notwagg)
#   8 / 20   double both caps
#   12 / 30  triple both
#   40 / 40  effectively no expiry within a 40-round run: trust alone decides
#
# IMPORTANT CAVEAT. byzantine_fraction is 0 in these runs. The staleness cap
# exists to stop a client that once earned trust from coasting forever without
# being checked again -- an anti-Sybil / anti-drift guard. Finding it costs
# accuracy nothing HERE says nothing about its value under attack. Do not
# report a relaxed cap as safe on the strength of this sweep.
#
# Usage:   bash experiments/run_staleness_sweep.sh
# Resume:  re-run it -- every job passes --skip-completed.
# Seeds:   SEEDS=1,2,3,4,5 bash ... to halve the time for a first look.
#
# Only the tavs_skip arm is trained (the only arm the analysis reads), so
# about 30 min per job: roughly 1.5 h for all three at 10 seeds.

set -u   # deliberately NOT -e: one failed job must not abandon the rest

PY="${PY:-.venv/bin/python3}"
C10N="${C10N:-data/cifar10n/CIFAR-10_human.pt}"
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"

SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10}"
ARMS="${ARMS:-tavs_skip}"
LOG_DIR="results/_staleness_logs"

if [ ! -x "$PY" ]; then
  echo "ERROR: python not executable at '$PY'. Set PY, e.g. PY=python3 bash $0" >&2
  exit 1
fi
if [ ! -f "$C10N" ]; then
  echo "ERROR: CIFAR-10N labels not found at '$C10N'. Set C10N." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# Matches the strength/floor sweeps except --seeds and the caps, so these arms
# are directly comparable with results/sweep_a10_twagg. --trust-weight-strength
# is not passed: 1.0 is the dataclass default and is untagged, so the stored
# config and the directory name are identical either way (verified against the
# on-disk tag).
COMMON=(
  --rounds 40 --local-epochs 5
  --num-clients 50 --clients-per-round 10
  --dataset cifar10 --label-noise-num-classes 10
  --data-split dirichlet --data-alpha 1.0
  --label-noise-rate 1.0 --label-noise-type cifar10n
  --cifar10n-path "$C10N" --cifar10n-label-set worst
  --noisy-client-fraction 0.4
  --trust-signal pretrain_loss_var --pretrain-loss-var-rescale 0.6,1.5
  --alpha-trust 0.7
  --trust-weighted-aggregation
  --log-oracle-signals
  --arms "$ARMS"
  --seeds "$SEEDS"
  --skip-completed
)

run_job () {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  echo ""
  echo "=============================================================="
  echo ">>> $name            started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================================="
  local t0=$SECONDS
  if "$PY" experiments/pilot_skip_comparison.py "${COMMON[@]}" "$@" >"$log" 2>&1; then
    printf '<<< %s  DONE in %d min\n' "$name" $(( (SECONDS - t0) / 60 ))
  else
    printf '<<< %s  FAILED after %d min -- see %s\n' "$name" $(( (SECONDS - t0) / 60 )) "$log"
    tail -n 15 "$log" | sed 's/^/      /'
  fi
}

echo "Batch started $(date). Seeds: $SEEDS. Arms: $ARMS. Logs in $LOG_DIR/"
BATCH_T0=$SECONDS

# Ordered so a short night still answers the question: the no-expiry arm is
# the one that brackets the whole effect. If it costs nothing, nothing in
# between can cost anything either.
run_job "stale_40_40_no_expiry" \
  --s-max-appearances 40 --s-max-rounds 40 \
  --results-dir results/stale_a10_sa40_sr40

run_job "stale_08_20_double" \
  --s-max-appearances 8 --s-max-rounds 20 \
  --results-dir results/stale_a10_sa08_sr20

run_job "stale_12_30_triple" \
  --s-max-appearances 12 --s-max-rounds 30 \
  --results-dir results/stale_a10_sa12_sr30

echo ""
echo "=============================================================="
printf 'Batch finished %s  (total %d min)\n' "$(date '+%Y-%m-%d %H:%M:%S')" $(( (SECONDS - BATCH_T0) / 60 ))
echo "=============================================================="
echo "tavs_skip seed count per config subdirectory (want ONE line per arm):"
for d in results/stale_a10_sa40_sr40 results/stale_a10_sa08_sr20 results/stale_a10_sa12_sr30; do
  [ -d "$d" ] || continue
  echo "  $d"
  find "$d" -path '*tavs_skip_seed*' -name pipeline_results.json \
    | sed 's#/skip[0-9]*/tavs_skip_seed.*##' | sort | uniq -c | sed 's#^#    #'
done
echo ""
echo "Two lines for one arm means the config tag changed and seeds are split"
echo "across configurations -- say so before analysing."
echo ""
echo "Already on disk, not re-run:"
echo "  caps 4 / 10 (current)  : results/sweep_a10_twagg"
echo "  weighting OFF reference: results/sweep_a10_notwagg"
echo ""
echo "Commit and push results/, then ask Claude to pull and analyse."
