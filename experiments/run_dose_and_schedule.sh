#!/usr/bin/env bash
# Two follow-ups to the trust-weighting negative result, both at a FIXED
# partition (Dirichlet 1.0) and a FIXED signal, so the only thing that moves
# is the knob under test.
#
# WHY THESE TWO
#
# The finding so far: trust weighting helps under IID (+0.0125 .. +0.0166,
# t=11-14, 5/5 seeds) and hurts under every Dirichlet partition tried. The
# proposed explanation -- that the harm grows with how hard the weighting
# pulls -- currently rests on ONE pair of runs (v3 vs v5) that differ in the
# trust SIGNAL, not in the pull strength, so signal and dose are confounded.
# And its headline t-statistic is endpoint-dependent: -3.30 on the final
# round, -2.18 on the last 3, -1.81 on the last 5, -3.02 on the last 10.
# Direction is negative under all four, significance under only two.
#
# PHASE 1 (dose) replaces that pair with a controlled sweep: one signal, one
#   partition, strength varied alone. If accuracy falls monotonically as
#   strength rises, the dose-response claim is earned. If it is flat between
#   0.25 and 1.0, the harm is a threshold effect, not a dose effect, and the
#   writeup has to say so.
#
# PHASE 2 (schedule) tests the one design choice every published method makes
#   differently from us. FedCNI (ICME 2023, Sec. III-E) aggregates by data
#   size until round Ts=15 of 100, then switches to clean-count weighting.
#   FedLAW (ICLR 2026) does the reverse -- learns weights for the first 20
#   rounds, then freezes them. Both say WHEN matters; we have always weighted
#   from round 1. Two arms: round 6 (FedCNI's 15% of training, scaled to our
#   40 rounds) and round 15 (their absolute warm-up length).
#
# WHAT THE ENDPOINTS COST
# strength=1.0 and the weighting-off baseline are ALREADY on disk as
# results/sweep_a10_twagg and results/sweep_a10_notwagg, so they are not
# re-run. Phase 0 is a one-job validity check: strength=0.0 must land on the
# existing notwagg numbers. The strategy test asserts it is bit-identical at
# round 1; this checks it end-to-end over 40 rounds. Skip it only if time is
# short.
#
# Usage:   bash experiments/run_dose_and_schedule.sh
# Resume:  re-run the same command; every job passes --skip-completed.
#
# Expect roughly 1.5 h per job on an M-series Mac (5.8 min/run x 15 runs),
# so about 7.5 h for all five. Phase 1 alone is about 3 h and is the half
# that matters most.

set -u   # deliberately NOT -e: one failed job must not abandon the rest

PY="${PY:-.venv/bin/python3}"
C10N="${C10N:-data/cifar10n/CIFAR-10_human.pt}"
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"

LOG_DIR="results/_dose_logs"
mkdir -p "$LOG_DIR"

# Identical to the arms already on disk (sweep_a10_*), so the new points are
# directly comparable with them rather than forming a separate family.
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
  --seeds 1,2,3,4,5
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

echo "Batch started $(date). Logs in $LOG_DIR/"
echo "Python: $PY"
BATCH_T0=$SECONDS

# ---------------------------------------------------------------------------
# PHASE 0 -- validity check. strength=0.0 must reproduce weighting-off.
# If this does NOT match results/sweep_a10_notwagg, something other than the
# aggregation weight differs between the arms and every strength point is
# suspect. Cheapest possible way to find that out before trusting the sweep.
# ---------------------------------------------------------------------------
run_job "p0_dose_000_selfcheck" \
  --trust-weight-strength 0.0 \
  --results-dir results/dose_a10_str000

# ---------------------------------------------------------------------------
# PHASE 1 -- the dose sweep. Interior points only; 0.0 is Phase 0 and 1.0 is
# already on disk as results/sweep_a10_twagg.
# ---------------------------------------------------------------------------
run_job "p1_dose_025" \
  --trust-weight-strength 0.25 \
  --results-dir results/dose_a10_str025

run_job "p1_dose_050" \
  --trust-weight-strength 0.5 \
  --results-dir results/dose_a10_str050

# ---------------------------------------------------------------------------
# PHASE 2 -- the schedule. Full strength throughout, switched on late.
# ---------------------------------------------------------------------------
run_job "p2_twstart_06" \
  --trust-weighting-start-round 6 \
  --results-dir results/sched_a10_start06

run_job "p2_twstart_15" \
  --trust-weighting-start-round 15 \
  --results-dir results/sched_a10_start15

echo ""
echo "=============================================================="
printf 'Batch finished %s  (total %d min)\n' "$(date '+%Y-%m-%d %H:%M:%S')" $(( (SECONDS - BATCH_T0) / 60 ))
echo "=============================================================="
echo "Completed result directories:"
for d in results/dose_a10_* results/sched_a10_*; do
  [ -d "$d" ] && printf '  %-40s %s runs\n' "$d" "$(find "$d" -name pipeline_results.json | wc -l | tr -d ' ')"
done
echo ""
echo "Each directory should hold 15 runs (3 arms x 5 seeds)."
echo "Endpoints already on disk, not re-run:"
echo "  strength 0 (reference) : results/sweep_a10_notwagg"
echo "  strength 1            : results/sweep_a10_twagg"
echo ""
echo "Commit and push results/, then ask Claude to pull and analyse."
