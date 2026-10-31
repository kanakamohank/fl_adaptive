#!/usr/bin/env bash
# Overnight batch: does the IID-helps / non-IID-hurts sign flip survive a
# DIFFERENT noise model?
#
# Everything measured so far uses CIFAR-10N "worst" -- real human annotator
# errors, which have structure (annotators confuse particular class pairs).
# If the sign flip depends on that structure rather than on heterogeneity,
# the central claim is wrong. This batch replicates the 2x2 with UNIFORM
# synthetic noise at a matched effective rate, which is the cheapest
# experiment that could overturn the finding.
#
# Matched rate: CIFAR-10N worst puts ~40% wrong labels on 40% of clients
# -> ~16% of all labels wrong. Uniform noise at rate 0.4 on 40% of clients
# gives 0.4 x 0.4 = the same ~16%. Everything else is held identical to the
# existing runs so the comparison is clean.
#
# Usage:   bash experiments/run_overnight_noise_model.sh
# Resume:  re-run the same command -- every job passes --skip-completed, so
#          finished (arm, seed) pairs are reconstructed from disk instead of
#          retrained.
#
# Ordering matters: PHASE 1 is the falsification test and completes the 2x2
# on its own. PHASE 2 is a bonus that locates where the sign flips. If the
# night runs out partway, you still have a self-contained result.

set -u   # deliberately NOT -e: one failed job must not abandon the rest

PY="${PY:-.venv/bin/python3}"
C10N="${C10N:-data/cifar10n/CIFAR-10_human.pt}"
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"

LOG_DIR="results/_overnight_logs"
mkdir -p "$LOG_DIR"

COMMON=(
  --rounds 40 --local-epochs 5
  --num-clients 50 --clients-per-round 10
  --dataset cifar10 --label-noise-num-classes 10
  --noisy-client-fraction 0.4
  --trust-signal pretrain_loss_var --pretrain-loss-var-rescale 0.6,1.5
  --alpha-trust 0.7
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
# PHASE 1 -- the falsification test. Uniform noise, both partitions, OFF/ON.
# Self-contained: these four jobs alone complete the 2x2 replication.
# IID pair runs first, because that is where the current claim predicts a
# POSITIVE effect -- the harder prediction to get right by luck.
# ---------------------------------------------------------------------------
run_job "p1_uniform_iid_OFF" \
  --data-split iid \
  --label-noise-rate 0.4 --label-noise-type uniform \
  --results-dir results/nm_uniform_iid_notwagg

run_job "p1_uniform_iid_ON" \
  --data-split iid \
  --label-noise-rate 0.4 --label-noise-type uniform \
  --trust-weighted-aggregation \
  --results-dir results/nm_uniform_iid_twagg

run_job "p1_uniform_a10_OFF" \
  --data-split dirichlet --data-alpha 1.0 \
  --label-noise-rate 0.4 --label-noise-type uniform \
  --results-dir results/nm_uniform_a10_notwagg

run_job "p1_uniform_a10_ON" \
  --data-split dirichlet --data-alpha 1.0 \
  --label-noise-rate 0.4 --label-noise-type uniform \
  --trust-weighted-aggregation \
  --results-dir results/nm_uniform_a10_twagg

# ---------------------------------------------------------------------------
# PHASE 2 -- bonus. Where between IID and alpha=1.0 does the sign flip?
# Uses the ORIGINAL CIFAR-10N noise so it is comparable with the existing
# IID / a=1.0 / a=0.3 points rather than with Phase 1.
# ---------------------------------------------------------------------------
run_job "p2_c10n_a30_OFF" \
  --data-split dirichlet --data-alpha 3.0 \
  --label-noise-rate 1.0 --label-noise-type cifar10n \
  --cifar10n-path "$C10N" --cifar10n-label-set worst \
  --results-dir results/sweep_a30_notwagg

run_job "p2_c10n_a30_ON" \
  --data-split dirichlet --data-alpha 3.0 \
  --label-noise-rate 1.0 --label-noise-type cifar10n \
  --cifar10n-path "$C10N" --cifar10n-label-set worst \
  --trust-weighted-aggregation \
  --results-dir results/sweep_a30_twagg

echo ""
echo "=============================================================="
printf 'Batch finished %s  (total %d min)\n' "$(date '+%Y-%m-%d %H:%M:%S')" $(( (SECONDS - BATCH_T0) / 60 ))
echo "=============================================================="
echo "Completed result directories:"
for d in results/nm_uniform_* results/sweep_a30_*; do
  [ -d "$d" ] && printf '  %-44s %s runs\n' "$d" "$(find "$d" -name pipeline_results.json | wc -l | tr -d ' ')"
done
echo ""
echo "Each directory should hold 15 runs (3 arms x 5 seeds)."
echo "Commit and push results/, then ask Claude to pull and analyse."
