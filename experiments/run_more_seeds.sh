#!/usr/bin/env bash
# Double the seed count on every arm whose number we actually quote.
#
# WHY
# The dose/schedule batch gave n=5, and the weighting-off arm's own accuracy
# swings 0.0239 between adjacent rounds -- larger than any effect measured.
# The consequence is visible in the results: the headline harm (strength 1.0
# vs off) clears |t|>2.78 on only ONE of the four endpoints. Going to n=10
# shrinks the standard error by about 1/sqrt(2) and should either confirm the
# effect on most endpoints or kill it. Nothing else about the setup changes.
#
# PAIRING IS THE WHOLE POINT
# Every reported figure is a PAIRED per-seed difference against the
# weighting-off arm, and the analysis intersects seeds. Adding seeds 6-10 to
# one arm alone contributes nothing -- they are dropped for lack of a
# partner. So each arm here is paired with the OFF arm at the same seeds, and
# PHASE 1 adds BOTH halves of the headline comparison or neither.
#
# WHERE THE RUNS LAND
# --results-dir points at the EXISTING directories and both new flags are
# passed at their defaults, so _config_tag emits the same tag as before and
# seeds 6-10 land beside seeds 1-5 in the same config subdirectory. Verified
# against the directories on disk before writing this script. Do not change
# any other flag: a single altered flag changes the tag, the new seeds land
# in a SEPARATE subdirectory, and the analysis would then read one
# subdirectory's config while mixing both subdirectories' seeds.
#
# ORDERING
# Phase 1 alone completes the headline comparison at n=10 and is worth
# running on its own. Phase 2 restores the dose claim (super-linearity needs
# strength 0.5 and 1.0). Phase 3 restores the schedule claim. If the night
# runs short you still have a self-contained result.
#
# Usage:   bash experiments/run_more_seeds.sh
# Resume:  re-run it -- every job passes --skip-completed, so finished
#          (arm, seed) pairs are reconstructed from disk instead of retrained.
#
# Only the tavs_skip arm is trained (ARMS), because that is the only arm the
# analysis reads. At ~2 min/round-set that is about 30 min per job: ~1 h for
# Phase 1, ~2 h for Phases 2+3, ~3 h for all six. Training all three arms
# instead would take about 9 h and discard two thirds of it.

set -u   # deliberately NOT -e: one failed job must not abandon the rest

PY="${PY:-.venv/bin/python3}"
C10N="${C10N:-data/cifar10n/CIFAR-10_human.pt}"
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"

# Fail fast on a missing interpreter or dataset. Without this, `set -u`
# without `-e` lets all six jobs fail in 0 min and the script still prints
# its full green-looking summary, so the only way to notice is to open six
# logs.
if [ ! -x "$PY" ]; then
  echo "ERROR: python not executable at '$PY'." >&2
  echo "       Set PY, e.g.  PY=python3 bash $0" >&2
  exit 1
fi
if [ ! -f "$C10N" ]; then
  echo "ERROR: CIFAR-10N labels not found at '$C10N'." >&2
  echo "       Set C10N to the CIFAR-10_human.pt path." >&2
  exit 1
fi

SEEDS="${SEEDS:-6,7,8,9,10}"
ARMS="${ARMS:-tavs_skip}"
LOG_DIR="results/_more_seeds_logs"
mkdir -p "$LOG_DIR"

# Byte-for-byte the COMMON block of the original runs, except --seeds.
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
  --log-oracle-signals
  # Only the tavs_skip arm is analysed, so only it is trained. A job is
  # len(arms) x len(seeds) runs: at the default three arms, 10 of every 15
  # runs would be discarded, which is about 6 h of a 9 h batch. This also
  # bypasses the matched-rate pass, and with it the skip{rate} directory
  # whose name is the mean observed skip rate of the seeds just run -- a
  # different seed set can round to a different integer and split or
  # overwrite an earlier matched-rate record. That edge is already live:
  # sweep_a10_twagg sits in skip31 while dose_a10_str050 and
  # sched_a10_start15 sit in skip32, from rates of 0.3185-0.3210.
  # Set ARMS=full_verify,tavs_skip,random_skip to restore the full pilot.
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
echo "Python: $PY"
BATCH_T0=$SECONDS

# ---------------------------------------------------------------------------
# PHASE 1 -- the headline comparison. BOTH halves, or the new seeds are
# unpairable and contribute nothing. OFF runs first: it is the reference for
# every other arm, so if the night dies after one job this is the one to have.
# ---------------------------------------------------------------------------
run_job "p1_OFF" \
  --results-dir results/sweep_a10_notwagg

run_job "p1_ON_strength100" \
  --trust-weighted-aggregation \
  --trust-weight-strength 1.0 --trust-weighting-start-round 1 \
  --results-dir results/sweep_a10_twagg

# ---------------------------------------------------------------------------
# PHASE 2 -- the dose claim. Super-linearity is measured as
# effect(0.5) - 0.5*effect(1.0), so it needs strength 0.5 at the new seeds
# as well. strength 0.0 is NOT re-run: the Phase 0 gate already passed on
# every endpoint and it is a control, not a quoted result.
# ---------------------------------------------------------------------------
run_job "p2_strength050" \
  --trust-weighted-aggregation --trust-weight-strength 0.5 \
  --results-dir results/dose_a10_str050

run_job "p2_strength025" \
  --trust-weighted-aggregation --trust-weight-strength 0.25 \
  --results-dir results/dose_a10_str025

# ---------------------------------------------------------------------------
# PHASE 3 -- the schedule claim. Full strength, switched on late.
# ---------------------------------------------------------------------------
run_job "p3_start06" \
  --trust-weighted-aggregation --trust-weighting-start-round 6 \
  --results-dir results/sched_a10_start06

run_job "p3_start15" \
  --trust-weighted-aggregation --trust-weighting-start-round 15 \
  --results-dir results/sched_a10_start15

echo ""
echo "=============================================================="
printf 'Batch finished %s  (total %d min)\n' "$(date '+%Y-%m-%d %H:%M:%S')" $(( (SECONDS - BATCH_T0) / 60 ))
echo "=============================================================="
# Counted PER CONFIG SUBDIRECTORY, not per result directory. A plain
# recursive count spans every config subdir, so the exact failure this check
# exists to catch -- 5 seeds in one tag and 5 in another, because a flag
# differed and the tag changed -- would total 10 and look correct.
echo "tavs_skip seed count per config subdirectory (want ONE line per arm,"
echo "showing 10):"
for d in results/sweep_a10_notwagg results/sweep_a10_twagg \
         results/dose_a10_str025 results/dose_a10_str050 \
         results/sched_a10_start06 results/sched_a10_start15; do
  [ -d "$d" ] || continue
  echo "  $d"
  find "$d" -path '*tavs_skip_seed*' -name pipeline_results.json \
    | sed 's#/skip[0-9]*/tavs_skip_seed.*##' | sort | uniq -c \
    | sed 's#^#    #'
done
echo ""
echo "Two lines for one arm means the config tag changed and seeds are split"
echo "across configurations -- say so before analysing. The analysis reads"
echo "one subdirectory's config while pooling both subdirectories' seeds."
echo ""
echo "Commit and push results/, then ask Claude to pull and analyse."
