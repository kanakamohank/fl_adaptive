#!/usr/bin/env bash
# Does the harm come from clients crushed toward ZERO weight, or from the
# weighting dose itself?
#
# THE QUESTION
# The strength sweep showed the harm is super-linear: strength 0.25 and 0.50
# cost nothing, strength 1.00 costs about -0.007 at the last10 endpoint. But
# the accepted-client weight is
#
#     max(floor, 1 - strength*(1 - score))        floor = 0.05
#
# so the floor can only bind when strength > 1 - floor = 0.95. At strength
# 0.50 the smallest possible multiplier is already 0.50; at 0.25 it is 0.75.
# Full strength is therefore the ONLY arm in that sweep where any accepted
# client's weight approaches zero, which leaves two readings indistinguishable:
#
#   (a) harm grows super-linearly with how hard the weighting pulls
#   (b) harm comes specifically from the few clients crushed toward zero
#
# Holding strength at 1.00 and RAISING the floor separates them. Under (a)
# the floor should not matter. Under (b) the harm should shrink as the floor
# rises.
#
# WHY (b) IS THE INTERESTING ANSWER
# It is exactly how the published methods differ from us. FedCNI (ICME 2023)
# and FedClean (ICML 2025) weight a client by its count of detected-clean
# samples, so a client with 40% label noise keeps roughly 0.60 weight and can
# never be silenced. Our score can take the same client to 0.05 -- about 12x
# more aggressive. Floor 0.50 is the arm that most closely matches their
# calibration.
#
# WHAT THIS CANNOT ANSWER
# The floor applies to ACCEPTED clients only. BVD-flagged clients stay
# unfloored and can still reach zero weight, deliberately, so the Byzantine
# path is untouched and this sweep carries no security cost (unlike
# --trust-weight-strength, which does). If the harm survives a high floor,
# flagged clients become the remaining candidate and need their own arm.
#
# ARMS (strength 1.00 throughout; only the floor moves)
#   floor 0.05  already on disk as results/sweep_a10_twagg -- NOT re-run
#   floor 0.25  mild bound
#   floor 0.50  matches FedCNI/FedClean calibration
#   floor 1.00  no inlier down-weighting at all. A control: it isolates what
#               the OUTLIER branch alone costs, since that path is still live.
#               If this arm still shows harm, the inlier weighting was never
#               the cause.
#
# The weighting-OFF reference is also already on disk
# (results/sweep_a10_notwagg, 10 seeds) and is not re-run.
#
# Usage:   bash experiments/run_floor_sweep.sh
# Resume:  re-run it -- every job passes --skip-completed.
# Seeds:   SEEDS=1,2,3,4,5 bash ... to halve the time for a first look.
#
# Only the tavs_skip arm is trained, because that is the only arm the
# analysis reads. About 30 min per job, so roughly 1.5 h for all three at 10
# seeds.

set -u   # deliberately NOT -e: one failed job must not abandon the rest

PY="${PY:-.venv/bin/python3}"
C10N="${C10N:-data/cifar10n/CIFAR-10_human.pt}"
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"

SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10}"
ARMS="${ARMS:-tavs_skip}"
LOG_DIR="results/_floor_logs"

# Fail fast rather than letting every job die in 0 min while the summary
# still prints green.
if [ ! -x "$PY" ]; then
  echo "ERROR: python not executable at '$PY'. Set PY, e.g. PY=python3 bash $0" >&2
  exit 1
fi
if [ ! -f "$C10N" ]; then
  echo "ERROR: CIFAR-10N labels not found at '$C10N'. Set C10N." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# Byte-identical to the strength sweep's block except --seeds and the floor,
# so these arms are directly comparable with sweep_a10_twagg (floor 0.05).
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
  --trust-weight-strength 1.0
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

# Ordered so a short night still answers the question. floor 0.50 first: it
# is the arm matched to the published methods' calibration and the one most
# likely to move the result.
run_job "floor_050" \
  --trust-weight-floor 0.50 \
  --results-dir results/floor_a10_f050

run_job "floor_025" \
  --trust-weight-floor 0.25 \
  --results-dir results/floor_a10_f025

run_job "floor_100" \
  --trust-weight-floor 1.00 \
  --results-dir results/floor_a10_f100

echo ""
echo "=============================================================="
printf 'Batch finished %s  (total %d min)\n' "$(date '+%Y-%m-%d %H:%M:%S')" $(( (SECONDS - BATCH_T0) / 60 ))
echo "=============================================================="
echo "tavs_skip seed count per config subdirectory (want ONE line per arm):"
for d in results/floor_a10_f025 results/floor_a10_f050 results/floor_a10_f100; do
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
echo "  floor 0.05 (current) : results/sweep_a10_twagg"
echo "  weighting OFF        : results/sweep_a10_notwagg"
echo ""
echo "Commit and push results/, then ask Claude to pull and analyse."
