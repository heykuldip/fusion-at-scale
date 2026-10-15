#!/bin/bash
# Checks an experiment before its full job is submitted, stopping at the first failure:
#   1. the CPU tests, including the distributed logic checks on CPU processes (pytest -m "not gpu")
#   2. sbatch --test-only: Slurm checks the job script and request (partition, GPUs, limits) without submitting it
#   3. in a short allocation with the job's GPUs, CPUs and memory: the environment, then two short runs of the real
#      pipeline at the job's batch and model sizes, timed (wall clock and peak memory)
#   4. the estimate: the full run's time and memory, scaled from the short runs, against the job's --time and --mem
#   5. with --submit, the job itself
#
#   bash scripts/preflight.sh <experiment> [--submit]
#   SLURM_ARGS="-p <partition> -A <account>" bash scripts/preflight.sh scaling --submit
#
# Experiments are the job scripts in slurm/: ddp_equivalence, nccl_smoke, scaling, sync_windows, pose_throughput,
# pose_throughput_multi, fusion_folds and inference_latency. Settings (environment): SLURM_ARGS, passed to srun,
# sbatch --test-only and sbatch; PREFLIGHT_TIME, the short allocation's limit (default 0-00:30:00); MARGIN, the safety
# factor on time (default 1.5); SKIP_TESTS=1 skips step 1. The job's own settings (NS, CONFIGS, ...) apply as in the
# job, so the estimate is for the run that will be submitted.
set -euo pipefail
EXP=${1:?usage: bash scripts/preflight.sh <experiment> [--submit]}
shift
INSIDE=0; SUBMIT=0
for a in "$@"; do
  case "$a" in --submit) SUBMIT=1 ;; --inside) INSIDE=1 ;; *) echo "unknown option: $a" >&2; exit 2 ;; esac
done
JOB=slurm/$EXP.sbatch
[[ -f "$JOB" ]] || { echo "no job script $JOB (run from the repository root)" >&2; exit 2; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"                 # this repository's code first
eval "$(python -m fusion_scale.preflight options --job "$JOB" --shell)"   # JOB_TIME, JOB_MEM, JOB_CPUS, JOB_GPUS, settings
MARGIN=${MARGIN:-1.5}

lines() { grep -c '[^[:space:]]' <<< "$1" || true; }
first_line() { grep -m 1 '[^[:space:]]' <<< "$1"; }
largest() { tr ' ' '\n' <<< "$1" | sort -n | tail -n 1; }

# The two short runs and the full run's size, in the same units, for each experiment.
short_runs() {
  local work=$1 run="python -m fusion_scale.preflight run"
  case "$EXP" in
    ddp_equivalence|nccl_smoke)
      local module=fusion_scale.train.ddp_equivalence
      [[ "$EXP" == nccl_smoke ]] && module=fusion_scale.bench.nccl_smoke
      $run --out "$work/a.json" -- python -m torch.distributed.run --standalone --nproc_per_node="$(largest "$NS")" \
        -m "$module" --out "$work/a-result.json"
      MEASURED="$work/a.json:1"; FULL_UNITS=$(wc -w <<< "$NS"); LAUNCHES=0 ;;
    scaling)
      local n; n=$(largest "$NS")
      for steps in 10 30; do
        $run --out "$work/s$steps.json" -- python -m torch.distributed.run --standalone --nproc_per_node="$n" \
          -m fusion_scale.train.scaling $SCALING_ARGS --steps "$steps" --warmup 5 --out "$work/s$steps-result.json"
      done
      local per_run=$((200 + 20)) runs=$(( $(wc -w <<< "$NS") * (REPEATS + PROFILE) ))
      MEASURED="$work/s10.json:15 $work/s30.json:35"; FULL_UNITS=$((runs * per_run)); LAUNCHES=$runs ;;
    sync_windows)
      [[ -d "$VIDIMU_DIR" ]] || { echo "no VIDIMU at $VIDIMU_DIR (set VIDIMU_DIR)" >&2; return 1; }
      for b in 2 10; do
        $run --out "$work/b$b.json" -- bash -c "python scripts/build_windows.py --root '$VIDIMU_DIR' \
          --video-size '$VIDEO_SIZE' --out '$work/w$b' && python scripts/loader_timing.py --root '$VIDIMU_DIR' \
          --video-size '$VIDEO_SIZE' --index \"\$(ls '$work/w$b'/windows-*.csv)\" --workers 0 2 --batches $b --out '$work/w$b' $LOADER_ARGS"
      done
      MEASURED="$work/b2.json:8 $work/b10.json:40"; FULL_UNITS=$((40 * 4 * 2)); LAUNCHES=1 ;;
    pose_throughput|pose_throughput_multi)
      [[ -d "$VIDIMU_DIR" ]] || { echo "no VIDIMU at $VIDIMU_DIR (set VIDIMU_DIR)" >&2; return 1; }
      local config=${CONFIG:-$(first_line "${CONFIGS:-}")}
      for f in 32 96; do
        $run --out "$work/f$f.json" -- python scripts/pose_throughput.py --root "$VIDIMU_DIR" $config --subjects S40 \
          --warmup 16 --frames "$f" --keypoints "$work/kp" --out "$work"
      done
      local configs=1; [[ "$EXP" == pose_throughput ]] && configs=$(lines "$CONFIGS")
      MEASURED="$work/f32.json:48 $work/f96.json:112"; FULL_UNITS=$((configs * (64 + 1920))); LAUNCHES=$configs ;;
    fusion_folds)
      [[ -f "$FUSION_DATA" ]] || { echo "no $FUSION_DATA: run scripts/fusion_export.py first (or set FUSION_DATA)" >&2; return 1; }
      for e in 2 6; do
        $run --out "$work/e$e.json" -- python scripts/fusion_train.py --data "$FUSION_DATA" --fold 1 --seeds 0 \
          --epochs "$e" --out "$work/runs$e" $FUSION_ARGS
      done
      MEASURED="$work/e2.json:2 $work/e6.json:6"; FULL_UNITS=$((3 * 60)); LAUNCHES=1 ;;      # 3 seeds, up to 60 epochs
    inference_latency)
      [[ -d "$VIDIMU_DIR" ]] || { echo "no VIDIMU at $VIDIMU_DIR (set VIDIMU_DIR)" >&2; return 1; }
      for f in 32 64; do                                     # TensorRT: the slowest start-up, so the safe side
        $run --out "$work/f$f.json" -- python scripts/inference_latency.py --root "$VIDIMU_DIR" --variant trt \
          --size 800 --batch 16 --precision bf16 --frames "$f" --rounds 1 --out "$work"
      done
      local configs; configs=$(lines "$CONFIGS")
      MEASURED="$work/f32.json:96 $work/f64.json:192"; FULL_UNITS=$((configs * 256 * 5)); LAUNCHES=$configs ;;
    *) echo "unknown experiment $EXP" >&2; return 2 ;;
  esac
}

if [[ $INSIDE == 1 ]]; then
  [[ "$JOB" == *pose* || "$JOB" == *inference* ]] && source "${POSE_ACTIVATE:-environment/activate_pose.sh}"
  echo "== environment ($JOB_GPUS GPUs needed)"
  python -m fusion_scale.preflight env --gpus "$JOB_GPUS"
  work=$(mktemp -d "${TMPDIR:-/tmp}/preflight-$EXP.XXXX")
  trap 'rm -rf "$work"' EXIT
  echo "== short runs of the real pipeline"
  short_runs "$work"
  echo "== the full run against $JOB's request (--time $JOB_TIME, --mem $JOB_MEM)"
  python -m fusion_scale.preflight estimate --job "$JOB" --measured $MEASURED --full-units "$FULL_UNITS" \
    --launches "$LAUNCHES" --margin "$MARGIN"
  exit 0
fi

if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
  echo "== 1. CPU tests"
  python -m pytest -q -m "not gpu"
fi
echo "== 2. sbatch --test-only"
python -m fusion_scale.preflight test-only --job "$JOB" -- ${SLURM_ARGS:-}
echo "== 3-4. short runs and the estimate"
if [[ -n "${SLURM_JOB_ID:-}" ]] || ! command -v srun > /dev/null; then
  bash "$0" "$EXP" --inside                                    # already in an allocation, or no Slurm
else
  gres=(); [[ "$JOB_GPUS" -gt 0 ]] && gres=(--gres="gpu:$JOB_GPUS")
  srun --nodes=1 --ntasks=1 --cpus-per-task="$JOB_CPUS" ${JOB_MEM:+--mem="$JOB_MEM"} "${gres[@]}" \
    --time="${PREFLIGHT_TIME:-0-00:30:00}" ${SLURM_ARGS:-} bash "$0" "$EXP" --inside
fi
if [[ $SUBMIT == 1 ]]; then
  echo "== 5. submitting"
  sbatch ${SLURM_ARGS:-} "$JOB"
else
  echo "All checks passed. Submit with: sbatch ${SLURM_ARGS:-} $JOB"
fi
