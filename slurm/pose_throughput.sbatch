#!/bin/bash
#SBATCH --job-name=pose_throughput
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# The pose pipeline's configurations on one GPU, in the pose environment (environment/activate_pose.sh). Each line of
# CONFIGS runs in a fresh process; a failing line is reported and the rest still run.
#   sbatch -p <partition> slurm/pose_throughput.sbatch
#   bash scripts/preflight.sh pose_throughput --submit
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${POSE_ACTIVATE:-environment/activate_pose.sh}"
: "${VIDIMU_DIR:=data/vidimu}"
: "${CONFIGS:=
--decoder cuda-beta --batch 1 --precision fp32
--decoder cuda-beta --batch 4 --precision fp32
--decoder cuda-beta --batch 8 --precision fp32
--decoder cuda-beta --batch 16 --precision fp32
--decoder cuda-beta --batch 32 --precision fp32
--decoder cuda-beta --batch 1 --precision bf16
--decoder cuda-beta --batch 4 --precision bf16
--decoder cuda-beta --batch 8 --precision bf16
--decoder cuda-beta --batch 16 --precision bf16
--decoder cuda-beta --batch 32 --precision bf16
--decoder cuda --batch 16 --precision bf16
--decoder cpu --batch 16 --precision bf16
--decoder cpu --mode overlapped --workers 8 --batch 16 --precision bf16
--decoder cuda-beta --batch 16 --precision fp32 --min-size 480
--decoder cuda-beta --batch 16 --precision bf16 --min-size 480
}"
mkdir -p results/pose_throughput results/env
python scripts/check_env.py > "results/env/$(hostname -s)-${SLURM_JOB_ID}.json" || true
while read -r line; do
  [ -z "$line" ] && continue
  echo "== $line"
  python scripts/pose_throughput.py --root "$VIDIMU_DIR" $line || echo "FAILED: $line"
done <<< "$CONFIGS"
