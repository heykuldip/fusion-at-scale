#!/bin/bash
#SBATCH --job-name=inference_latency
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Inference variants of Keypoint R-CNN on one GPU, in the pose environment with Torch-TensorRT. Each line of CONFIGS
# runs in a fresh process, and a failure is written to that configuration's result file.
#   sbatch -p <partition> slurm/inference_latency.sbatch
#   bash scripts/preflight.sh inference_latency --submit
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${POSE_ACTIVATE:-environment/activate_pose.sh}"
: "${VIDIMU_DIR:=data/vidimu}"
: "${CONFIGS:=
--variant eager --size 800 --batch 16 --precision fp32
--variant eager --size 800 --batch 16 --precision bf16
--variant capped --size 800 --batch 16 --precision bf16
--variant compile --size 800 --batch 16 --precision bf16
--variant compile-cg --size 800 --batch 16 --precision bf16
--variant trt --size 800 --batch 16 --precision bf16
--variant eager --size 480 --batch 16 --precision bf16
--variant capped --size 480 --batch 16 --precision bf16
--variant compile --size 480 --batch 16 --precision bf16
--variant trt --size 480 --batch 16 --precision bf16
--variant eager --size 800 --batch 1 --precision bf16
--variant compile-cg --size 800 --batch 1 --precision bf16
--variant trt --size 800 --batch 1 --precision bf16
--variant eager --size 480 --batch 1 --precision bf16
--variant compile-cg --size 480 --batch 1 --precision bf16
--variant trt --size 480 --batch 1 --precision bf16
}"
mkdir -p results/inference_latency results/env
python scripts/check_env.py > "results/env/$(hostname -s)-${SLURM_JOB_ID}.json" || true
while read -r line; do
  [ -z "$line" ] && continue
  echo "== $line"
  python scripts/inference_latency.py --root "$VIDIMU_DIR" $line || echo "FAILED: $line"
done <<< "$CONFIGS"
