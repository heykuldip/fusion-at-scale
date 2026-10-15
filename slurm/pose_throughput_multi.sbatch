#!/bin/bash
#SBATCH --job-name=pose_throughput_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# The pose pipeline on several GPUs at once as independent processes, one per allocated GPU, each with its own
# subject's videos, so the node's shared CPUs, memory and disks are part of the test.
#   sbatch -p <partition> slurm/pose_throughput_multi.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${POSE_ACTIVATE:-environment/activate_pose.sh}"
: "${VIDIMU_DIR:=data/vidimu}" "${CONFIG:=--decoder cuda-beta --batch 16 --precision bf16}" "${SUBJECTS:=S40 S41 S42 S44}"
mkdir -p results/pose_throughput
read -ra subjects <<< "$SUBJECTS"
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
n=${#gpus[@]}
for i in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} python scripts/pose_throughput.py --root "$VIDIMU_DIR" $CONFIG \
    --subjects "${subjects[$i]}" --tag "gpu$i-of$n" &
done
wait
