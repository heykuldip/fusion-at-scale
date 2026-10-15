#!/bin/bash
#SBATCH --job-name=nccl_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# The environment record, the GPU topology, and an all-reduce correctness and bandwidth check at each GPU count in NS.
#   sbatch -p <partition> slurm/nccl_smoke.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${ENV_ACTIVATE:-.venv/bin/activate}"
: "${NS:=2 4}"
mkdir -p results/env results/nccl
host=$(hostname -s)
python scripts/check_env.py > "results/env/${host}-${SLURM_JOB_ID}.json"
nvidia-smi topo -m > "results/env/${host}-topo-${SLURM_JOB_ID}.txt"
export NCCL_DEBUG=WARN
for N in $NS; do
  python -m torch.distributed.run --standalone --nproc_per_node="$N" -m fusion_scale.bench.nccl_smoke \
    --out "results/nccl/${host}-n${N}-${SLURM_JOB_ID}.json"
done
