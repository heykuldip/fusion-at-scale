#!/bin/bash
#SBATCH --job-name=ddp_equivalence
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# One DDP step against one full-batch step, at each GPU count in NS. Submit from the repository root with the
# cluster's partition and account (and a GPU type in --gres if the cluster needs one):
#   sbatch -p <partition> slurm/ddp_equivalence.sbatch
#   bash scripts/preflight.sh ddp_equivalence --submit
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${ENV_ACTIVATE:-.venv/bin/activate}"
: "${NS:=2 4}"
mkdir -p results/ddp_equivalence
for N in $NS; do
  python -m torch.distributed.run --standalone --nproc_per_node="$N" -m fusion_scale.train.ddp_equivalence \
    --out "results/ddp_equivalence/$(hostname -s)-n${N}-${SLURM_JOB_ID}.json"
done
