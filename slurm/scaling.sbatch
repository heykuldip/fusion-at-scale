#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# For each GPU count in NS, REPEATS timed runs and (PROFILE=1) one profiled run. SCALING_ARGS sets the model size
# or batch, for example SCALING_ARGS="--width 1024 --depth 8" for the larger model.
#   sbatch -p <partition> slurm/scaling.sbatch
#   bash scripts/preflight.sh scaling --submit
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${ENV_ACTIVATE:-.venv/bin/activate}"
: "${NS:=1 2 4}" "${REPEATS:=3}" "${PROFILE:=1}" "${SCALING_ARGS:=}"
mkdir -p results/scaling results/env
host=$(hostname -s)
python scripts/check_env.py > "results/env/${host}-${SLURM_JOB_ID}.json"
for N in $NS; do
  for r in $(seq 1 "$REPEATS"); do
    python -m torch.distributed.run --standalone --nproc_per_node="$N" -m fusion_scale.train.scaling \
      --repeat "$r" $SCALING_ARGS --out "results/scaling/${host}-n${N}-r${r}-${SLURM_JOB_ID}.json"
  done
  if [[ "$PROFILE" == 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node="$N" -m fusion_scale.train.scaling \
      --profile $SCALING_ARGS --out "results/scaling/${host}-n${N}-profile-${SLURM_JOB_ID}.json"
  fi
done
