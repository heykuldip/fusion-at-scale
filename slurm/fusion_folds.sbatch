#!/bin/bash
#SBATCH --job-name=fusion_folds
#SBATCH --array=1-16%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# One leave-one-subject-out fold per array task (16 subjects), every half, model and seed, on CPUs: the models are
# small. FUSION_DATA is the file scripts/fusion_export.py writes.
#   sbatch -p <partition> slurm/fusion_folds.sbatch
#   bash scripts/preflight.sh fusion_folds --submit
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${ENV_ACTIVATE:-.venv/bin/activate}"
: "${FUSION_DATA:=results/fusion/windows.npz}" "${FUSION_ARGS:=}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/fusion_train.py --data "$FUSION_DATA" --fold "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK" \
  $FUSION_ARGS
