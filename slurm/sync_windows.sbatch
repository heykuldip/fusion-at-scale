#!/bin/bash
#SBATCH --job-name=sync_windows
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# On a CPU node: build the window index from VIDIMU (VIDIMU_DIR), then time the loader with and without video.
#   sbatch -p <partition> slurm/sync_windows.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this repository's code first
source "${ENV_ACTIVATE:-.venv/bin/activate}"
: "${VIDIMU_DIR:=data/vidimu}" "${VIDEO_SIZE:=small}" "${LOADER_ARGS:=}"
host=$(hostname -s)
python scripts/build_windows.py --root "$VIDIMU_DIR" --video-size "$VIDEO_SIZE"
python scripts/loader_timing.py --root "$VIDIMU_DIR" --video-size "$VIDEO_SIZE" \
  --index "results/sync_windows/windows-${host}.csv" $LOADER_ARGS
