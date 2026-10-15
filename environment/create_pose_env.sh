#!/bin/bash
# Creates the pose environment at POSE_ENV (default .pose-env): Python 3.11 and FFmpeg 7.1 from conda-forge, the
# packages in requirements/pose.txt, then TensorRT's packages without their dependencies. Needs conda and internet
# access; run it from the repository root, then: source environment/activate_pose.sh
set -euo pipefail
POSE_ENV=${POSE_ENV:-$PWD/.pose-env}
CONDA=${CONDA:-conda}
if [[ -e "$POSE_ENV" ]]; then echo "$POSE_ENV exists: remove it or set POSE_ENV" >&2; exit 1; fi
"$CONDA" create --yes --quiet --prefix "$POSE_ENV" --override-channels --channel conda-forge python=3.11 ffmpeg=7.1
"$POSE_ENV/bin/python" -m pip install --quiet -r requirements/pose.txt
"$POSE_ENV/bin/python" -m pip install --quiet --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements/tensorrt.txt
"$POSE_ENV/bin/python" -m pip install --quiet --no-deps -e .
echo "Ready at $POSE_ENV. Activate with: source environment/activate_pose.sh"
