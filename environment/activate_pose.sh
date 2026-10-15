# Activates the pose environment (POSE_ENV, default .pose-env): its Python, and the FFmpeg and NPP libraries TorchCodec
# loads. Source it from the repository root, in its own shell or job.
: "${POSE_ENV:=$PWD/.pose-env}"
if [ ! -x "$POSE_ENV/bin/python" ]; then
  echo "No environment at $POSE_ENV; run environment/create_pose_env.sh first" >&2
  return 1 2>/dev/null || exit 1
fi
export POSE_ENV
export PATH="$POSE_ENV/bin:$PATH"
_npp="$("$POSE_ENV/bin/python" -c "import nvidia.npp, os; print(os.path.join(list(nvidia.npp.__path__)[0], 'lib'))" 2>/dev/null)"
export LD_LIBRARY_PATH="${_npp:+$_npp:}$POSE_ENV/lib:${LD_LIBRARY_PATH:-}"
unset _npp
