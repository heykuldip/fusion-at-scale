"""Print the software and GPU environment as JSON; exit non-zero if CUDA isn't usable.

Run inside a GPU job (or on a login node, where it reports no GPUs and exits 1).
"""
import json
import os
import platform
import socket
import sys

import torch


def nccl_version():
    if not torch.cuda.is_available():
        return None
    v = torch.cuda.nccl.version()
    return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)


def main():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "nccl": nccl_version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": n,
        "devices": [torch.cuda.get_device_name(i) for i in range(n)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for i in range(n):  # a tiny matmul on each GPU, to see kernels run
        a = torch.randn(1024, 1024, device=f"cuda:{i}")
        (a @ a).sum().item()
    print(json.dumps(info, indent=2))
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
