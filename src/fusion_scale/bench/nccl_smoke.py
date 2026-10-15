"""NCCL smoke test: is all-reduce correct across N processes, and roughly how fast?

An environment check, separate from the scaling benchmark. Run with torchrun:
  GPUs:  python -m torch.distributed.run --standalone --nproc_per_node=N -m fusion_scale.bench.nccl_smoke \
             --out results/nccl/n4.json
  CPU:   python -m torch.distributed.run --standalone --nproc_per_node=2 -m fusion_scale.bench.nccl_smoke --backend gloo
"""
import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--size-mb", type=float, default=None, help="buffer size; default 256 MB (nccl) or 8 MB (gloo)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    use_cuda = args.backend == "nccl"
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()

    # 1) Correctness: rank r contributes r + 1, so each rank should end with world * (world + 1) / 2.
    t = torch.full((1024,), float(rank + 1), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ok = bool(torch.allclose(t, torch.full_like(t, world * (world + 1) / 2)))
    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)  # passes only if every rank passed

    # 2) Rough bandwidth on a large buffer (zeros, so repeated sums stay finite).
    size_mb = args.size_mb if args.size_mb is not None else (256.0 if use_cuda else 8.0)
    n = int(size_mb * 1024 * 1024 / 4)
    buf = torch.zeros(n, dtype=torch.float32, device=device)
    for _ in range(args.warmup):
        dist.all_reduce(buf)
    if use_cuda:
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(buf)
    if use_cuda:
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / args.iters
    algbw = n * 4 / seconds / 1e9                    # GB/s of data reduced per call
    busbw = algbw * 2 * (world - 1) / world          # nccl-tests' bus-bandwidth convention for all-reduce

    if rank == 0:
        result = {
            "test": "nccl_smoke", "host": socket.gethostname(), "backend": args.backend, "world_size": world,
            "correct": bool(flag.item()), "size_mb": size_mb, "iters": args.iters,
            "seconds_per_allreduce": seconds, "algbw_GBps": algbw, "busbw_GBps": busbw,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(local_rank) if use_cuda else "cpu",
        }
        print(json.dumps(result, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
