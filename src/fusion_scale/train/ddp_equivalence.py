"""Does one distributed data-parallel (DDP) step match the same step on a single device?

Every process builds the same model and the same global batch. Rank 0 takes one plain SGD step on the whole batch;
each process takes one DDP step on its own equal-sized shard. With a mean loss, DDP's averaged gradients equal the
full-batch gradients, so gradients and parameters should agree to within floating-point rounding.

    python -m torch.distributed.run --standalone --nproc_per_node=2 -m fusion_scale.train.ddp_equivalence --backend gloo
    python -m torch.distributed.run --standalone --nproc_per_node=N -m fusion_scale.train.ddp_equivalence \
        --out results/ddp_equivalence/n4.json
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS; set before CUDA starts

import argparse
import json
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def set_determinism(seed: int) -> None:
    """fp32 with deterministic kernels; TF32 is switched off, so fp32 stays fp32."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(seed: int) -> nn.Module:
    """A small MLP; the same seed gives the same starting weights in every process."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def make_global_batch(seed: int, global_batch: int, device: torch.device):
    """Synthetic data, generated identically in every process (CPU generator, then moved)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(global_batch, 64, generator=g)
    y = torch.randint(0, 10, (global_batch,), generator=g)
    return x.to(device), y.to(device)



def reference_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float):
    """One plain SGD step on one device with the full global batch. Returns detached copies of the gradients
    (after backward) and of the parameters (after the step).
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn(model(x), y).backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    optimizer.step()
    params = [p.detach().clone() for p in model.parameters()]
    return grads, params


def ddp_step(model: nn.Module, x_shard: torch.Tensor, y_shard: torch.Tensor, lr: float, device_ids):
    """One DDP step on this process's shard. Returns (grads, params) as reference_step does."""

    ddp_model = DDP(model, device_ids=device_ids)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn(ddp_model(x_shard), y_shard).backward()          # DDP averages the gradients during backward
    grads = [p.grad.detach().clone() for p in model.parameters()]
    optimizer.step()
    params = [p.detach().clone() for p in model.parameters()]
    return grads, params


def max_abs_diff(a, b) -> float:
    """Largest absolute elementwise difference across matching lists of tensors."""
    return max((g1 - g2).abs().max().item() for g1, g2 in zip(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--per-gpu-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-5, help="largest allowed absolute difference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    use_cuda = args.backend == "nccl"
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    set_determinism(args.seed)

    global_batch = args.per_gpu_batch * world
    x, y = make_global_batch(args.seed, global_batch, device)              # identical in every process
    lo, hi = rank * args.per_gpu_batch, (rank + 1) * args.per_gpu_batch
    x_shard, y_shard = x[lo:hi], y[lo:hi]                                    # contiguous, equal-sized shards

    ref = None
    if rank == 0:
        ref = reference_step(make_model(args.seed).to(device), x, y, args.lr)
    dist.barrier()

    ddp = ddp_step(make_model(args.seed).to(device), x_shard, y_shard, args.lr,
                   device_ids=[local_rank] if use_cuda else None)

    if rank == 0:
        d_grad = max_abs_diff(ref[0], ddp[0])
        d_param = max_abs_diff(ref[1], ddp[1])
        result = {
            "test": "ddp_equivalence", "host": socket.gethostname(), "backend": args.backend, "world_size": world,
            "per_gpu_batch": args.per_gpu_batch, "global_batch": global_batch, "lr": args.lr, "seed": args.seed,
            "max_abs_dgrad": d_grad, "max_abs_dparam": d_param, "tol": args.tol,
            "pass": bool(d_grad <= args.tol and d_param <= args.tol),
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
