"""How do throughput and time per step change from 1 to N GPUs, and where does the time go?

Weak scaling: the per-GPU batch is fixed, so N GPUs process N times as many samples per step, and perfect scaling
keeps the time per step constant as N grows. A step's time is taken as the slowest process's. With --profile, 20
steps after the warmup are profiled and split into data, forward, backward and optimizer phases, plus NCCL time.

    python -m torch.distributed.run --standalone --nproc_per_node=N -m fusion_scale.train.scaling \
        --out results/scaling/n4.json
"""
import argparse
import json
import os
import socket
import statistics
from contextlib import nullcontext
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fusion_scale.data.synthetic_features import SyntheticFeatures
from fusion_scale.models.features_model import FeaturesModel

PROFILE_ACTIVE = 20   # profiled steps, after the warmup


def sync(device: torch.device) -> None:
    """Wait for all queued GPU work to finish, so that a time taken next means what it says. No-op on CPU."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_steps(model, loader, sampler, optimizer, loss_fn, device, steps, warmup, bf16, profiler=None) -> list:
    """Run warmup + steps training steps and return the measured steps' times, in seconds.

    Each step's time includes the GPU's work (a synchronize ends it). The sampler's epoch is advanced whenever the
    loader runs out, so every epoch is shuffled differently. With a profiler, every step is recorded and split into
    phases.
    """
    times = []
    epoch = 0
    iterator = iter(loader)

    for step in range(warmup + steps):
        start = time.perf_counter()

        with torch.profiler.record_function('data'):
            try:
                x, y = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.profiler.record_function('forward'):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(x)
                loss = loss_fn(output, y)

        with torch.profiler.record_function('backward'):
            loss.backward()

        with torch.profiler.record_function('optimizer'):
            optimizer.step()

        if device.type == 'cuda':
            sync(device)

        elapsed = time.perf_counter() - start

        if step >= warmup:
            times.append(elapsed)

        if profiler is not None:
            profiler.step()

    return times


def profile_summary(prof, device) -> dict:
    """Milliseconds per profiled step, for each phase: CPU wall time and, on GPU, the GPU time of the phase's kernels.
    Each phase appears twice in the profiler's averages, as a CPU range and as a GPU annotation; both figures come
    from the CPU range. NCCL time is the all-reduce kernels' GPU time, which overlaps the backward pass."""
    def device_time(e):
        return getattr(e, 'device_time_total', None) or getattr(e, 'cuda_time_total', 0) or 0

    def on_cpu(e):
        return 'CPU' in str(getattr(e, 'device_type', 'CPU'))

    phases = ('data', 'forward', 'backward', 'optimizer')
    cpu, gpu, nccl = dict.fromkeys(phases, 0.0), dict.fromkeys(phases, 0.0), 0.0
    for e in prof.key_averages():
        if e.key in cpu and on_cpu(e):
            cpu[e.key] += e.cpu_time_total / 1e3 / PROFILE_ACTIVE
            gpu[e.key] += device_time(e) / 1e3 / PROFILE_ACTIVE
        elif 'nccl' in e.key.lower() and not on_cpu(e):
            nccl += device_time(e) / 1e3 / PROFILE_ACTIVE
    out = {f'{k}_cpu_ms_per_step': round(v, 3) for k, v in cpu.items()}
    if device.type == 'cuda':
        out.update({f'{k}_gpu_ms_per_step': round(v, 3) for k, v in gpu.items()})
    out['nccl_gpu_ms_per_step'] = round(nccl, 3)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'])
    p.add_argument('--per-gpu-batch', type=int, default=256)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--n-samples', type=int, default=50_000)
    p.add_argument('--workers', type=int, default=4, help='DataLoader workers per process')
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--profile', action='store_true', help='profile 20 steps after the warmup, and save a table')
    p.add_argument('--repeat', type=int, default=1, help='which repeat this run is, for the record')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    if args.backend == 'nccl':
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
    else:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)

    data = SyntheticFeatures(n=args.n_samples, seed=args.seed)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(data, batch_size=args.per_gpu_batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', drop_last=True, persistent_workers=args.workers > 0)
    model = FeaturesModel(width=args.width, depth=args.depth).to(device)
    params = sum(t.numel() for t in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == 'cuda' else None)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    ctx = (torch.profiler.profile(activities=activities,
                                  schedule=torch.profiler.schedule(wait=args.warmup, warmup=5, active=PROFILE_ACTIVE))
           if args.profile else nullcontext())
    with ctx as prof:
        times = train_steps(ddp, loader, sampler, optimizer, loss_fn, device, args.steps, args.warmup,
                            not args.no_bf16, prof)
    if len(times) != args.steps:
        raise RuntimeError(f'train_steps returned {len(times)} times for {args.steps} steps')

    # A step ends when the slowest process finishes it, so take each step's maximum across processes.
    everyone = [None] * world
    dist.all_gather_object(everyone, times)
    step = [max(per_step) for per_step in zip(*everyone)]

    if rank == 0:
        global_batch = args.per_gpu_batch * world
        mean = statistics.mean(step)
        result = {
            'test': 'scaling', 'host': socket.gethostname(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'backend': args.backend,
            'n_gpus': world, 'per_gpu_batch': args.per_gpu_batch, 'global_batch': global_batch,
            'width': args.width, 'depth': args.depth, 'params': params, 'bf16': not args.no_bf16,
            'workers': args.workers, 'steps': args.steps, 'warmup': args.warmup, 'repeat': args.repeat,
            'step_s_mean': mean, 'step_s_sd': statistics.stdev(step) if len(step) > 1 else 0.0,
            'step_s_median': statistics.median(step), 'samples_per_s': global_batch / mean,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'nccl': '.'.join(map(str, torch.cuda.nccl.version())) if device.type == 'cuda' else None,
            'profile': profile_summary(prof, device) if args.profile else None,
        }
        print(json.dumps(result, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(result, f, indent=2)
            if args.profile:
                table = prof.key_averages().table(
                    sort_by='cuda_time_total' if device.type == 'cuda' else 'cpu_time_total', row_limit=25)
                with open(args.out.replace('.json', '-table.txt'), 'w') as f:
                    f.write(table)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
