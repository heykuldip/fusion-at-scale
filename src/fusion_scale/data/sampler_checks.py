"""Does each epoch see every sample exactly once across processes, and what changes with drop_last and set_epoch?

For several dataset sizes, each process builds a DistributedSampler, the indices of every process are gathered for
three epochs, and what each epoch covered is compared with what DistributedSampler's documentation and source say.

    python -m torch.distributed.run --standalone --nproc_per_node=3 -m fusion_scale.data.sampler_checks --backend gloo

Rank 0 prints one line per case and writes results/sampler_checks/<host>-n<N>.json.
"""
import argparse
import json
import os
import socket

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from collections import Counter

SIZES = (12, 13, 10, 7)    # 12 divides by 2, 3 and 4; the others leave remainders of every size at some N
EPOCHS = 3
SEED = 0


class IndexDataset(Dataset):
    """A dataset whose i-th sample is i itself, so the indices a sampler yields are the samples."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def epoch_indices(sampler: DistributedSampler, epoch: int, call_set_epoch: bool) -> list:
    """The indices this process's sampler yields in one epoch, in order."""
    if call_set_epoch:
        sampler.set_epoch(epoch)
    return list(iter(sampler))


def summarize(per_rank: list, n: int) -> dict:
    """What one epoch covered, across all processes: distinct indices seen (covered), appearances beyond each index's
    first (repeated), indices not seen (left_out), and each process's list length (per_rank_len).
    """
    counts = Counter(index for rank_indices in per_rank for index in rank_indices)
    covered = len(counts)
    repeated = sum(count - 1 for count in counts.values())
    # an index seen k times counts as k - 1 repeats
    left_out = n - covered

    return {
        "covered": covered,
        "repeated": repeated,
        "left_out": left_out,
        "per_rank_len": [len(indices) for indices in per_rank],
    }


def expected(n: int, world_size: int, drop_last: bool) -> dict:
    """What DistributedSampler's documentation and source say one epoch gives, in the form summarize() returns."""
    # as in DistributedSampler: ceil((n - N) / N) with drop_last, otherwise ceil(n / N)
    if drop_last and n % world_size != 0:
        num_samples = (n - world_size + world_size - 1) // world_size
    else:
        num_samples = (n + world_size - 1) // world_size

    total_size = num_samples * world_size

    if drop_last:
        covered = total_size
        repeated = 0
        left_out = n - total_size
    else:
        covered = n
        repeated = total_size - n
        left_out = 0

    return {
        "covered": covered,
        "repeated": repeated,
        "left_out": left_out,
        "per_rank_len": [num_samples] * world_size,
    }


def run_case(n: int, world: int, rank: int, drop_last: bool, call_set_epoch: bool, gather) -> list:
    """Every epoch's indices, from every process: a list over epochs of lists over ranks."""
    sampler = DistributedSampler(IndexDataset(n), num_replicas=world, rank=rank, shuffle=True, seed=SEED,
                                 drop_last=drop_last)
    return [gather(epoch_indices(sampler, epoch, call_set_epoch)) for epoch in range(EPOCHS)]


def evaluate(n: int, world: int, drop_last: bool, call_set_epoch: bool, epochs: list) -> dict:
    """Each epoch against expected(); whether the epochs repeat one order; and what no epoch ever reached."""
    exp = expected(n, world, drop_last)
    per_epoch = [summarize(per_rank, n) for per_rank in epochs]
    seen_ever = {i for per_rank in epochs for indices in per_rank for i in indices}
    return {
        'n': n, 'world_size': world, 'drop_last': drop_last, 'set_epoch': call_set_epoch,
        'per_epoch': per_epoch, 'expected': exp,
        'matches_docs': all(s == exp for s in per_epoch),
        'every_epoch_same_order': all(e == epochs[0] for e in epochs[1:]),
        'never_seen_in_any_epoch': n - len(seen_ever),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='gloo', choices=['gloo', 'nccl'])
    p.add_argument('--out', default=None, help='default: results/sampler_checks/<host>-n<N>.json')
    args = p.parse_args()
    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()

    def gather(obj):
        everyone = [None] * world
        dist.all_gather_object(everyone, obj)
        return everyone

    rows = []
    for n in SIZES:
        for drop_last in (False, True):
            for call_set_epoch in (True, False):
                epochs = run_case(n, world, rank, drop_last, call_set_epoch, gather)
                if rank == 0:
                    rows.append(evaluate(n, world, drop_last, call_set_epoch, epochs))
    if rank == 0:
        for r in rows:
            first = r['per_epoch'][0]
            print(f"n={r['n']:3d} N={r['world_size']} drop_last={str(r['drop_last']):5s} "
                  f"set_epoch={str(r['set_epoch']):5s} "
                  f"covered={first['covered']:3d} repeated={first['repeated']:2d} left_out={first['left_out']:2d} "
                  f"matches_docs={r['matches_docs']} same_order_every_epoch={r['every_epoch_same_order']} "
                  f"never_seen={r['never_seen_in_any_epoch']}")
        host = socket.gethostname().split('.')[0]
        out = args.out or f'results/sampler_checks/{host}-n{world}.json'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'test': 'sampler_checks', 'host': socket.gethostname(), 'world_size': world, 'epochs': EPOCHS,
                       'seed': SEED, 'torch': torch.__version__, 'cases': rows}, f, indent=2)
        print(f'wrote {out}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
