"""The window loader's throughput, in windows a second, with and without decoding each window's video frames.

Every trial's streams are loaded once first (timed and reported as preload_s), so the rates measure steady-state
loading rather than each worker parsing files the first time it meets a trial.

    python scripts/loader_timing.py --index results/sync_windows/windows-<host>.csv

Writes results/sync_windows/loader-<host>-<job>.json. Run build_windows.py first.
"""
import argparse
import json
import os
import socket
import time

import torch
from torch.utils.data import DataLoader

from fusion_scale.data.window_dataset import WindowDataset
from fusion_scale.data.sync_windows import read_index, synchronized_streams
from fusion_scale.data.vidimu import inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--root', default=os.environ.get('VIDIMU_DIR', 'data/vidimu'))
    p.add_argument('--video-size', default='small', choices=['small', 'full'])
    p.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4, 8])
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--video-batch', type=int, default=4,
                   help='batch with video: each window is 60 frames of 640x480 RGB, about 55 MB, so 16 windows '
                        'a batch, times the batches in flight per worker, overflows memory')
    p.add_argument('--batches', type=int, default=40)
    p.add_argument('--out', default='results/sync_windows')
    args = p.parse_args()

    windows, _ = read_index(args.index)
    trials, _ = inventory(args.root, args.video_size)
    cpus = len(os.sched_getaffinity(0))
    base = WindowDataset(windows, trials, synchronized_streams)
    preload_s = base.preload()
    print(f'loaded {len(base._cache)} trials in {preload_s:.1f} s')
    runs = []
    video_index_s = None
    for with_video in (False, True):
        if with_video:
            video_index_s = base.preload_video_index()
            print(f'indexed {len({w.key for w in windows})} videos in {video_index_s:.1f} s')
        for workers in args.workers:
            if workers > cpus:
                continue
            ds = base.sharing(with_video)
            batch_size = args.video_batch if with_video else args.batch
            spawn = with_video and workers > 0                  # workers forked after PyAV was used can crash
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                                multiprocessing_context='spawn' if spawn else None,
                                generator=torch.Generator().manual_seed(0))
            it = iter(loader)
            for _ in range(2):                                    # warmup: workers start, first trials load
                next(it)
            t = time.perf_counter()
            n = 0
            for _ in range(args.batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                n += len(batch['label'])
            dt = time.perf_counter() - t
            runs.append({'with_video': with_video, 'workers': workers, 'batch': batch_size,
                         'start': 'spawn' if spawn else 'fork', 'windows': n, 'seconds': round(dt, 3),
                         'windows_per_s': round(n / dt, 1),
                         'frames_per_s': round(60 * n / dt, 1) if with_video else None})
            print(runs[-1])
    host, job = socket.gethostname().split('.')[0], os.environ.get('SLURM_JOB_ID', 'local')
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f'loader-{host}-{job}.json')
    with open(out, 'w') as f:
        json.dump({'test': 'loader_timing', 'host': socket.gethostname(), 'cpus': cpus, 'batch': args.batch,
                   'trials_loaded': len(base._cache), 'preload_s': round(preload_s, 2),
                   'video_index_s': round(video_index_s, 2) if video_index_s is not None else None,
                   'video_size': args.video_size, 'torch': torch.__version__, 'runs': runs}, f, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
