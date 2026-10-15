"""One configuration of the pose pipeline on one GPU, written to one result file.

    python scripts/pose_throughput.py --decoder cuda-beta --batch 16 --precision bf16
    python scripts/pose_throughput.py --decoder cpu --mode overlapped --workers 8 --batch 16

Keypoints go to <keypoints>/<configuration>/, results to results/pose_throughput/<host>-<configuration>-<job>.json.
"""
import argparse
import json
import os
import socket
import time

import torch

from fusion_scale.pose.pipeline import (DECODERS, KeypointWriter, load_model, overlapped_batches, run_pipeline,
                                           sequential_batches, throughput, video_list)


def versions():
    import torchcodec
    import torchvision
    return {'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'torchcodec': getattr(torchcodec, '__version__', '?'), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('VIDIMU_DIR', 'data/vidimu'))
    p.add_argument('--subjects', nargs='*', default=['S40', 'S41', 'S42', 'S44'])
    p.add_argument('--decoder', default='cuda-beta', choices=DECODERS)
    p.add_argument('--mode', default='sequential', choices=['sequential', 'overlapped'])
    p.add_argument('--workers', type=int, default=8, help='overlapped mode: DataLoader workers decoding on the CPU')
    p.add_argument('--threads', type=int, default=1, help='FFmpeg threads per CPU decoder')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--min-size', type=int, default=800, choices=[800, 480])
    p.add_argument('--precision', default='fp32', choices=['fp32', 'bf16'])
    p.add_argument('--warmup', type=int, default=64, help='frames left out of the timing')
    p.add_argument('--frames', type=int, default=1920, help='frames timed after the warmup')
    p.add_argument('--keypoints', default='results/pose_throughput/keypoints')
    p.add_argument('--tag', default='', help='added to the configuration name')
    p.add_argument('--out', default='results/pose_throughput')
    args = p.parse_args()
    if args.mode == 'overlapped' and args.decoder != 'cpu':
        p.error('the overlapped mode decodes in CPU workers: use --decoder cpu')
    videos = video_list(args.root, subjects=args.subjects)
    if not videos:
        p.error(f'no videos for {args.subjects} under {args.root}')

    config = f'{args.decoder}-{args.mode}-b{args.batch}-{args.precision}-s{args.min_size}'
    config += f'-w{args.workers}' if args.mode == 'overlapped' else ''
    config += f'-t{args.threads}' if args.decoder == 'cpu' else ''
    config += f'-{args.tag}' if args.tag else ''
    device = torch.device('cuda')
    t = time.perf_counter()
    model = load_model(device, args.min_size)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t

    batches = (sequential_batches(videos, args.decoder, args.batch, args.threads) if args.mode == 'sequential'
               else overlapped_batches(videos, args.batch, args.workers, args.threads))
    writer = KeypointWriter(os.path.join(args.keypoints, config))
    result = run_pipeline(batches, model, device, args.precision, writer, args.warmup, args.frames)
    batches.close()                                   # stops the DataLoader's workers, if any
    rates = throughput(result)

    host, job = socket.gethostname().split('.')[0], os.environ.get('SLURM_JOB_ID', 'local')
    record = {'test': 'pose_throughput', 'host': socket.gethostname(), 'slurm_job': job, 'config': config, **vars(args),
              'videos': len(videos), 'cpus': len(os.sched_getaffinity(0)), 'model_load_s': round(load_s, 2),
              'result': result, 'rates': rates, **versions()}
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f'{host}-{config}-{job}.json')
    with open(out, 'w') as f:
        json.dump(record, f, indent=2, default=float)
    print(json.dumps({'config': config, 'rates': rates}, default=float))
    print('wrote', out)


if __name__ == '__main__':
    main()
