"""One inference variant of Keypoint R-CNN, timed and checked against eager inference; one result file.

    python scripts/inference_latency.py --variant trt --size 480 --batch 16 --precision bf16

The reference for the correctness check is eager fp32 inference at the same size, on the same frames. At size 480 a
second check compares with eager fp32 at 800: not ground truth, but how far keypoints move with the frame size.
A failure is written to the result file instead of being raised, so a sweep keeps going.
"""
import argparse
import json
import os
import socket
import statistics
import traceback

import torch

from fusion_scale.pose.pipeline import video_list
from fusion_scale.pose.inference_variants import (VARIANTS, backbone_shape, compile_backbone, infer, load_frames,
                                                  load_model, time_inference)
from fusion_scale.pose.keypoint_metrics import compare, top_detection


def versions():
    import torchvision
    v = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
         'gpu': torch.cuda.get_device_name(0)}
    try:
        import tensorrt
        import torch_tensorrt
        v.update(torch_tensorrt=torch_tensorrt.__version__, tensorrt=tensorrt.__version__)
    except Exception:                         # absent, or unable to start without a GPU
        pass
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('VIDIMU_DIR', 'data/vidimu'))
    p.add_argument('--variant', default='eager', choices=VARIANTS)
    p.add_argument('--size', type=int, default=800, choices=[800, 480])
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'])
    p.add_argument('--frames', type=int, default=256, help='frames timed, split across the videos')
    p.add_argument('--videos', nargs='*', default=['S40_A01_T01', 'S40_A05_T01'])
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--capped', action='store_true', help='cap proposals and detections in any variant')
    p.add_argument('--trt-mode', default='mixed', choices=['mixed', 'fp32', 'half'],
                   help="trt only: 'mixed' lets TensorRT use FP16 kernels behind fp32 inputs")
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results/inference_latency')
    args = p.parse_args()
    variant = f'{args.variant}-{args.trt_mode}' if args.variant == 'trt' else args.variant
    variant += '-capped' if args.capped and args.variant != 'capped' else ''
    config = f'{variant}-s{args.size}-b{args.batch}-{args.precision}' + (f'-{args.tag}' if args.tag else '')
    host, job = socket.gethostname().split('.')[0], os.environ.get('SLURM_JOB_ID', 'local')
    record = {'test': 'inference_latency', 'host': socket.gethostname(), 'slurm_job': job, 'config': config,
              **vars(args)}
    try:
        record.update(versions())
        device = torch.device('cuda')
        by_name = {os.path.basename(v)[:-4]: v for v in video_list(args.root)}
        paths = [by_name[k] for k in args.videos]
        frames = load_frames(paths, args.frames // len(paths), device)
        frames = frames[:len(frames) - len(frames) % args.batch] if len(frames) >= args.batch else frames

        reference = top_detection(infer(load_model(device, args.size), frames, 16, 'fp32'))
        at_800 = reference if args.size == 800 else top_detection(infer(load_model(device, 800), frames, 16, 'fp32'))
        torch.cuda.empty_cache()

        model = load_model(device, args.size, capped=args.capped or args.variant == 'capped')
        compile_s = None
        if args.variant in ('compile', 'compile-cg', 'trt'):
            shape = backbone_shape(model, frames[0], args.batch)
            compile_s = compile_backbone(model, args.variant, shape, args.precision, args.trt_mode)
        times = time_inference(model, frames, args.batch, args.precision, args.rounds)
        found = top_detection(infer(model, frames, args.batch, args.precision))
        per_batch = sorted(times)
        record.update({
            'frames_timed': len(frames) * args.rounds, 'compile_s': compile_s,
            'frames_per_s': len(frames) * args.rounds / sum(times),
            'ms_per_batch_median': 1e3 * statistics.median(per_batch),
            'ms_per_batch_p95': 1e3 * per_batch[min(len(per_batch) - 1, int(0.95 * len(per_batch)))],
            'vs_eager_fp32_same_size': compare(reference, found),
            'vs_eager_fp32_size_800': compare(at_800, found),
            'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
        })
    except Exception as e:                                        # recorded, so the sweep carries on
        record.update({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f'{host}-{config}-{job}.json')
    with open(out, 'w') as f:
        json.dump(record, f, indent=2, default=float)
    summary = {k: record.get(k) for k in ('config', 'frames_per_s', 'ms_per_batch_median', 'compile_s', 'error')}
    print(json.dumps(summary, default=float))
    print('wrote', out)


if __name__ == '__main__':
    main()
