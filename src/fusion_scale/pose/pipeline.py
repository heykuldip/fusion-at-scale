"""From encoded video to body-joint positions on one GPU, and which stage limits the rate.

Four stages are timed separately: decode, preprocess, inference (torchvision's Keypoint R-CNN) and write. Batches come
either from decoding in this process (on the CPU or the GPU, with TorchCodec) or from DataLoader workers decoding on
the CPU while the GPU works. torch and TorchCodec are imported inside the functions that need them, so the plain parts
run and test anywhere.
"""
import glob
import os
from contextlib import nullcontext

import numpy as np
import time
import torch


FULL_VIDEOS = 'videosfullsize/videosfullsize/videosoriginal'
FRAMES_AT_SCALE = 324_000_000          # corpus size for the GPU-hours estimate: 3,000 hours of 30 fps video
DECODERS = ('cpu', 'cuda', 'cuda-beta')  # TorchCodec on the CPU; its default CUDA backend (FFmpeg's NVDEC); its beta


def video_list(root, subjects=None, activities=None, videos_dir=FULL_VIDEOS):
    """The trials' .mp4 files (not the N-pose clips), sorted, optionally only some subjects and activities."""
    paths = sorted(glob.glob(os.path.join(root, videos_dir, 'S*', 'S*_A*_T*.mp4')))
    keep = []
    for p in paths:
        name = os.path.basename(p)[:-4]
        if '_Npose' in name:
            continue
        s, a, _ = name.split('_')[:3]
        if (subjects and s not in subjects) or (activities and a not in activities):
            continue
        keep.append(p)
    return keep


def make_decoder(path, kind, threads=1):
    """A TorchCodec VideoDecoder. kind: 'cpu' (with `threads` FFmpeg threads), 'cuda' (the default CUDA backend,
    FFmpeg's NVDEC) or 'cuda-beta' (TorchCodec drives NVDEC itself). The backend is fixed when the decoder is made."""
    import torchcodec
    import torchcodec.decoders as dec
    if kind == 'cpu':
        return dec.VideoDecoder(path, num_ffmpeg_threads=threads)
    if kind not in DECODERS:
        raise ValueError(f'unknown decoder {kind!r}; choose from {DECODERS}')
    set_backend = getattr(dec, 'set_cuda_backend', None) or getattr(torchcodec, 'set_cuda_backend', None)
    with set_backend('beta' if kind == 'cuda-beta' else 'ffmpeg') if set_backend else nullcontext():
        return dec.VideoDecoder(path, device='cuda')


def sequential_batches(videos, kind, batch_size, threads=1):
    """Yield (video key, first frame index, uint8 frames (n, 3, height, width)) on the decoder's device.

    Decoding happens in this process, inside next(), including opening each video. So the time a caller spends in
    next() is the decode stage (for a CUDA decoder, after a synchronize)."""
    for path in videos:
        d = make_decoder(path, kind, threads)
        key = os.path.basename(path)[:-4]
        for start in range(0, len(d), batch_size):
            yield key, start, d.get_frames_in_range(start, min(start + batch_size, len(d))).data


def overlapped_batches(videos, batch_size, workers, threads=1):
    """The same batches, decoded on the CPU by `workers` DataLoader workers while the caller uses the GPU. Each
    worker takes every workers-th video, so videos interleave. Time in next() is now only waiting for decoders that
    fall behind. Frames arrive in pinned memory, ready for a non-blocking copy to the GPU."""
    import torch
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info

    class _CpuChunks(IterableDataset):
        def __iter__(self):
            info = get_worker_info()
            mine = videos if info is None else videos[info.id::info.num_workers]
            yield from sequential_batches(mine, 'cpu', batch_size, threads)

    loader = DataLoader(_CpuChunks(), batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                        prefetch_factor=4 if workers else None)
    yield from loader


def load_model(device, min_size=800):
    """torchvision's Keypoint R-CNN (ResNet-50 FPN, COCO's 17 keypoints), eval mode. min_size=800 is torchvision's
    default (it scales 640x480 up to 1067x800); min_size=480 keeps VIDIMU's frames at their own size."""
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
    max_size = 1333 if min_size == 800 else int(round(min_size * 4 / 3))
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                                      min_size=min_size, max_size=max_size)
    return model.eval().to(device)


def _to_numpy(x):
    return x.detach().float().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x, dtype=np.float32)


def top_person(outputs) -> np.ndarray:
    """(frames, 17, 3): for each frame, the most confident detection's keypoints, as x, y and the keypoint's score;
    NaN where nobody was found. This moves the outputs to the CPU, so it waits for the GPU."""
    out = np.full((len(outputs), 17, 3), np.nan, dtype=np.float32)
    for i, o in enumerate(outputs):
        scores = _to_numpy(o['scores'])
        if scores.size == 0:
            continue
        k = int(np.argmax(scores))
        keypoints = _to_numpy(o['keypoints'])[k]
        out[i, :, :2] = keypoints[:, :2]
        out[i, :, 2] = _to_numpy(o['keypoints_scores'])[k] if 'keypoints_scores' in o else keypoints[:, 2]
    return out


class KeypointWriter:
    """Collects each video's keypoints, whatever order the chunks come in, and writes one .npy per video on close()."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._parts = {}

    def add(self, key, start, keypoints):
        self._parts.setdefault(key, []).append((start, keypoints))

    def close(self) -> int:
        for key, parts in self._parts.items():
            parts.sort(key=lambda p: p[0])
            np.save(os.path.join(self.out_dir, f'{key}.npy'), np.concatenate([p[1] for p in parts]))
        written, self._parts = len(self._parts), {}
        return written


def run_pipeline(batches, model, device, precision, writer, warmup_frames, frame_limit) -> dict:
    """Push frames through the four stages and time each: decode (the time inside next(batches)), preprocess,
    inference and write, synchronizing the GPU around each stage so its time is its own. The first warmup_frames frames
    are processed but not timed; then frame_limit frames are timed. Returns a dict:
      frames    frames timed
      seconds   {'decode', 'preprocess', 'inference', 'write'}: total seconds over the timed frames
      wall      wall-clock seconds over the timed frames
      people    timed frames in which a person was found
    """
    if precision not in ('fp32', 'bf16'):
        raise ValueError("precision must be 'fp32' or 'bf16'")

    dev = torch.device(device)
    is_cuda = dev.type == 'cuda'

    def sync():
        if is_cuda:
            torch.cuda.synchronize(dev)

    seconds = {
        'decode': 0.0,
        'preprocess': 0.0,
        'inference': 0.0,
        'write': 0.0,
    }

    warm_left = max(0, int(warmup_frames))
    limit = max(0, int(frame_limit))

    frames_timed = 0
    people = 0
    wall_start = None

    it = iter(batches)

    def process(frames, key, start, timed):
        nonlocal people

        sync()
        t0 = time.perf_counter()

        x = frames.to(dev, non_blocking=True).float().div_(255.0)
        images = list(x.unbind(0))

        sync()
        if timed:
            seconds['preprocess'] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()

        amp = (
            torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
            if precision == 'bf16'
            else nullcontext()
        )

        with torch.inference_mode():
            with amp:
                outputs = model(images)

        sync()
        if timed:
            seconds['inference'] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()

        keypoints = top_person(outputs)
        writer.add(key, start, keypoints)              # every frame, so each file starts at frame 0
        if timed:
            people += int(np.isfinite(keypoints[:, 0, 0]).sum())

        sync()
        if timed:
            seconds['write'] += time.perf_counter() - t0

    while frames_timed < limit:

        # Once warmup is finished, next() itself is the decode stage.
        timing_decode = (warm_left == 0)

        sync()
        t0 = time.perf_counter()

        if timing_decode and wall_start is None:
            wall_start = t0

        try:
            key, start, frames = next(it)
        except StopIteration:
            break

        sync()
        decode_time = time.perf_counter() - t0

        n = int(frames.shape[0])

        if warm_left:
            n_warm = min(warm_left, n)

            if n_warm:
                process(
                    frames[:n_warm],
                    key,
                    start,
                    timed=False,
                )

            warm_left -= n_warm
            frames = frames[n_warm:]
            start += n_warm
            n -= n_warm

            if n == 0:
                continue

            # warmup ended inside this batch, whose decoding was untimed: the wall clock starts now
            if wall_start is None:
                sync()
                wall_start = time.perf_counter()

        else:
            seconds['decode'] += decode_time

        remaining = limit - frames_timed
        take = min(n, remaining)

        if take <= 0:
            break

        # Handles both a video's short final batch and a frame-limit cut.
        frames = frames[:take]

        process(
            frames,
            key,
            start,
            timed=True,
        )

        frames_timed += take

    # np.save happens in close(), so count it as write time too.
    if frames_timed:
        sync()
        t0 = time.perf_counter()

        writer.close()

        sync()
        seconds['write'] += time.perf_counter() - t0
        wall = time.perf_counter() - wall_start
    else:
        wall = 0.0

    return {
        'frames': frames_timed,
        'seconds': seconds,
        'wall': wall,
        'people': people,
    }


def throughput(result) -> dict:
    """Rates from run_pipeline()'s result: frames a second for each stage and end to end (from the wall clock), the
    limiting stage, and GPU-hours for FRAMES_AT_SCALE frames at the end-to-end rate.
    """

    frames = result['frames']
    seconds = result['seconds']
    wall = result['wall']

    def fps(s):
        return frames / s if s > 0 else float('inf')

    decode_fps = fps(seconds['decode'])
    preprocess_fps = fps(seconds['preprocess'])
    inference_fps = fps(seconds['inference'])
    write_fps = fps(seconds['write'])

    end_to_end_fps = frames / wall if wall > 0 else 0.0

    limiting_stage = max(
        seconds,
        key=seconds.get,
    )

    gpu_hours_at_scale = (
        FRAMES_AT_SCALE / end_to_end_fps / 3600.0
        if end_to_end_fps > 0
        else float('inf')
    )

    return {
        'decode_fps': decode_fps,
        'preprocess_fps': preprocess_fps,
        'inference_fps': inference_fps,
        'write_fps': write_fps,
        'end_to_end_fps': end_to_end_fps,
        'limiting_stage': limiting_stage,
        'gpu_hours_at_scale': gpu_hours_at_scale,
    }
