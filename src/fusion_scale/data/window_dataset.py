"""A Dataset over the window index, and PyAV frame decoding.

With video, start DataLoader workers with "spawn" rather than "fork": a worker forked from a process that has
already used PyAV can crash. Spawned workers don't inherit the parent's state, so the dataset carries the loaded
streams and the videos' frame indexes itself, and they reach each worker by pickling.

Each process (and each DataLoader worker) keeps the synchronized streams of the trials it has touched, so a trial's
files are read once per worker, not once per window. Video frames are decoded only when asked for (with_video=True).
"""
import numpy as np


_FRAME_INDEX = {}


def frame_index(path: str):
    """Every frame's presentation timestamp, in display order, and which frames are keyframes: one full decoding
    pass per video, cached per process (DataLoader workers inherit it if it's built before they start)."""
    if path not in _FRAME_INDEX:
        import av
        with av.open(path) as container:
            pts, keys = [], []
            for frame in container.decode(container.streams.video[0]):
                pts.append(frame.pts)
                keys.append(bool(frame.key_frame))
        _FRAME_INDEX[path] = (pts, [i for i, k in enumerate(keys) if k])
    return _FRAME_INDEX[path]


def decode_frames(path: str, start: int, count: int, fps: float = 30.0, index=None) -> np.ndarray:
    """Frames start .. start + count - 1 of a video, as uint8 (count, height, width, 3) RGB, in display order.

    Frames are found by their exact timestamps from frame_index(), not by arithmetic on timestamps. The decoder
    seeks to the last keyframe at or before `start`; if that doesn't yield every frame wanted (one VIDIMU video lost
    frames after a seek), it decodes from the beginning instead, and raises an error if even that misses any.
    Returns fewer frames only where the video ends first. `fps` is unused, kept for the old signature."""
    import bisect

    import av
    pts, keyframes = index if index is not None else frame_index(path)
    want = pts[start:start + count]
    if not want:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    slot = {p: i for i, p in enumerate(want)}

    def collect(seek_pts):
        out = [None] * len(want)
        with av.open(path) as container:
            stream = container.streams.video[0]
            if seek_pts is not None:
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            found = 0
            for frame in container.decode(stream):
                i = slot.get(frame.pts)
                if i is not None and out[i] is None:
                    out[i] = frame.to_ndarray(format='rgb24')
                    found += 1
                    if found == len(want):
                        break
                elif frame.pts is not None and frame.pts > want[-1]:
                    break
        return out, found

    k = bisect.bisect_right(keyframes, start) - 1
    out, found = collect(pts[keyframes[k]] if k >= 0 else None)
    if found < len(want):
        out, found = collect(None)
        if found < len(want):
            raise RuntimeError(f'{path}: {found} of {len(want)} frames from {start} decoded, even from the beginning')
    return np.stack(out)


class WindowDataset:
    """Windows as dicts of tensors: joints (frames, 34, 3), imu (rows, channels), label, and with_video, frames
    (frames, 3, height, width). Subclasses torch's Dataset when torch is importable; otherwise a plain class."""

    def __init__(self, windows, trials, streams_fn, with_video: bool = False):
        self.windows = list(windows)
        self.trials = {t.key: t for t in trials}
        self.streams_fn = streams_fn          # e.g. sync_windows.synchronized_streams
        self.with_video = with_video
        self._cache = {}
        self.video_index = {}                 # video path: frame_index(), carried to spawned workers

    def __len__(self):
        return len(self.windows)

    def preload(self) -> float:
        """Load every trial the windows use, in this process, and return the seconds it took. DataLoader workers
        started afterwards (by fork) inherit the loaded streams, so their timing measures steady-state loading,
        not each worker parsing files the first time it meets a trial."""
        import time
        t = time.perf_counter()
        for key in sorted({w.key for w in self.windows}):
            self._streams(key)
        return time.perf_counter() - t

    def preload_video_index(self) -> float:
        """Build frame_index() for every video the windows use, in this process, and return the seconds it took."""
        import time
        t = time.perf_counter()
        for key in sorted({w.key for w in self.windows}):
            path = self.trials[key].video
            self.video_index[path] = frame_index(path)
        return time.perf_counter() - t

    def sharing(self, with_video: bool):
        """The same windows and loaded streams, with or without decoding video."""
        other = type(self)(self.windows, self.trials.values(), self.streams_fn, with_video=with_video)
        other._cache = self._cache
        other.video_index = self.video_index
        return other

    def _streams(self, key):
        if key not in self._cache:
            self._cache[key] = self.streams_fn(self.trials[key])
        return self._cache[key]

    def __getitem__(self, i):
        import torch
        w = self.windows[i]
        s = self._streams(w.key)
        joints = np.ascontiguousarray(s['joints'][w.frame_start:w.frame_stop], dtype=np.float32)
        imu = np.ascontiguousarray(s['imu'][w.imu_start:w.imu_stop], dtype=np.float32)
        sample = {'joints': torch.from_numpy(joints), 'imu': torch.from_numpy(imu),
                  'label': w.label}
        if self.with_video:
            n = w.frame_stop - w.frame_start
            path = self.trials[w.key].video
            frames = decode_frames(path, w.video_frame_start, n, index=self.video_index.get(path))
            if len(frames) != n:
                raise RuntimeError(f'{w.key}: {len(frames)} of {n} frames decoded from {w.video_frame_start}')
            sample['frames'] = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return sample


try:                                          # a real torch Dataset when torch is there, for DataLoader
    import torch.utils.data as _tud
    WindowDataset = type('WindowDataset', (WindowDataset, _tud.Dataset), {})
except ImportError:
    pass
