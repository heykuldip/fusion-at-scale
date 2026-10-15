"""Windows of synchronized video and IMU data from VIDIMU.

Each trial's body joints (30 a second) and IMU joint angles (50 a second) are put on one timeline starting at 0, cut
into fixed-length windows that fit both streams, checked for alignment against the activity's joint of interest, and
split by subject, so no subject appears in more than one split. numpy only.
"""
import csv
import random
from dataclasses import asdict, dataclass, fields

import numpy as np

from fusion_scale.data.vidimu import (
    IMU_HZ, JOINT_OF_INTEREST, VIDEO_FPS, Trial,
    read_joints, read_mot,
)

LOWER_MEASURED = {
    'pelvis_tilt',
    'pelvis_list',
    'pelvis_rotation',
    'hip_flexion_r',
    'hip_adduction_r',
    'hip_rotation_r',
    'knee_angle_r',
    'hip_flexion_l',
    'hip_adduction_l',
    'hip_rotation_l',
    'knee_angle_l',
}

UPPER_MEASURED = {
    'lumbar_extension',
    'lumbar_bending',
    'lumbar_rotation',
    'arm_flex_r',
    'arm_add_r',
    'arm_rot_r',
    'elbow_flex_r',
    'pro_sup_r',
    'arm_flex_l',
    'arm_add_l',
    'arm_rot_l',
    'elbow_flex_l',
    'pro_sup_l',
}

MEASURED = {
    'lower': LOWER_MEASURED,
    'upper': UPPER_MEASURED,
}


@dataclass(frozen=True)
class Window:
    """One window: where to find each stream's slice, and what it is."""
    subject: str
    activity: str
    trial: str
    label: int                  # activity index, 0-12
    body: str                   # 'lower' or 'upper'
    start_s: float              # on the synchronized timeline
    end_s: float
    frame_start: int            # rows of the synchronized joints, [frame_start, frame_stop)
    frame_stop: int
    video_frame_start: int      # the .mp4's frame for frame_start
    imu_start: int              # rows of the synchronized IMU stream, [imu_start, imu_stop)
    imu_stop: int
    tracked_fraction: float     # share of the window's frames BodyTrack tracked

    @property
    def key(self) -> str:
        return f'{self.subject}_{self.activity}_{self.trial}'


def synchronized_streams(trial: Trial) -> dict:
    """One trial's two streams on a shared timeline that starts at 0, as a dict:
      joints        (F, 34, 3) mm, one row per video frame (30 a second)
      joint_names   the 34 names
      video_offset  the .mp4's frame that row 0 of joints corresponds to
      tracked       (F,) bool, False where BodyTrack lost the subject
      imu           (S, C) degrees, one row per 0.02 s; columns not measured in this activity are NaN
      imu_names     the C names
      imu_time      (S,) seconds, starting at 0
      imu_measured  (C,) bool, the measured columns
    The authors' synchronized file is used where one exists; the .mot's first row, the N-pose calibration, is dropped.
    """
    joints_path = trial.joints_sync or trial.joints
    mot_path = trial.mot_sync or trial.mot

    if mot_path is None:
        raise FileNotFoundError(f'{trial.key}: no .mot file')

    joints, joint_names = read_joints(joints_path)
    mot_time, imu_names, imu = read_mot(mot_path)

    # First .mot data row is the N-pose calibration, including in sync files.
    if len(imu):
        imu = imu[1:]
        mot_time = mot_time[1:]

    # Rebuild the time axis from the sample count: retained absolute timestamps may be off after a synchronization cut.
    imu_time = np.arange(len(imu), dtype=float) / IMU_HZ

    # Columns this activity's sensors don't estimate are constant: zero, or a locked value (pelvis_ty sits at 0.94).
    imu = imu.astype(float, copy=True)

    in_half = np.array(
        [name in MEASURED[trial.body] for name in imu_names],
        dtype=bool,
    )

    if len(imu):
        # A channel is usable only when:
        #   1. its coordinate belongs to the activity's instrumented body half; and
        #   2. it varies during this trial.
        varying = np.ptp(imu, axis=0) > 0
        measured = in_half & varying
    else:
        measured = np.zeros(len(imu_names), dtype=bool)

    imu[:, ~measured] = np.nan

    # The paper does not publish the exact BodyTrack default 3-D sentinel.
    # Flag obvious default/constant poses and non-finite rows.
    flat = joints.reshape(len(joints), -1)
    finite = np.all(np.isfinite(flat), axis=1)

    constant = np.zeros(len(joints), dtype=bool)
    if len(joints):
        constant = np.ptp(flat, axis=1) == 0

    # Exact repeated full poses are another sign of BodyTrack's fixed default.
    repeated = np.zeros(len(joints), dtype=bool)
    if len(joints) > 1:
        same = np.all(flat[1:] == flat[:-1], axis=1)
        repeated[1:] |= same
        repeated[:-1] |= same

    tracked = finite & ~constant & ~repeated

    return {
        'joints': joints,
        'joint_names': joint_names,
        'video_offset': trial.video_cut,
        'tracked': tracked,
        'imu': imu,
        'imu_names': imu_names,
        'imu_time': imu_time,
        'imu_measured': measured,
    }


def build_windows(trial: Trial, streams: dict, length_s: float = 2.0, stride_s: float = 1.0):
    """Cut one trial into windows that fit inside both streams. Returns (windows, dropped): the Window list, and how
    many candidate windows ran past either stream's end. Each window's position is computed from its index, so rounding
    doesn't accumulate.
    """
    joints = streams['joints']
    imu = streams['imu']
    tracked = streams['tracked']

    n_frames = int(round(length_s * VIDEO_FPS))
    n_imu = int(round(length_s * IMU_HZ))

    video_duration = len(joints) / VIDEO_FPS
    imu_duration = len(imu) / IMU_HZ

    def complete_windows(duration):
        if duration < length_s:
            return 0
        return int(np.floor((duration - length_s) / stride_s)) + 1

    # Consider every window that fully exists in at least one stream.
    n_candidates = max(
        complete_windows(video_duration),
        complete_windows(imu_duration),
    )

    windows = []
    dropped = 0

    for k in range(n_candidates):
        start_s = k * stride_s
        end_s = start_s + length_s

        # Compute from k/time every time: no accumulated rounding.
        frame_start = int(round(start_s * VIDEO_FPS))
        imu_start = int(round(start_s * IMU_HZ))

        frame_stop = frame_start + n_frames
        imu_stop = imu_start + n_imu

        if frame_stop > len(joints) or imu_stop > len(imu):
            dropped += 1
            continue

        windows.append(Window(
            subject=trial.subject,
            activity=trial.activity,
            trial=trial.trial,
            label=trial.label,
            body=trial.body,
            start_s=start_s,
            end_s=end_s,
            frame_start=frame_start,
            frame_stop=frame_stop,
            video_frame_start=streams['video_offset'] + frame_start,
            imu_start=imu_start,
            imu_stop=imu_stop,
            tracked_fraction=float(np.mean(
                tracked[frame_start:frame_stop]
            )),
        ))

    return windows, dropped


def alignment_lag(trial: Trial, streams: dict, max_lag: int = 10, seconds: float = 6.0) -> int:
    """The lag, in video frames, that best matches the video-derived and the IMU-derived angle of the activity's joint
    of interest (JOINT_OF_INTEREST, from the dataset's paper) over the first `seconds`, by RMSE after removing each
    angle's mean. 0 means aligned; positive means the video lags the IMU.
    """

    seg_a, seg_b, imu_name = JOINT_OF_INTEREST[trial.activity]

    video_angle = segment_angle(
        streams['joints'],
        streams['joint_names'],
        seg_a,
        seg_b,
    ).astype(float)

    # Tracking failures are missing observations, not motion.
    tracked = streams.get('tracked')
    if tracked is not None:
        video_angle[~tracked] = np.nan

    imu_names = list(streams['imu_names'])
    if imu_name not in imu_names:
        raise ValueError(f'{trial.key}: {imu_name!r} not in .mot columns')

    imu_angle = streams['imu'][:, imu_names.index(imu_name)]

    # IMU 50 Hz -> video 30 Hz.
    imu_angle = to_video_rate(
        streams['imu_time'],
        imu_angle,
        len(video_angle),
    )

    def fill_missing(x):
        x = np.asarray(x, dtype=float).copy()
        good = np.flatnonzero(np.isfinite(x))
        if len(good) < 2:
            return x
        bad = np.flatnonzero(~np.isfinite(x))
        x[bad] = np.interp(bad, good, x[good])
        return x

    video_angle = fill_missing(video_angle)
    imu_angle = fill_missing(imu_angle)

    # Authors used a 5-sample moving average before synchronization.
    kernel = np.ones(5, dtype=float) / 5.0
    video_angle = np.convolve(video_angle, kernel, mode='valid')
    imu_angle = np.convolve(imu_angle, kernel, mode='valid')

    fit = int(round(seconds * VIDEO_FPS))
    available = min(len(video_angle), len(imu_angle))

    # Every tested lag compares the same number of samples.
    fit = min(fit, available - max_lag)
    if fit <= 0:
        raise ValueError(f'{trial.key}: too little data for alignment check')

    prefix = fit + max_lag
    video_angle = video_angle[:prefix]
    imu_angle = imu_angle[:prefix]

    # Remove modality-dependent angular offsets before comparing.
    video_angle = video_angle - np.mean(video_angle)
    imu_angle = imu_angle - np.mean(imu_angle)

    best_lag = 0
    best_rmse = np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            # Positive: video occurs later, so cut its first `lag` frames.
            a = imu_angle[:fit]
            b = video_angle[lag:lag + fit]
        else:
            # Negative: IMU occurs later.
            shift = -lag
            a = imu_angle[shift:shift + fit]
            b = video_angle[:fit]

        valid = np.isfinite(a) & np.isfinite(b)
        if not np.any(valid):
            continue

        rmse = float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag

    if not np.isfinite(best_rmse):
        raise ValueError(f'{trial.key}: no lag had valid samples to compare')
    return best_lag


def segment_angle(joints: np.ndarray, names: list, segment_a: tuple, segment_b: tuple) -> np.ndarray:
    """Degrees between two segments, each (from joint, to joint), for every frame: 0 when they point the same way."""
    idx = {n: i for i, n in enumerate(names)}
    a = joints[:, idx[segment_a[1]]] - joints[:, idx[segment_a[0]]]
    b = joints[:, idx[segment_b[1]]] - joints[:, idx[segment_b[0]]]
    with np.errstate(invalid='ignore', divide='ignore'):        # a zero-length segment gives NaN, not a warning
        cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def to_video_rate(times: np.ndarray, values: np.ndarray, frames: int) -> np.ndarray:
    """Values sampled at `times` (seconds), linearly interpolated at the video's frame times, 0 to frames - 1."""
    return np.interp(np.arange(frames) / VIDEO_FPS, times, values, left=np.nan, right=np.nan)


def split_by_subject(subjects, seed: int = 0, n_val: int = 2, n_test: int = 2) -> dict:
    """{subject: 'train' | 'val' | 'test'}: a fixed-seed shuffle of the subjects, so every window of one subject
    lands in the same split."""
    order = sorted(set(subjects))
    random.Random(seed).shuffle(order)
    return {s: ('test' if i < n_test else 'val' if i < n_test + n_val else 'train') for i, s in enumerate(order)}


def write_index(windows, splits: dict, path: str) -> None:
    """The window index as CSV: every Window field, plus the window's split."""
    names = [f.name for f in fields(Window)] + ['split']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for win in windows:
            w.writerow({**asdict(win), 'split': splits[win.subject]})


def read_index(path: str):
    """The window index back as (windows, splits)."""
    types = {f.name: f.type for f in fields(Window)}
    convert = {'int': int, 'float': float, 'str': str, int: int, float: float, str: str}
    windows, splits = [], {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            splits[row['subject']] = row.pop('split')
            windows.append(Window(**{k: convert[types[k]](v) for k, v in row.items()}))
    return windows, splits


__all__ = ['Window', 'synchronized_streams', 'build_windows', 'alignment_lag', 'segment_angle', 'to_video_rate',
           'split_by_subject', 'write_index', 'read_index', 'IMU_HZ', 'JOINT_OF_INTEREST', 'VIDEO_FPS']
