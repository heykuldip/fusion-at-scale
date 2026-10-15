"""Windows as arrays for the fusion comparison: features blind to where a person stands, folds, and leakage
controls. numpy only; every statistic is fitted on training subjects only.

  - the two body halves are separate problems (lower: A01-A04, 4 classes; upper: A05-A13, 9 classes), each with a
    fixed list of IMU channels, so the channels present don't give the half away; 'all' (13 classes) is secondary
  - joints are centred on the window's mean hip midpoint and scaled by its median torso length, so standing position
    and body size are left out; lost-tracking frames are flagged
  - each stream is split into its motion (every coordinate or angle minus its own window mean, over time) and its
    posture (those window means, one vector per window), so posture offsets that differ between windows don't swamp
    the motion in the first layer. Lost frames and missing IMU values sit at the window mean (0 in the motion input),
    with a mask
  - folds leave one subject out for testing and one other for early stopping; metadata isn't used as an input
"""
import numpy as np

ACTIVITIES = tuple(f'A{i:02d}' for i in range(1, 14))
LOWER_CHANNELS = ('pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'hip_flexion_r', 'hip_adduction_r',
                  'hip_rotation_r', 'knee_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                  'knee_angle_l')
UPPER_CHANNELS = ('lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                  'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l')
HALVES = {'lower': tuple(range(0, 4)), 'upper': tuple(range(4, 13)), 'all': tuple(range(13))}
CHANNELS = {'lower': LOWER_CHANNELS, 'upper': UPPER_CHANNELS, 'all': LOWER_CHANNELS + UPPER_CHANNELS}


def check_channels():
    """Compare the channel lists with the windows' measured sets; returns a message, or None if sync_windows isn't
    importable."""
    try:
        from fusion_scale.data.sync_windows import MEASURED
    except Exception:
        return None
    ok = set(MEASURED['lower']) == set(LOWER_CHANNELS) and set(MEASURED['upper']) == set(UPPER_CHANNELS)
    return 'match' if ok else 'MISMATCH: the fusion and window measured-channel sets differ'


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def half_view(d, half):
    """Rows of `d` in this half, and their labels renumbered 0..K-1 within it."""
    keep = np.isin(d['labels'], HALVES[half])
    rows = np.flatnonzero(keep)
    lookup = {a: i for i, a in enumerate(HALVES[half])}
    return rows, np.array([lookup[int(v)] for v in d['labels'][rows]], np.int64), len(HALVES[half])


def folds(subjects):
    """Leave-one-subject-out: (test, validation, training subjects) per fold, in sorted order. The validation
    subject is the next one in that order, fixed before any result is seen."""
    order = sorted(set(str(s) for s in subjects))
    n = len(order)
    if n < 3:
        raise ValueError('need at least 3 subjects')
    for i, s in enumerate(order):
        v = order[(i + 1) % n]
        yield s, v, [x for x in order if x not in (s, v)]


def joint_index(names, name):
    names = [str(x) for x in names]
    if name not in names:
        raise KeyError(f'joint {name!r} not among {names}')
    return names.index(name)


def joint_features(joints, tracked, names):
    """(N, 60, 34, 3) mm -> motion (N, 60, 103) and posture (N, 102).

    Every joint is taken relative to the window's mean hip midpoint and divided by the window's median torso length
    (neck to hip midpoint), so neither standing position nor body size survives. Posture is each coordinate's mean
    over the window's tracked frames; motion is the coordinate minus that mean, 0 in lost frames, plus the tracked
    flag as the last feature."""
    lh, rh, nk = (joint_index(names, n) for n in ('left_hip', 'right_hip', 'neck'))
    j = joints.astype(np.float64)
    t = tracked.astype(bool)
    hip = 0.5 * (j[:, :, lh] + j[:, :, rh])                       # (N, T, 3)
    count = t.sum(1)
    w = t[:, :, None]
    center = np.where(count[:, None] > 0, (hip * w).sum(1) / np.maximum(count, 1)[:, None], 0.0)
    torso = np.linalg.norm(j[:, :, nk] - hip, axis=2)              # (N, T)
    torso = np.where(t, torso, np.nan)
    with np.errstate(all='ignore'):
        scale = np.nanmedian(torso, axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    x = (j - center[:, None, None, :]) / scale[:, None, None, None]
    n, f = x.shape[:2]
    x = x.reshape(n, f, -1)
    posture = np.where(count[:, None] > 0, (x * w).sum(1) / np.maximum(count, 1)[:, None], 0.0)
    motion = np.where(w, x - posture[:, None, :], 0.0)
    return (np.concatenate([motion, w.astype(np.float64)], axis=2).astype(np.float32), posture.astype(np.float32))


def position_features(joints, tracked, names):
    """The negative control's input: the hip midpoint's path in camera coordinates (mm), before any centring."""
    lh, rh = joint_index(names, 'left_hip'), joint_index(names, 'right_hip')
    t = tracked.astype(bool)
    hip = 0.5 * (joints[:, :, lh] + joints[:, :, rh]).astype(np.float64)
    hip[~t] = 0.0
    return np.concatenate([hip, t[:, :, None]], axis=2).astype(np.float32)


def imu_features(imu, imu_names, half):
    """The half's fixed channels: motion (N, 100, C), each channel minus its window mean and 0 where missing; the
    mask (N, 100, C); and posture (N, C), the window means (0 for a channel missing all window)."""
    names = [str(x) for x in imu_names]
    missing = [c for c in CHANNELS[half] if c not in names]
    if missing:
        raise KeyError(f'IMU channels missing from the data: {missing}')
    x = imu[:, :, [names.index(c) for c in CHANNELS[half]]].astype(np.float64)
    m = np.isfinite(x)
    v = np.where(m, x, 0.0)
    n = m.sum(1)
    posture = np.where(n > 0, v.sum(1) / np.maximum(n, 1), 0.0)
    motion = np.where(m, v - posture[:, None, :], 0.0)
    return motion.astype(np.float32), m, posture.astype(np.float32)


def mask_features(imu, imu_names, tracked, half):
    """The negative control's input: for each of the half's channels, the share of the window it's measured, plus
    the share of frames tracked. Not the values themselves."""
    _, m, _ = imu_features(imu, imu_names, half)
    return np.concatenate([m.mean(1), tracked.mean(1, keepdims=True)], axis=1).astype(np.float32)


class Standardizer:
    """Per-feature mean and standard deviation over the last axis, fitted on training rows only. Entries where
    `mask` is False are ignored when fitting and set to 0 after transforming; `keep` features are passed through."""

    def fit(self, x, mask=None, keep=()):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        m = np.ones_like(x, bool) if mask is None else mask.reshape(-1, mask.shape[-1])
        n = np.maximum(m.sum(0), 1)
        self.mean = np.where(m, x, 0).sum(0) / n
        var = np.where(m, (x - self.mean) ** 2, 0).sum(0) / n
        self.std = np.where(np.sqrt(var) > 1e-8, np.sqrt(var), 1.0)
        self.keep = np.zeros(x.shape[1], bool)
        self.keep[list(keep)] = True
        self.mean[self.keep], self.std[self.keep] = 0.0, 1.0
        return self

    def transform(self, x, mask=None):
        y = (x - self.mean) / self.std
        if mask is not None:
            y = np.where(mask, y, 0.0)
        return y.astype(np.float32)


def trial_shuffle(labels, trial_keys, rng):
    """Labels permuted between trials, so every window of a trial keeps one (wrong) label and the trials' label
    counts are unchanged, so memorizing trials doesn't help the shuffled-label control either."""
    keys = np.asarray(trial_keys).astype(str)
    uniq, first = np.unique(keys, return_index=True)
    per_trial = labels[first]
    perm = rng.permutation(len(uniq))
    new = dict(zip(uniq, per_trial[perm]))
    return np.array([new[k] for k in keys], labels.dtype)


__all__ = ['ACTIVITIES', 'LOWER_CHANNELS', 'UPPER_CHANNELS', 'HALVES', 'CHANNELS', 'check_channels',
           'load', 'half_view', 'folds', 'joint_features', 'position_features', 'imu_features', 'mask_features',
           'Standardizer', 'trial_shuffle']
