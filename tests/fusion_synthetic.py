"""A small synthetic stand-in for the fusion window file: the export's schema, a known signal per activity, missing
channels per body half, some lost tracking, and (optionally) a standing position that differs by activity, the leak
the centring removes and the position-only control should detect.
"""
import numpy as np

from fusion_scale.fusion.data import LOWER_CHANNELS, UPPER_CHANNELS

JOINTS = ['pelvis', 'left_hip', 'right_hip', 'neck', 'torso', 'left_shoulder', 'right_shoulder', 'left_elbow',
          'right_elbow', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
JOINTS = JOINTS + [f'extra_{k}' for k in range(34 - len(JOINTS))]
IMU = list(LOWER_CHANNELS) + list(UPPER_CHANNELS) + ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']


def make(path, n_subjects=6, trials=2, windows=3, seed=0, position_leak=True, wrist_posture=False, lost=0.05):
    rng = np.random.default_rng(seed)
    subjects = [f'S{40 + k}' for k in range(n_subjects)]
    rows = []
    for s_i, s in enumerate(subjects):
        size = 1 + 0.1 * s_i                                   # body size differs by subject
        base = rng.normal(0, 150, (34, 3)) * size
        base[JOINTS.index('neck')] = base[JOINTS.index('left_hip')] + np.array([0, 500 * size, 0])
        for a in range(13):
            for t in range(trials):
                for w in range(windows):
                    rows.append((s, a, t, w, size, base))
    n = len(rows)
    joints = np.zeros((n, 60, 34, 3), np.float32)
    tracked = rng.random((n, 60)) >= lost
    imu = np.full((n, 100, len(IMU)), np.nan, np.float32)
    labels, subj, acts, trs, keys, starts = [], [], [], [], [], []
    tj, ti = np.arange(60) / 30.0, np.arange(100) / 50.0
    for k, (s, a, t, w, size, base) in enumerate(rows):
        f = 0.5 + 0.5 * (a % 9)                                # resolvable in a 2-second window
        j = np.repeat(base[None], 60, axis=0) + rng.normal(0, 5, (60, 34, 3))
        j[:, JOINTS.index('right_wrist'), 1] += 100 * size * np.sin(2 * np.pi * f * (tj + w))
        if wrist_posture:                                      # an easy, posture-like joint signal
            j[:, JOINTS.index('right_wrist'), 0] += 60 * size * a
        j += np.array([2000.0, 0.0, 3000.0]) + (np.array([300.0 * a, 0, 0]) if position_leak else 0)
        joints[k] = j
        half = LOWER_CHANNELS if a < 4 else UPPER_CHANNELS
        for c in half:
            imu[k, :, IMU.index(c)] = 20 * np.sin(2 * np.pi * f * (ti + w)) + rng.normal(0, 2, 100)
        labels.append(a)
        subj.append(s)
        acts.append(f'A{a + 1:02d}')
        trs.append(f'T{t + 1:02d}')
        keys.append(f'{s}_A{a + 1:02d}_T{t + 1:02d}')
        starts.append(float(w))
    np.savez(path, joints=joints, tracked=tracked, imu=imu, labels=np.array(labels, np.int16), subjects=np.array(subj),
             activities=np.array(acts), trials=np.array(trs), keys=np.array(keys), start_s=np.array(starts, np.float32),
             tracked_fraction=tracked.mean(1).astype(np.float32), index_split=np.array(['train'] * n),
             joint_names=np.array(JOINTS), imu_names=np.array(IMU))
    return path
