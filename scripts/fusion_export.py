"""The synchronized windows as one array file, cut by the same code that built the index.

    python scripts/fusion_export.py --index results/sync_windows/windows-<host>.csv --out results/fusion/windows.npz

Reads the window index and each trial's synchronized streams, slices every window, and writes one .npz plus a
manifest beside it (<out>.json): counts per half and subject, the channel check, how many windows lack any of their
half's channels, and the file's SHA-256. Problems aren't patched over: a window whose slice is the wrong length, or a
trial whose names differ, stops the export.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import subprocess

import numpy as np

from fusion_scale.data.sync_windows import read_index, synchronized_streams
from fusion_scale.data.vidimu import inventory
from fusion_scale.fusion.data import CHANNELS, HALVES, check_channels


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--index', default=None, help="windows-<host>.csv (default: the only one in results/sync_windows)")
    p.add_argument('--root', default=os.environ.get('VIDIMU_DIR', 'data/vidimu'))
    p.add_argument('--video-size', default='small', choices=['small', 'full'])
    p.add_argument('--out', default='results/fusion/windows.npz')
    a = p.parse_args()
    if a.index is None:
        found = sorted(glob.glob('results/sync_windows/windows-*.csv'))
        if len(found) != 1:
            raise SystemExit(f'give --index: found {found}')
        a.index = found[0]
    windows, index_splits = read_index(a.index)
    trials, problems = inventory(a.root, a.video_size)
    by_key = {t.key: t for t in trials}
    by_trial = collections.defaultdict(list)
    for k, w in enumerate(windows):
        by_trial[w.key].append(k)

    n = len(windows)
    joint_names = imu_names = None
    joints = tracked = imu = None
    for key in sorted(by_trial):
        s = synchronized_streams(by_key[key])
        if joint_names is None:
            joint_names, imu_names = list(s['joint_names']), list(s['imu_names'])
            joints = np.zeros((n, 60, len(joint_names), 3), np.float32)
            tracked = np.zeros((n, 60), bool)
            imu = np.full((n, 100, len(imu_names)), np.nan, np.float32)
        if list(s['joint_names']) != joint_names:
            raise SystemExit(f'{key}: joint names differ from the first trial')
        if sorted(s['imu_names']) != sorted(imu_names):
            raise SystemExit(f'{key}: IMU channels differ from the first trial')
        order = [list(s['imu_names']).index(c) for c in imu_names]
        for k in by_trial[key]:
            w = windows[k]
            j = s['joints'][w.frame_start:w.frame_stop]
            t = s['tracked'][w.frame_start:w.frame_stop]
            m = s['imu'][w.imu_start:w.imu_stop][:, order]
            if j.shape[0] != 60 or t.shape[0] != 60 or m.shape[0] != 100:
                raise SystemExit(f'{key} window at {w.start_s} s: {j.shape[0]} frames, {m.shape[0]} IMU rows')
            joints[k], tracked[k], imu[k] = j, t, m

    labels = np.array([w.label for w in windows], np.int16)
    subjects = np.array([w.subject for w in windows])
    arrays = dict(joints=joints, tracked=tracked, imu=imu, labels=labels, subjects=subjects,
                  activities=np.array([w.activity for w in windows]), trials=np.array([w.trial for w in windows]),
                  keys=np.array([w.key for w in windows]), start_s=np.array([w.start_s for w in windows], np.float32),
                  tracked_fraction=np.array([w.tracked_fraction for w in windows], np.float32),
                  index_split=np.array([index_splits[w.subject] for w in windows]),
                  joint_names=np.array(joint_names), imu_names=np.array(imu_names))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **arrays)

    complete = {}
    for half in ('lower', 'upper'):
        rows = np.isin(labels, HALVES[half])
        cols = [imu_names.index(c) for c in CHANNELS[half]]
        full = np.isfinite(imu[rows][:, :, cols]).all(axis=(1, 2))
        complete[half] = {'windows': int(rows.sum()), 'all_half_channels_present': int(full.sum())}
    h = hashlib.sha256()
    with open(a.out, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    manifest = {'out': os.path.abspath(a.out), 'sha256': h.hexdigest(), 'index': os.path.abspath(a.index),
                'root': a.root, 'windows': n, 'trials': len(by_trial), 'subjects': sorted(set(subjects)),
                'per_half_and_subject': {half: {s: int(((subjects == s) & np.isin(labels, HALVES[half])).sum())
                                                for s in sorted(set(subjects))} for half in ('lower', 'upper')},
                'per_class': {int(c): int((labels == c).sum()) for c in sorted(set(labels))},
                'channel_completeness': complete, 'channels_vs_windows': check_channels(),
                'inventory_problems': len(problems), 'joint_names': joint_names, 'imu_names': imu_names}
    try:
        manifest['code_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                                 timeout=10).stdout.strip()
    except Exception:
        manifest['code_commit'] = ''
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ('windows', 'trials', 'channel_completeness', 'channels_vs_windows',
                                               'per_class')}, indent=1))
    print(f'wrote {a.out} and its manifest')


if __name__ == '__main__':
    main()
