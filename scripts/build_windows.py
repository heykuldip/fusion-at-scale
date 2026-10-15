"""Build the window index from VIDIMU, run the checks that need the real data, and write the results.

    python scripts/build_windows.py --root data/vidimu
    python scripts/build_windows.py --root /tmp/vidimu-fixture --out /tmp    # on a synthetic tree

Writes, under --out (default results/sync_windows):
  windows-<host>.csv   the index: one row per window, with its split
  summary-<host>.json  counts per split, windows dropped, each trial's durations and alignment lag, video frame
                       counts against joint rows, and each problem found, so problems stay visible
"""
import argparse
import collections
import json
import os
import socket

from fusion_scale.data.sync_windows import (alignment_lag, build_windows, split_by_subject, synchronized_streams,
                                            write_index)
from fusion_scale.data.vidimu import inventory, read_joints


def video_frames(path):
    """The .mp4's frame count as its header declares it (quick), or None without PyAV."""
    try:
        import av
    except ImportError:
        return None
    with av.open(path) as c:
        return c.streams.video[0].frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('VIDIMU_DIR', 'data/vidimu'))
    p.add_argument('--video-size', default='small', choices=['small', 'full'])
    p.add_argument('--out', default='results/sync_windows')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    trials, problems = inventory(args.root, args.video_size)
    print(f'{len(trials)} trials, {len(problems)} problems from the inventory')
    windows, per_trial, failures = [], [], []
    for t in trials:
        try:
            s = synchronized_streams(t)
            w, dropped = build_windows(t, s)
            lag = alignment_lag(t, s)
        except NotImplementedError:
            raise
        except Exception as e:                                   # one bad trial is reported, not fatal
            failures.append(f'{t.key}: {type(e).__name__}: {e}')
            continue
        windows += w
        rows_original = len(read_joints(t.joints)[0])
        mp4 = video_frames(t.video) if t.video else None
        per_trial.append({'key': t.key, 'body': t.body, 'video_cut': t.video_cut, 'imu_cut': t.imu_cut,
                          'frames': len(s['joints']), 'imu_rows': len(s['imu']),
                          'video_s': round(len(s['joints']) / 30, 3), 'imu_s': round(len(s['imu']) / 50, 3),
                          'tracked_fraction': round(float(s['tracked'].mean()), 4), 'windows': len(w),
                          'dropped': dropped, 'lag_frames': int(lag), 'mp4_frames': mp4,
                          'joint_rows_original': rows_original})
        if mp4 is not None and mp4 != rows_original:
            problems.append(f'{t.key}: the .mp4 has {mp4} frames, the original joints {rows_original} rows')

    splits = split_by_subject([t.subject for t in trials], seed=args.seed)
    host = socket.gethostname().split('.')[0]
    os.makedirs(args.out, exist_ok=True)
    write_index(windows, splits, os.path.join(args.out, f'windows-{host}.csv'))
    lags = collections.Counter(r['lag_frames'] for r in per_trial)
    summary = {'test': 'sync_windows', 'host': socket.gethostname(), 'root': args.root, 'video_size': args.video_size,
               'seed': args.seed, 'trials': len(trials), 'trials_failed': len(failures),
               'windows': len(windows), 'windows_dropped': sum(r['dropped'] for r in per_trial),
               'windows_by_split': dict(collections.Counter(splits[w.subject] for w in windows)),
               'subjects_by_split': {k: sorted(s for s, v in splits.items() if v == k)
                                     for k in ('train', 'val', 'test')},
               'lag_frames_counts': {str(k): v for k, v in sorted(lags.items())},
               'problems': problems, 'failures': failures, 'per_trial': per_trial}
    with open(os.path.join(args.out, f'summary-{host}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"{len(windows)} windows ({summary['windows_by_split']}), {summary['windows_dropped']} dropped")
    print(f"alignment lag in frames, number of trials: {summary['lag_frames_counts']}")
    for line in [r for r in per_trial if abs(r['lag_frames']) > 1][:20]:
        print(f"  lag {line['lag_frames']:+d}: {line['key']} (cuts: video {line['video_cut']}, IMU {line['imu_cut']})")
    for line in (problems + failures)[:30]:
        print('  problem:', line)
    print(f"wrote {args.out}/windows-{host}.csv and summary-{host}.json")


if __name__ == '__main__':
    main()
