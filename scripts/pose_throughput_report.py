"""Each pose-pipeline configuration's rates and limiting stage, and GPU-hours for 324 million frames.

    python scripts/pose_throughput_report.py        # prints the table; writes results/pose_throughput/summary.csv

Runs tagged gpuI-ofN (slurm/pose_throughput_multi.sbatch) are also summed per job: N GPUs working at once. The rates
are recomputed from each file's timings with throughput().
"""
import collections
import csv
import glob
import json
import re

from fusion_scale.pose.pipeline import FRAMES_AT_SCALE, throughput

rows, together = [], collections.defaultdict(list)
for path in sorted(glob.glob('results/pose_throughput/*.json')):
    r = json.load(open(path))
    if r.get('test') != 'pose_throughput':
        continue
    try:
        rates = throughput(r['result'])
    except Exception as e:
        print(f"skipped {path}: throughput() failed: {type(e).__name__}: {e}")
        continue
    row = {'gpu': r.get('gpu'), 'config': r['config'], 'job': r['slurm_job'],
           'people_share': round(r['result']['people'] / max(r['result']['frames'], 1), 3)}
    row.update({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rates.items()})
    rows.append(row)
    m = re.search(r'-of(\d+)', r['config'])
    if m:
        together[(r['slurm_job'], int(m.group(1)))].append(rates['end_to_end_fps'])
for (job, n), fps in sorted(together.items()):
    total = sum(fps)
    wall_hours = FRAMES_AT_SCALE / total / 3600                 # hours with all n GPUs working at once
    rows.append({'gpu': f'{n} GPUs together', 'config': f'job {job}, {len(fps)} of {n} processes reported', 'job': job,
                 'end_to_end_fps': round(total, 1), 'wall_hours_at_scale': round(wall_hours, 1),
                 'gpu_hours_at_scale': round(n * wall_hours, 1)})
for row in rows:
    print(row)
if rows:
    names = sorted({k for row in rows for k in row}, key=lambda k: (k not in ('gpu', 'config', 'job'), k))
    with open('results/pose_throughput/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    print('wrote results/pose_throughput/summary.csv')
