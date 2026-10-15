"""Each inference variant's rate, time per batch, compile time, and keypoint drift against eager fp32.

    python scripts/inference_latency_report.py      # prints the table; writes results/inference_latency/summary.csv
"""
import csv
import glob
import json

from fusion_scale.pose.pipeline import FRAMES_AT_SCALE

rows = []
for path in sorted(glob.glob('results/inference_latency/*.json')):
    r = json.load(open(path))
    if r.get('test') != 'inference_latency':
        continue
    row = {'gpu': r.get('gpu'), 'config': r['config'], 'job': r['slurm_job']}
    if 'error' in r:
        row['error'] = r['error'][:120]
    else:
        same, big = r['vs_eager_fp32_same_size'], r['vs_eager_fp32_size_800']
        fps = r['frames_per_s']
        row.update({'frames_per_s': round(fps, 1), 'ms_per_batch': round(r['ms_per_batch_median'], 2),
                    'ms_per_batch_p95': round(r['ms_per_batch_p95'], 2),
                    'compile_s': round(r['compile_s'], 1) if r['compile_s'] else None,
                    'gpu_hours_at_scale': round(FRAMES_AT_SCALE / fps / 3600, 1),
                    'px_vs_eager': same['keypoint_px_mean'], 'other_person_vs_eager': same['different_person'],
                    'missed_vs_eager': same['found_by_one_only'], 'px_vs_size_800': big['keypoint_px_mean']})
    rows.append(row)
for row in rows:
    print(row)
if rows:
    names = sorted({k for row in rows for k in row}, key=lambda k: (k not in ('gpu', 'config', 'job'), k))
    with open('results/inference_latency/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    print('wrote results/inference_latency/summary.csv')
