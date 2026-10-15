"""Throughput and weak-scaling efficiency for each GPU type and count, from results/scaling/*.json.

    python scripts/scaling_report.py      # prints the table; writes results/scaling/summary.csv, and scaling.png

Runs are grouped by GPU, model size and per-GPU batch; profiled runs are left out of the timings.
"""
import csv
import glob
import json
import statistics
from collections import defaultdict


def efficiency(throughput_n: float, throughput_1: float, n: int) -> float:
    """Weak-scaling efficiency at n GPUs, as a fraction: throughput at n over n times the throughput on one (1.0 is
    perfect).
    """
    if n <= 0 or throughput_1 <= 0:
        raise ValueError("n and throughput_1 must be positive")
    return throughput_n / (n * throughput_1)


runs = defaultdict(list)
for path in sorted(glob.glob('results/scaling/*.json')):
    r = json.load(open(path))
    if r.get('test') != 'scaling' or r.get('profile'):
        continue
    runs[(r['device'], r['width'], r['depth'], r['per_gpu_batch'], r['n_gpus'])].append(r)

rows = []
for (device, width, depth, batch, n), rs in sorted(runs.items()):
    tput = [r['samples_per_s'] for r in rs]
    one = runs.get((device, width, depth, batch, 1))
    base = statistics.mean(r['samples_per_s'] for r in one) if one else None
    rows.append({'device': device, 'width': width, 'depth': depth, 'per_gpu_batch': batch, 'n_gpus': n, 'runs': len(rs),
                 'samples_per_s_mean': round(statistics.mean(tput), 1),
                 'samples_per_s_sd': round(statistics.stdev(tput), 1) if len(tput) > 1 else 0.0,
                 'step_ms_mean': round(1e3 * statistics.mean(r['step_s_mean'] for r in rs), 3),
                 'efficiency': round(efficiency(statistics.mean(tput), base, n), 3) if base else None})

for r in rows:
    print(r)
if rows:
    with open('results/scaling/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for key in sorted({(r['device'], r['width'], r['depth'], r['per_gpu_batch']) for r in rows}):
            pts = sorted((r['n_gpus'], r['samples_per_s_mean'], r['samples_per_s_sd']) for r in rows
                         if (r['device'], r['width'], r['depth'], r['per_gpu_batch']) == key)
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker='o', capsize=3,
                        label=f'{key[0]}, width {key[1]}, depth {key[2]}')
        ax.set_xlabel('GPUs')
        ax.set_ylabel('samples per second')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig('results/scaling/scaling.png', dpi=150)
        print('wrote results/scaling/summary.csv and results/scaling/scaling.png')
    except ImportError:
        print('wrote results/scaling/summary.csv (no matplotlib, so no plot)')
