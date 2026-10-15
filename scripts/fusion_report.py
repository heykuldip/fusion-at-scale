"""The fusion results, with uncertainty over subjects, not windows.

    python scripts/fusion_report.py --data results/fusion/windows.npz --runs results/fusion/runs

For each half and model: each subject's macro-F1 (averaged over seeds), their mean with a bootstrap 95% interval over
subjects, at window level and by majority vote over each trial's windows, with all windows and with only those at
least 90% tracked. Late fusion averages the joints and IMU models' probabilities, run by run. Paired differences per
subject (intermediate minus late, and each against the better single stream) with bootstrap intervals and an exact
Wilcoxon signed-rank test, and the controls against chance. Writes <runs>/../summary.json and summary.md.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from fusion_scale.fusion.data import load
from fusion_scale.fusion.metrics import bootstrap_mean, macro_f1, wilcoxon_exact


def subject_scores(runs, data, level, min_tracked):
    """{subject: macro-F1 averaged over seeds} for one model's runs [(subject, seed, index, y, probs)]."""
    per = collections.defaultdict(list)
    for subject, seed, index, y, probs in runs:
        keep = data['tracked_fraction'][index] >= min_tracked
        if not keep.any():
            continue
        index, y, probs = index[keep], y[keep], probs[keep]
        if level == 'window':
            per[subject].append(macro_f1(y, probs.argmax(1)))
        else:
            keys = data['keys'][index].astype(str)
            trial_y, trial_p = [], []
            for k in np.unique(keys):
                m = keys == k
                trial_y.append(y[m][0])
                trial_p.append(probs[m].mean(0).argmax())
            per[subject].append(macro_f1(np.array(trial_y), np.array(trial_p)))
    return {s: float(np.mean(v)) for s, v in per.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', required=True)
    p.add_argument('--runs', default='results/fusion/runs')
    p.add_argument('--min-tracked', type=float, default=0.9)
    a = p.parse_args()
    data = load(a.data)
    runs = collections.defaultdict(list)                   # (half, model) -> [(subject, seed, index, y, probs)]
    info = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(a.runs, '*', '*', '*.json'))):
        r = json.load(open(path))
        z = np.load(path[:-5] + '.npz')
        runs[(r['half'], r['model'])].append((r['test_subject'], r['seed'], z['index'], z['y_true'], z['probs']))
        info[(r['half'], r['model'])].append(r)
    for half in sorted({h for h, _ in runs}):              # late fusion: the two single streams' probabilities
        j = {(s, sd): (i, y, pr) for s, sd, i, y, pr in runs.get((half, 'joints'), [])}
        m = {(s, sd): (i, y, pr) for s, sd, i, y, pr in runs.get((half, 'imu'), [])}
        for key in sorted(set(j) & set(m)):
            (ij, yj, pj), (ii, yi, pi) = j[key], m[key]
            assert np.array_equal(ij, ii) and np.array_equal(yj, yi), f'{half} {key}: windows differ'
            runs[(half, 'late')].append((key[0], key[1], ij, yj, (pj + pi) / 2))

    summary = {'min_tracked': a.min_tracked, 'halves': {}}
    lines = ['# Fusion results', '', 'Macro-F1 per subject (mean over 3 seeds), then the mean over subjects with a '
             'bootstrap 95% interval over subjects.', '']
    for half in [h for h in ('lower', 'upper', 'all') if any(k[0] == h for k in runs)]:
        classes = next(r['classes'] for r in info[next(k for k in info if k[0] == half)])
        H = {'classes': classes, 'random_guess': 1 / classes, 'always_one_class': 2 / (classes * (classes + 1)),
             'models': {}, 'paired': {}}
        lines += [f'## {half} ({classes} classes)', '',
                  f'Chance is the shuffled-label control below. For reference: guessing at random scores about '
                  f'{1 / classes:.3f}, and always predicting one class about {2 / (classes * (classes + 1)):.3f}.', '',
                  '| model | window level | trial level | window level, tracked >= {:.0%} '
                  '| seeds x subjects | seed SD |'
                  .format(a.min_tracked), '|---|---|---|---|---|---|']
        scores = {}
        for model in ('joints', 'imu', 'late', 'intermediate', 'shuffled', 'mask', 'position'):
            if (half, model) not in runs:
                continue
            R = runs[(half, model)]
            w = subject_scores(R, data, 'window', 0.0)
            t = subject_scores(R, data, 'trial', 0.0)
            wt = subject_scores(R, data, 'window', a.min_tracked)
            seeds = sorted({sd for _, sd, *_ in R})
            by_seed = [np.mean([macro_f1(y, pr.argmax(1)) for s, sd, i, y, pr in R if sd == k]) for k in seeds]
            scores[model] = w
            entry = {'window': bootstrap_mean(list(w.values())), 'trial': bootstrap_mean(list(t.values())),
                     'window_tracked': bootstrap_mean(list(wt.values())) if wt else None,
                     'subjects': len(w), 'seeds': len(seeds), 'runs': len(R),
                     'seed_sd_of_mean': float(np.std(by_seed)) if len(by_seed) > 1 else None,
                     'per_subject': w}
            H['models'][model] = entry
            def fmt(b):
                return f'{b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]' if b else '-'
            lines.append(f"| {model} | {fmt(entry['window'])} | {fmt(entry['trial'])} | {fmt(entry['window_tracked'])} "
                         f"| {len(seeds)} x {len(w)} | {entry['seed_sd_of_mean'] or 0:.3f} |")
        singles = [m for m in ('joints', 'imu') if m in scores]
        best = max(singles, key=lambda m: np.mean(list(scores[m].values()))) if singles else None
        pairs = [('intermediate', 'late')] + ([('intermediate', best), ('late', best)] if best else [])
        lines += ['', '| comparison (per subject) | mean difference [95% interval] | Wilcoxon p (exact) | subjects |',
                  '|---|---|---|---|']
        for x, yname in pairs:
            if x in scores and yname in scores:
                common = sorted(set(scores[x]) & set(scores[yname]))
                diff = [scores[x][s] - scores[yname][s] for s in common]
                b = bootstrap_mean(diff)
                wplus, n_used, pval = wilcoxon_exact(diff)
                H['paired'][f'{x} - {yname}'] = {'mean': b[0], 'low': b[1], 'high': b[2], 'wilcoxon_p': pval,
                                                 'subjects': len(common), 'per_subject': dict(zip(common, diff))}
                lines.append(f'| {x} - {yname} | {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] | {pval:.3g} | {len(common)} |')
        summary['halves'][half] = H
        lines.append('')
    out_dir = os.path.dirname(os.path.abspath(a.runs.rstrip('/')))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out_dir}/summary.json and summary.md')


if __name__ == '__main__':
    main()
