"""Train every model of one leave-one-subject-out fold, and write each run's test predictions.

    python scripts/fusion_train.py --data results/fusion/windows.npz --fold 1

Fold k tests the k-th subject in sorted order and stops early on the next one. Each half, model and seed writes
<out>/<half>/<model>/<test subject>-seed<s>.json (settings, sizes, scores) and .npz (test indices, labels and class
probabilities). Late fusion isn't trained: the report averages the joints and IMU models' probabilities.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from fusion_scale.fusion.data import (Standardizer, check_channels, folds, half_view, imu_features,
                                         joint_features, load, mask_features, position_features, trial_shuffle)
from fusion_scale.fusion.metrics import macro_f1
from fusion_scale.fusion.models import Intermediate, Single, VectorMLP, parameters

MODELS = {'lower': ['joints', 'imu', 'intermediate', 'shuffled', 'mask', 'position'],
          'upper': ['joints', 'imu', 'intermediate', 'shuffled', 'mask', 'position'],
          'all': ['intermediate', 'mask']}
SETTINGS = {'epochs': 60, 'min_epochs': 20, 'patience': 10, 'batch': 64, 'lr': 1e-3, 'weight_decay': 1e-2,
            'optimizer': 'AdamW',
            'loss': 'cross-entropy weighted by inverse class frequency in training',
            'early_stopping': 'best validation macro-F1, validation subject only; not before min_epochs, '
                              'since a small fold can sit at one class for its first epochs'}


def file_sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(d, half, rows, train):
    """Every input this half's models can use, for the half's rows; statistics from training rows only.

    joints_t, joints_s   the joints' motion (N, 60, 103) and posture (N, 102)
    imu_t, imu_s         the IMU's motion and mask (N, 100, 2C) and posture (N, C)
    position             the hip's path in camera coordinates (N, 60, 4), for the negative control
    mask                 which channels are measured and how much was tracked (N, C + 1), for the negative control
    """
    tr = d['tracked'][rows].astype(bool)
    jt, js = joint_features(d['joints'][rows], tr, d['joint_names'])
    jmask = np.broadcast_to(tr[:, :, None], jt.shape)
    sjt = Standardizer().fit(jt[train], jmask[train], keep=[jt.shape[-1] - 1])
    sjs = Standardizer().fit(js[train])
    it, im, is_ = imu_features(d['imu'][rows], d['imu_names'], half)
    sit = Standardizer().fit(it[train], im[train])
    sis = Standardizer().fit(is_[train])
    pf = position_features(d['joints'][rows], tr, d['joint_names'])
    pmask = np.broadcast_to(tr[:, :, None], pf.shape)
    sp = Standardizer().fit(pf[train], pmask[train], keep=[pf.shape[-1] - 1])
    return {'joints_t': sjt.transform(jt, jmask), 'joints_s': sjs.transform(js),
            'imu_t': np.concatenate([sit.transform(it, im), im.astype(np.float32)], axis=2),
            'imu_s': sis.transform(is_),
            'position': sp.transform(pf, pmask),
            'mask': mask_features(d['imu'][rows], d['imu_names'], tr, half)}


def build(name, F, classes):
    """The model and the inputs it takes, in order."""
    if name in ('intermediate', 'shuffled'):
        return (Intermediate(F['joints_t'].shape[-1], F['joints_s'].shape[-1], F['imu_t'].shape[-1],
                             F['imu_s'].shape[-1], classes), ['joints_t', 'joints_s', 'imu_t', 'imu_s'])
    if name == 'joints':
        return Single(F['joints_t'].shape[-1], classes, F['joints_s'].shape[-1]), ['joints_t', 'joints_s']
    if name == 'imu':
        return Single(F['imu_t'].shape[-1], classes, F['imu_s'].shape[-1]), ['imu_t', 'imu_s']
    if name == 'position':
        return Single(F['position'].shape[-1], classes), ['position']
    if name == 'mask':
        return VectorMLP(F['mask'].shape[-1], classes), ['mask']
    raise ValueError(name)


def predict(model, xs, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xs[0]), batch):
            out.append(torch.softmax(model(*[x[i:i + batch] for x in xs]), 1).cpu())
    return torch.cat(out).numpy()


def train(model, xs_tr, y_tr, xs_va, y_va, classes, seed, s):
    counts = np.bincount(y_tr, minlength=classes).astype(float)
    weight = np.where(counts > 0, counts.sum() / (classes * np.maximum(counts, 1)), 0.0)
    device = xs_tr[0].device
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
    g = torch.Generator().manual_seed(seed)
    y = torch.as_tensor(y_tr, device=device)
    best, best_state, best_epoch, bad, epoch = -1.0, None, -1, 0, -1
    for epoch in range(s['epochs']):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(device)
        for i in range(0, len(y), s['batch']):
            idx = perm[i:i + s['batch']]
            if len(idx) < 2:                      # batch norm needs two samples
                continue
            loss = loss_fn(model(*[x[idx] for x in xs_tr]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        f1 = macro_f1(y_va, predict(model, xs_va).argmax(1))
        if f1 > best + 1e-9:
            best, best_state, best_epoch, bad = f1, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= s['patience'] and epoch + 1 >= s['min_epochs']:
                break
    model.load_state_dict(best_state)
    return best, best_epoch, epoch + 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', required=True)
    p.add_argument('--fold', type=int, required=True, help='1-based: the k-th subject in sorted order is the test')
    p.add_argument('--halves', nargs='+', default=['lower', 'upper', 'all'], choices=['lower', 'upper', 'all'])
    p.add_argument('--models', nargs='+', default=None, help='default: every model of each half (MODELS)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=None, help='a shorter check only; the experiment uses SETTINGS')
    p.add_argument('--out', default='results/fusion/runs')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    p.add_argument('--device', default='auto')
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if a.device == 'auto' else a.device))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    settings = dict(SETTINGS, **({'epochs': a.epochs, 'min_epochs': min(SETTINGS['min_epochs'], a.epochs)}
                                 if a.epochs else {}))
    d = load(a.data)
    subjects = d['subjects'].astype(str)
    plan = list(folds(subjects))
    test, val, train_subjects = plan[a.fold - 1]
    channels = check_channels()
    if channels and channels != 'match':
        raise SystemExit(channels)
    meta = {'data': os.path.abspath(a.data), 'data_sha256': file_sha256(a.data), 'host': socket.gethostname(),
            'torch': torch.__version__, 'device': str(device), 'python': platform.python_version(),
            'job': os.environ.get('SLURM_JOB_ID', ''), 'task': os.environ.get('SLURM_ARRAY_TASK_ID', ''),
            'channels_vs_windows': channels or 'not checked (sync_windows not importable)'}
    try:
        meta['code_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                             timeout=10).stdout.strip()
    except Exception:
        meta['code_commit'] = ''
    print(f'fold {a.fold}: test {test}, validation {val}, {len(train_subjects)} training subjects, device {device}')

    for half in a.halves:
        rows, y, classes = half_view(d, half)
        subj, keys = subjects[rows], d['keys'][rows].astype(str)
        tr, va, te = np.isin(subj, train_subjects), subj == val, subj == test
        assert not (set(subj[tr]) & {test, val}) and te.any() and va.any(), 'a subject crossed splits'
        F = prepare(d, half, rows, tr)
        for name in (a.models or MODELS[half]):
            if name not in MODELS[half]:
                continue
            for seed in a.seeds:
                out = os.path.join(a.out, half, name, f'{test}-seed{seed}')
                if os.path.exists(out + '.json') and not a.force:
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                y_tr, y_va = y[tr], y[va]
                if name == 'shuffled':
                    rng = np.random.default_rng(1000 + seed)
                    y_tr, y_va = trial_shuffle(y_tr, keys[tr], rng), trial_shuffle(y_va, keys[va], rng)
                model, uses = build(name, F, classes)
                model.to(device)
                tensors = {k: torch.as_tensor(F[k], device=device) for k in uses}
                def pick(m):
                    return [tensors[k][torch.as_tensor(np.flatnonzero(m), device=device)] for k in uses]
                t0 = time.time()
                val_f1, best_epoch, epochs_run = train(model, pick(tr), y_tr, pick(va), y_va, classes, seed, settings)
                probs = predict(model, pick(te))
                y_te = y[te]
                result = {'experiment': 'fusion', 'half': half, 'model': name, 'test_subject': test,
                          'validation_subject': val, 'training_subjects': train_subjects, 'seed': seed,
                          'classes': classes, 'windows': {'train': int(tr.sum()), 'validation': int(va.sum()),
                                                          'test': int(te.sum())},
                          'parameters': parameters(model), 'settings': settings, 'best_epoch': best_epoch,
                          'epochs_run': epochs_run, 'validation_macro_f1': val_f1,
                          'test_macro_f1': macro_f1(y_te, probs.argmax(1)),
                          'test_accuracy': float((probs.argmax(1) == y_te).mean()),
                          'seconds': round(time.time() - t0, 1), **meta}
                np.savez(out + '.npz', index=rows[te], y_true=y_te, probs=probs.astype(np.float32))
                with open(out + '.json', 'w') as f:
                    json.dump(result, f, indent=1)
                print(f"{half:5s} {name:12s} seed {seed}: test macro-F1 {result['test_macro_f1']:.3f} "
                      f"(validation {val_f1:.3f}, epoch {best_epoch}, {result['seconds']} s)", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
