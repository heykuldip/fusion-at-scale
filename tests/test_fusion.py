"""The fusion comparison's checks: the statistics, the folds, and the rules that keep the models from taking
shortcuts."""
import importlib
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from fusion_synthetic import make
from fusion_scale.fusion.data import (folds, half_view, imu_features, joint_features, load,
                                     trial_shuffle, UPPER_CHANNELS)
from fusion_scale.fusion.metrics import average_ranks, bootstrap_mean, macro_f1, wilcoxon_exact


def test_macro_f1_by_hand():
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert macro_f1([0, 0], [1, 1]) == 0.0                     # a predicted class absent from the truth costs


def brute_force_p(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    r = average_ranks(np.abs(d))
    obs = r[d > 0].sum()
    total = r.sum()
    sums = [sum(r[i] for i in range(len(d)) if signs[i]) for signs in itertools.product([0, 1], repeat=len(d))]
    return np.mean([abs(s - total / 2) >= abs(obs - total / 2) - 1e-9 for s in sums])


@pytest.mark.parametrize('d', [[1, 2, 3, 4, 5], [0.5, -1, 1.5, 2, -2.5, 3, 3.5, -4, 4.5], [1, 1, -2, 2, 3, -3, 4, 5],
                               [0.1, 0, -0.3, 0.2, 0.2, -0.1]])
def test_wilcoxon_exact_matches_enumeration(d):
    assert wilcoxon_exact(d)[2] == pytest.approx(brute_force_p(d))


def test_wilcoxon_all_positive_five():
    assert wilcoxon_exact([1, 2, 3, 4, 5])[2] == pytest.approx(2 / 32)


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    v = [0.6, 0.7, 0.8, 0.65, 0.9]
    m, lo, hi = bootstrap_mean(v)
    assert bootstrap_mean(v) == (m, lo, hi) and lo <= m <= hi


def test_folds_test_every_subject_once_and_never_mix():
    subjects = [f'S{k}' for k in range(40, 58) if k not in (43, 45)]
    plan = list(folds(subjects))
    assert sorted(t for t, _, _ in plan) == sorted(subjects)
    for t, v, train in plan:
        assert v != t and t not in train and v not in train and len(train) == len(subjects) - 2


def test_joint_features_ignore_where_and_how_big():
    rng = np.random.default_rng(0)
    names = ['left_hip', 'right_hip', 'neck'] + [f'j{k}' for k in range(31)]
    j = rng.normal(0, 200, (4, 60, 34, 3))
    t = rng.random((4, 60)) > 0.1
    motion, posture = joint_features(j, t, names)
    for other in (j + np.array([1500.0, -300.0, 4000.0]), 2.5 * j):     # moved, and a bigger person
        m2, p2 = joint_features(other, t, names)
        assert np.allclose(motion, m2, atol=1e-5) and np.allclose(posture, p2, atol=1e-5)
    assert np.all(motion[~t][:, :-1] == 0) and np.all(motion[..., -1] == t)
    assert np.allclose(motion[..., :-1].sum(1)[t.any(1)], 0, atol=1e-4)  # motion has no window mean left


def test_imu_features_use_the_half_s_fixed_channels(tmp_path):
    d = load(make(str(tmp_path / 'w.npz'), n_subjects=3))
    x, m, posture = imu_features(d['imu'], d['imu_names'], 'upper')
    assert x.shape[-1] == len(UPPER_CHANNELS) == posture.shape[-1] and np.all(x[~m] == 0)
    rows, y, k = half_view(d, 'upper')
    assert k == 9 and m[rows].all()                             # every upper window has every upper channel


def test_trial_shuffle_keeps_each_trial_whole():
    keys = np.array(['a'] * 3 + ['b'] * 2 + ['c'] * 4 + ['d'])
    labels = np.array([0] * 3 + [1] * 2 + [2] * 4 + [3])
    out = trial_shuffle(labels, keys, np.random.default_rng(1))
    for k in np.unique(keys):
        assert len(set(out[keys == k])) == 1
    assert sorted(out[np.unique(keys, return_index=True)[1]]) == [0, 1, 2, 3]


def test_statistics_come_from_training_rows_only(tmp_path):
    train_mod = importlib.import_module('fusion_train')
    d = load(make(str(tmp_path / 'w.npz'), n_subjects=4))
    rows, y, k = half_view(d, 'lower')
    subj = d['subjects'][rows].astype(str)
    train = subj != 'S43'
    a = train_mod.prepare(d, 'lower', rows, train)
    d2 = dict(d)
    d2['joints'] = d['joints'].copy()
    d2['imu'] = d['imu'].copy()
    test_rows = rows[~train]
    d2['joints'][test_rows] *= 7.0
    d2['joints'][test_rows] += 999.0
    d2['imu'][test_rows] *= -3.0
    b = train_mod.prepare(d2, 'lower', rows, train)
    for key in ('joints_t', 'joints_s', 'imu_t', 'imu_s', 'position', 'mask'):
        assert np.array_equal(a[key][train], b[key][train]), key


def test_one_fold_end_to_end(tmp_path):
    path = make(str(tmp_path / 'w.npz'), n_subjects=5)
    train_mod = importlib.import_module('fusion_train')
    out = str(tmp_path / 'runs')
    assert train_mod.main(['--data', path, '--fold', '1', '--seeds', '0', '--epochs', '2', '--out', out,
                           '--threads', '2']) == 0
    for half, models in train_mod.MODELS.items():
        for m in models:
            r = json.load(open(os.path.join(out, half, m, 'S40-seed0.json')))
            assert r['test_subject'] == 'S40' and r['validation_subject'] == 'S41'
            assert 'S40' not in r['training_subjects']
    here = os.path.dirname(__file__)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(here, '..', 'src'), here]))
    report = os.path.join(here, '..', 'scripts', 'fusion_report.py')
    subprocess.run([sys.executable, report, '--data', path, '--runs', out],
                   check=True, env=env, capture_output=True)
    s = json.load(open(tmp_path / 'summary.json'))
    assert {'lower', 'upper', 'all'} <= set(s['halves']) and 'late' in s['halves']['lower']['models']
    assert 'intermediate - late' in s['halves']['upper']['paired']


def test_streams_learn_and_controls_stay_at_chance(tmp_path):
    """Each stream alone should learn a signal any new subject shows (frequencies 0.5 Hz apart), while shuffled labels
    and the channel mask stay near chance. Lower half: chance is 0.25 for random guesses and 0.10 for always one class.
    """
    path = make(str(tmp_path / 'w.npz'), n_subjects=6, trials=3, windows=6)
    train_mod = importlib.import_module('fusion_train')
    out = str(tmp_path / 'runs')
    train_mod.main(['--data', path, '--fold', '1', '--halves', 'lower', '--models', 'joints', 'imu', 'shuffled',
                    'mask', '--seeds', '0', '--epochs', '40', '--out', out, '--threads', '2'])
    score = {m: json.load(open(os.path.join(out, 'lower', m, 'S40-seed0.json')))['test_macro_f1']
             for m in ('joints', 'imu', 'shuffled', 'mask')}
    assert score['joints'] >= 0.75 and score['imu'] >= 0.75, score
    assert score['shuffled'] <= 0.4 and score['mask'] <= 0.3, score
