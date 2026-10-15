"""Statistics for the fusion comparison: macro-F1, per-subject scores, a bootstrap over subjects, and an exact
Wilcoxon signed-rank test.
numpy only."""
import numpy as np


def macro_f1(y_true, y_pred):
    """Macro-F1 over the classes present in y_true or y_pred (so predicting an absent class costs)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.union1d(y_true, y_pred)
    f1 = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)) if f1 else float('nan')


def bootstrap_mean(values, n=10000, seed=0):
    """Mean of per-subject values with a percentile 95% interval from resampling subjects."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def average_ranks(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank p-value for paired differences d (zeros dropped, tied ranks averaged),
    by counting every sign pattern. Returns (W+, n used, p)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0, 1.0
    r2 = np.rint(2 * average_ranks(np.abs(d))).astype(int)          # doubled ranks are integers even with ties
    w_plus2 = int(r2[d > 0].sum())
    total2 = int(r2.sum())
    counts = np.zeros(total2 + 1, dtype=object)                     # exact integer counts of each W+ (doubled)
    counts[0] = 1
    for r in r2:
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[:len(counts) - r]
        counts = counts + shifted
    sums = np.arange(total2 + 1)
    extreme = np.abs(sums - total2 / 2) >= abs(w_plus2 - total2 / 2) - 1e-9
    p = float(sum(counts[extreme]) / sum(counts))
    return w_plus2 / 2, n, min(1.0, p)


__all__ = ['macro_f1', 'bootstrap_mean', 'average_ranks', 'wilcoxon_exact']
