"""Evaluation beyond accuracy: ranking quality, calibration, and threshold choice."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def expected_calibration_error(y, p, n_bins=15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            conf.append(float(p[m].mean()))
            acc.append(float(y[m].mean()))
            count.append(int(m.sum()))
    return conf, acc, count


def classification_report(y, p, threshold=0.5) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def threshold_for_precision(y_val, p_val, target_precision=0.98) -> float:
    """Lowest threshold whose validation precision meets the target.

    On a creator platform a false positive (a good article demoted as clickbait) is
    costlier than letting some clickbait through, so we fix precision first and
    take whatever recall that leaves. The threshold is chosen on validation only,
    then reported on the untouched test set.
    """
    prec, _, thr = precision_recall_curve(y_val, p_val)
    ok = np.flatnonzero(prec[:-1] >= target_precision)
    return float(thr[ok[0]]) if len(ok) else 0.5
