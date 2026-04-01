from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def _empty_metrics() -> Dict[str, float]:
    return {
        "AUPRC": 0.0,
        "AUROC": 0.0,
        "TPR_at_10FDR": 0.0,
        "FPR_at_10FDR": 0.0,
        "Actual_FDR": 0.0,
        "Actual_Precision": 0.0,
        "Threshold": 0.0,
    }


def compute_identification_metrics(
    y_true: Any,
    y_score: Any,
    target_fdr: float = 0.1,
) -> Dict[str, float]:
    """Compute anomaly-identification metrics with an FDR-controlled threshold.

    Parameters
    ----------
    y_true:
        Binary labels (0: in-reference, 1: abnormal/OOR).
    y_score:
        Continuous anomaly scores (higher means more abnormal).
    target_fdr:
        Desired false discovery rate for selecting a decision threshold.

    Returns
    -------
    dict
        Flat metric dictionary for direct DataFrame integration.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.size == 0 or y_score.size == 0 or y_true.size != y_score.size:
        return _empty_metrics()

    if np.unique(y_true).size < 2:
        return _empty_metrics()

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        selected_threshold = float(np.max(y_score))
    else:
        precision_by_threshold = precisions[:-1]
        valid_idx = np.where(precision_by_threshold >= (1 - target_fdr))[0]
        if valid_idx.size > 0:
            selected_threshold = float(thresholds[valid_idx[0]])
        else:
            selected_threshold = float(thresholds[np.argmax(precision_by_threshold)])

    y_pred = (y_score >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    actual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    actual_fdr = 1 - actual_precision

    return {
        "AUPRC": float(auprc),
        "AUROC": float(auroc),
        "TPR_at_10FDR": float(tpr),
        "FPR_at_10FDR": float(fpr),
        "Actual_FDR": float(actual_fdr),
        "Actual_Precision": float(actual_precision),
        "Threshold": float(selected_threshold),
    }
