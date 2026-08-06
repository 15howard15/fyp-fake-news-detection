"""
metrics.py — Shared evaluation (Section 3.6). All five metrics in one place.

Every training script calls compute_metrics() so the numbers are computed
identically across LR/SVM/CNN/BERT.
"""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

import config as cfg


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    y_true : array of 0/1
    y_pred : array of 0/1 (hard predictions)
    y_prob : array of P(fake) in [0,1] for AUC-ROC. If None, AUC is skipped.
    """
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            m["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            m["auc_roc"] = float("nan")
    else:
        m["auc_roc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return m


def save_metrics(metrics: dict, model_name: str, composition: str):
    """Write one JSON per (model, composition) into results/."""
    out = {"model": model_name, "composition": composition, **metrics}
    path = cfg.RESULTS_DIR / f"metrics_{model_name}_{composition}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def print_metrics(metrics, header=""):
    if header:
        print(header)
    for k in ("accuracy", "precision", "recall", "f1", "auc_roc"):
        v = metrics.get(k, float("nan"))
        print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion : {metrics['confusion']}")
