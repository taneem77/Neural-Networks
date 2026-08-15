# metrics.py
# Reusable evaluation metrics for benchmarking

import numpy as np


def confusion_counts(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN


def precision(tp, fp):
    return tp / (tp + fp + 1e-9)


def recall(tp, fn):
    return tp / (tp + fn + 1e-9)


def f1(p, r):
    return 2 * p * r / (p + r + 1e-9)


def accuracy(tp, tn, total):
    return (tp + tn) / total


def evaluate(y_true, y_pred):

    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

    p = precision(tp, fp)
    r = recall(tp, fn)

    return {
        "accuracy": accuracy(tp, tn, len(y_true)),
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "confusion": [[tn, fp], [fn, tp]]
    }