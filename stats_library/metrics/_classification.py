import numpy as np

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "balanced_accuracy_score",
]

def _check_inputs(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true.shape={y_true.shape} vs y_pred.shape={y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got y_true.ndim={y_true.ndim}")

def accuracy_score(y_true, y_pred):
    _check_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred):
    _check_inputs(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []

    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(precisions))


def recall_score(y_true, y_pred):
    _check_inputs(y_true, y_pred)
    labels = np.unique(y_true) 
    recalls = []

    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def f1_score(y_true, y_pred):
    _check_inputs(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_true == c) & (y_pred != c))

        if tp == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * (prec * rec) / (prec + rec))

    return float(np.mean(f1s))


def balanced_accuracy_score(y_true, y_pred):
    return recall_score(y_true, y_pred)
