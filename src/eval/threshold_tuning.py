# def/src/eval/threshold_tuning.py
import numpy as np
from sklearn.metrics import precision_recall_curve

def tune_thresholds_for_precision(y_true, y_probs, labels, precision_targets, base=0.5):
    """
    precision_targets: dict, ej. {"cardiovascular":0.70}
    Retorna thresholds (array) ajustando sólo clases con target.
    """
    thrs = np.full(y_true.shape[1], base, dtype=float)
    name_to_idx = {n:i for i,n in enumerate(labels)}
    for name, target in precision_targets.items():
        i = name_to_idx[name]
        P, R, T = precision_recall_curve(y_true[:, i], y_probs[:, i])
        # T tiene len = len(P)-1; buscamos el menor umbral con P >= target
        idx = np.where(P[:-1] >= target)[0]
        if len(idx) > 0:
            thrs[i] = float(np.clip(T[idx[0]], 0.01, 0.99))
    return thrs
