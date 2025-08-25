# def/src/train/calibration.py
import numpy as np

def _bce_loss(probs, y, eps=1e-8):
    return -np.mean(y*np.log(probs+eps) + (1-y)*np.log(1-probs+eps))

def tune_temperature(y_true, y_logits, T_grid=np.linspace(0.7, 2.5, 19)):
    # y_logits: logits en validación (sin sigmoid)
    best_T, best_loss = 1.0, float("inf")
    for T in T_grid:
        probs = 1.0 / (1.0 + np.exp(-y_logits / T))
        loss = _bce_loss(probs, y_true)
        if loss < best_loss:
            best_loss, best_T = loss, T
    return best_T
