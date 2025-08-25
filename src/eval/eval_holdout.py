# def/src/eval/eval_holdout.py
# Evalúa un modelo+umbrales sobre un CSV hold-out con métricas globales y por clase.

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from src.infer.predictor import MedicalPredictor

def binarize(sets, labels):
    return np.array([[1 if l in s else 0 for l in labels] for s in sets], dtype=int)

def main(args):
    pred = MedicalPredictor(
        model_path=args.model,
        thresholds_path=args.thresholds,
        labels_path=args.labels,
        encoder_name=args.encoder,
        default_max_length=args.max_len
    )

    df = pd.read_csv(args.csv, sep=";")
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    y_true_sets = df["group"].astype(str).str.split("|").apply(set)

    y_pred_sets = [set(p) for p in pred.predict_batch(texts, batch_size=args.batch_size, max_length=args.max_len)]
    labels = pred.labels

    y_true = binarize(y_true_sets, labels)
    y_pred = binarize(y_pred_sets, labels)

    print("\n=== MÉTRICAS GLOBALES (HOLD-OUT) ===")
    print(f"macro_f1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    print("\n=== POR CLASE ===")
    for cls, P, R, F in zip(labels, p, r, f1):
        print(f"{cls:15s}  P={P:.3f}  R={R:.3f}  F1={F:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="def/data/test.csv")
    ap.add_argument("--model", default="outputs/b3/final_model.pt")
    ap.add_argument("--thresholds", default="outputs/b3/thresholds.npy")
    ap.add_argument("--labels", default="outputs/b3/labels.json")
    ap.add_argument("--encoder", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    main(args)
