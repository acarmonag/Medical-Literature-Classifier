from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# ✅ Import absoluto dentro del paquete 'src'
from src.utils.text_cleaning import MedTextCleaner

REQUIRED_COLS = ("title", "abstract", "group")
SEP_RE = re.compile(r"\s*[|,;/]\s*")  # acepta | , ; /

def smart_read_csv(path: Path) -> pd.DataFrame:
    """
    Tolerant CSV reader for messy medical data.
    """
    candidates = [
        dict(sep=None, engine="python", encoding="utf-8-sig",
             quotechar='"', escapechar="\\", doublequote=True, dtype=str),
        dict(sep=",", engine="python", encoding="utf-8-sig",
             quotechar='"', escapechar="\\", doublequote=True, dtype=str),
        dict(sep=";", engine="python", encoding="utf-8-sig",
             quotechar='"', escapechar="\\", doublequote=True, dtype=str),
        dict(sep="\t", engine="python", encoding="utf-8-sig",
             quotechar='"', escapechar="\\", doublequote=True, dtype=str),
    ]
    last_err = None
    for opts in candidates:
        try:
            df = pd.read_csv(path, **opts)
            if set(REQUIRED_COLS).issubset(df.columns):
                return df
        except Exception as e:
            last_err = e
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                         on_bad_lines="skip", dtype=str)
        if set(REQUIRED_COLS).issubset(df.columns):
            return df
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Unable to read CSV with required columns {REQUIRED_COLS}. Last error: {last_err}")

def load_data(csv_path: Path) -> pd.DataFrame:
    df = smart_read_csv(csv_path)
    for col in REQUIRED_COLS:
        df[col] = df[col].fillna("").astype(str)
    # split multilabels (|, , ; /)
    df["labels"] = df["group"].astype(str).apply(
        lambda s: sorted({x.strip().lower() for x in SEP_RE.split(s) if x.strip()})
    )
    df["text"] = df["title"] + " " + df["abstract"]
    return df

def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("clean", MedTextCleaner(
            drop_citations=True,
            strip_quotes=True,
            remove_digits="all",
            preserve_hyphen_numbers=True,
            lower=True,
            drop_urls_emails=True,
            map_greek_letters=True,
        )),
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words="english",
            sublinear_tf=True,     # pequeño boost
            min_df=2               # reduce ruido
        )),
        # liblinear + balanced suele mejorar recall en clases minoritarias
        ("clf", OneVsRestClassifier(LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced"
        )))
    ])

def optimize_thresholds(y_true_bin: np.ndarray, scores: np.ndarray, classes: List[str]) -> Dict[str, float]:
    """
    Busca umbral por clase que maximiza F1 en el split de validación.
    """
    from sklearn.metrics import f1_score
    thresholds: Dict[str, float] = {}
    n_classes = y_true_bin.shape[1]
    for j in range(n_classes):
        s = scores[:, j]
        # Si no hay varianza, usa 0.0
        if np.allclose(s, s[0]):
            thresholds[classes[j]] = 0.0
            continue
        # grid compacto por cuantiles
        grid = np.unique(np.quantile(s, np.linspace(0.1, 0.9, 17)))
        best_t, best_f1 = 0.0, -1.0
        y_true_j = y_true_bin[:, j]
        for t in grid:
            y_hat = (s >= t).astype(int)
            f1 = f1_score(y_true_j, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[classes[j]] = best_t
    return thresholds

def apply_thresholds(scores: np.ndarray, classes: List[str], thresholds: Dict[str, float]) -> np.ndarray:
    """
    Aplica umbrales por clase a una matriz de scores (n_samples, n_classes).
    """
    T = np.array([thresholds.get(cls, 0.0) for cls in classes], dtype=float)  # default 0.0
    return (scores >= T).astype(int)

def main(args=None):
    parser = argparse.ArgumentParser(description="Train multilabel medical text classifier (TF-IDF + LR)")
    parser.add_argument("--csv", type=Path, default=Path("data/medical_lit.csv"), help="Path to CSV with title, abstract, group")
    parser.add_argument("--outdir", type=Path, default=Path("models"), help="Output directory for artifacts")
    ns = parser.parse_args(args=args)

    df = load_data(ns.csv)
    X = df["text"].tolist()
    y = df["labels"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin  = mlb.transform(y_val)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train_bin)

    # Pred por defecto (umbral 0.0)
    try:
        y_val_pred = pipe.predict(X_val)
        scores_val = pipe.decision_function(X_val)
    except Exception:
        # fallback si no hay decision_function
        prob = getattr(pipe, "predict_proba", None)
        if prob:
            scores_val = prob(X_val)
            y_val_pred = (scores_val >= 0.5).astype(int)
        else:
            y_val_pred = pipe.predict(X_val)
            scores_val = y_val_pred.astype(float)

    # Reporte estándar
    report = classification_report(y_val_bin, y_val_pred, target_names=mlb.classes_, zero_division=0)
    micro_f1 = f1_score(y_val_bin, y_val_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_val_bin, y_val_pred, average='macro', zero_division=0)
    micro_p  = precision_score(y_val_bin, y_val_pred, average='micro', zero_division=0)
    micro_r  = recall_score(y_val_bin, y_val_pred, average='micro', zero_division=0)

    print("\n=== Classification Report (threshold=0.0) ===\n")
    print(report)
    print(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f} | Micro P: {micro_p:.4f} | Micro R: {micro_r:.4f}")

    # 🔧 Tuning de umbrales por clase
    thresholds = optimize_thresholds(y_val_bin, scores_val, mlb.classes_.tolist())
    y_val_pred_tuned = apply_thresholds(scores_val, mlb.classes_.tolist(), thresholds)

    report2 = classification_report(y_val_bin, y_val_pred_tuned, target_names=mlb.classes_, zero_division=0)
    micro_f1_2 = f1_score(y_val_bin, y_val_pred_tuned, average='micro', zero_division=0)
    macro_f1_2 = f1_score(y_val_bin, y_val_pred_tuned, average='macro', zero_division=0)
    micro_p2  = precision_score(y_val_bin, y_val_pred_tuned, average='micro', zero_division=0)
    micro_r2  = recall_score(y_val_bin, y_val_pred_tuned, average='micro', zero_division=0)

    print("\n=== After per-class threshold tuning ===\n")
    print(report2)
    print(f"Micro F1: {micro_f1_2:.4f} | Macro F1: {macro_f1_2:.4f} | Micro P: {micro_p2:.4f} | Micro R: {micro_r2:.4f}")

    # Guardar artefactos
    ns.outdir.mkdir(parents=True, exist_ok=True)
    pipe_path = ns.outdir / "tfidf_lr_med_pipeline.joblib"
    mlb_path  = ns.outdir / "label_binarizer.joblib"
    th_path   = ns.outdir / "thresholds.json"
    metrics_path = ns.outdir / "metrics.json"

    joblib.dump(pipe, pipe_path)
    joblib.dump(mlb, mlb_path)
    with open(th_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold0": {"micro_f1": micro_f1, "macro_f1": macro_f1, "micro_p": micro_p, "micro_r": micro_r},
            "tuned": {"micro_f1": micro_f1_2, "macro_f1": macro_f1_2, "micro_p": micro_p2, "micro_r": micro_r2},
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved pipeline to: {pipe_path.resolve()}")
    print(f"Saved label binarizer to: {mlb_path.resolve()}")
    print(f"Saved per-class thresholds to: {th_path.resolve()}")
    print(f"Saved metrics to: {metrics_path.resolve()}")

if __name__ == "__main__":
    main()
