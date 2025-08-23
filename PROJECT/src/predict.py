from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

DEFAULT_PIPE = Path("models/tfidf_lr_med_pipeline.joblib")
DEFAULT_MLB  = Path("models/label_binarizer.joblib")
DEFAULT_THS  = Path("models/thresholds.json")

def load_artifacts(pipe_path: Path = DEFAULT_PIPE, mlb_path: Path = DEFAULT_MLB):
    pipeline = joblib.load(pipe_path)
    mlb = joblib.load(mlb_path)
    return pipeline, mlb

def _load_thresholds(thresholds_path: Path, classes: list[str]) -> np.ndarray:
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            ths_dict = json.load(f)
        return np.array([ths_dict.get(c, 0.0) for c in classes], dtype=float)
    return np.zeros(len(classes), dtype=float)

def predict_groups(title: str, abstract: str, threshold: float | None = None, top_k: int | None = None,
                   pipe_path: Path = DEFAULT_PIPE, mlb_path: Path = DEFAULT_MLB, thresholds_path: Path = DEFAULT_THS):
    pipeline, mlb = load_artifacts(pipe_path, mlb_path)
    text = f"{title or ''} {abstract or ''}"

    # scores
    try:
        scores = pipeline.decision_function([text])[0]
    except Exception:
        prob = getattr(pipeline, "predict_proba", None)
        if prob:
            scores = prob([text])[0]
        else:
            preds = pipeline.predict([text])[0]
            scores = preds.astype(float)

    classes = mlb.classes_.tolist()
    # Use per-class thresholds if available, unless user overrides with a scalar threshold
    if threshold is None:
        T = _load_thresholds(thresholds_path, classes)
    else:
        T = np.full(len(classes), float(threshold), dtype=float)

    mask = scores >= T

    if top_k is not None and top_k > 0:
        idx = np.argsort(scores)[::-1][:top_k]
        mask = np.array([i in idx for i in range(len(scores))])

    labels = [cls for cls, p in zip(classes, mask) if p]
    return {"labels": labels, "scores": scores.tolist(), "thresholds": T.tolist()}

def main(args=None):
    parser = argparse.ArgumentParser(description="Predict groups for a single title+abstract.")
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--abstract", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Scalar threshold. If omitted, uses per-class thresholds.json when available.")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--pipe", type=Path, default=DEFAULT_PIPE)
    parser.add_argument("--mlb", type=Path, default=DEFAULT_MLB)
    parser.add_argument("--ths", type=Path, default=DEFAULT_THS)
    ns = parser.parse_args(args=args)

    result = predict_groups(ns.title, ns.abstract, ns.threshold, ns.top_k, ns.pipe, ns.mlb, ns.ths)
    print(result)

if __name__ == "__main__":
    main()
