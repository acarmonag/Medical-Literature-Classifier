# def/src/eval/eval_predictor.py
# Evalúa el desempeño de MedicalPredictor sobre un dataset etiquetado (con batching).

from __future__ import annotations
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from src.infer.predictor import MedicalPredictor


class PredictionEvaluator:
    """
    Ejecuta la predicción sobre un CSV anotado y calcula métricas macro,
    además de listar TP/FP/FN por documento.
    """
    def __init__(self, predictor: MedicalPredictor):
        self.predictor = predictor

    def evaluate_file(self, csv_path: str, sep: str = ";", batch_size: int = 16, max_len: int | None = None):
        df = pd.read_csv(csv_path, sep=sep)
        df["clean_text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
        true_labels = df["group"].astype(str).str.split("|")

        preds = self.predictor.predict_batch(
            df["clean_text"].tolist(),
            batch_size=batch_size,
            max_length=max_len,
            return_probs=False
        )
        df["predicted"] = [set(p) for p in preds]
        df["true"] = [set(t) for t in true_labels]

        labels = self.predictor.labels
        def binarize(sets):
            return [[1 if l in s else 0 for l in labels] for s in sets]

        y_true = binarize(df["true"])
        y_pred = binarize(df["predicted"])

        results = {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "exact_match_ratio": float((df["true"] == df["predicted"]).mean())
        }

        df["FP"] = df.apply(lambda row: sorted(list(row["predicted"] - row["true"])), axis=1)
        df["FN"] = df.apply(lambda row: sorted(list(row["true"] - row["predicted"])), axis=1)
        df["TP"] = df.apply(lambda row: sorted(list(row["true"] & row["predicted"])), axis=1)

        return results, df