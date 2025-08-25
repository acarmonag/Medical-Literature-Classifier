# Script CLI para evaluar el rendimiento del clasificador médico sobre un CSV anotado (con batching configurable y métricas por clase).
# Purpose: Ejecuta inferencia con MedicalPredictor, calcula métricas globales y por clase, y guarda un CSV con TP/FP/FN.

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# --- PATH FALLBACK: permite ejecutar desde la raíz del proyecto (def/) o rutas distintas ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.infer.predictor import MedicalPredictor
    from src.eval.eval_predictor import PredictionEvaluator
except ModuleNotFoundError:
    # Fallback adicional si alguien ejecuta desde dentro de src/
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from infer.predictor import MedicalPredictor  # type: ignore
    from eval.eval_predictor import PredictionEvaluator  # type: ignore


def _binarize_sets(series_of_sets, labels):
    """Convierte una Serie de sets({label,...}) en matriz binaria (N, L) según 'labels'."""
    return np.array([[1 if lbl in s else 0 for lbl in labels] for s in series_of_sets], dtype=np.int32)


def main(args):
    predictor = MedicalPredictor(
        model_path=args.model,
        thresholds_path=args.thresholds,
        labels_path=args.labels,
        encoder_name=args.encoder,
        att_dim=args.att_dim,
        default_max_length=args.max_len
    )

    evaluator = PredictionEvaluator(predictor)
    results, df = evaluator.evaluate_file(
        csv_path=args.csv,
        sep=args.sep,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    # Reconstruir y_true / y_pred en 4 clases canónicas (o las del predictor) desde df
    labels = getattr(predictor, "labels", ["cardiovascular", "hepatorenal", "neurological", "oncological"])
    y_true = _binarize_sets(df["true"], labels)
    y_pred = _binarize_sets(df["predicted"], labels)

    print("\n=== MÉTRICAS GLOBALES ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Métricas por clase
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    print("\n=== MÉTRICAS POR CLASE ===")
    for cls, p, r, f in zip(labels, prec_c, rec_c, f1_c):
        print(f"{cls:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    print("\n=== EJEMPLOS DE ERRORES ===")
    print(df[["title", "true", "predicted", "FN", "FP"]].head())

    if args.save:
        df.to_csv(args.save, index=False)
        print(f"\n🔽 Resultados detallados guardados en: {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al archivo CSV anotado")
    parser.add_argument("--sep", default=";", help="Separador del CSV (default=';')")
    parser.add_argument("--model", default="outputs/final_model.pt")
    parser.add_argument("--thresholds", default="outputs/thresholds.npy")
    parser.add_argument("--labels", default="outputs/labels.json")
    parser.add_argument("--encoder", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de lote para inferencia")
    parser.add_argument("--max_len", type=int, default=256, help="Longitud máx. de secuencia (CPU=256 recomendado)")
    parser.add_argument("--save", help="Ruta para guardar resultados con TP/FP/FN")
    parser.add_argument("--att_dim", type=int, default=None, help="Override manual del att_dim del checkpoint")
    args = parser.parse_args()
    main(args)
