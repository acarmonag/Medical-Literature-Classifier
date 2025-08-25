# app.py
# Streamlit UI para inferencia y evaluación multilabel (con métricas y matriz de confusión por clase).
# Propósito: predicción simple y por CSV; si el CSV incluye 'group', calcula métricas globales y muestra una CM interactiva.

from __future__ import annotations
import os
import sys
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix
)

# --- Robust import: permite ejecutar app.py desde la raíz del proyecto ('def/') o rutas distintas ---
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.infer.predictor import MedicalPredictor
except ModuleNotFoundError:
    from infer.predictor import MedicalPredictor  # type: ignore


# -----------------------------
# Utilidades de métrica y binarización
# -----------------------------

def binarize_sets(series_of_sets: pd.Series, labels: List[str]) -> np.ndarray:
    """Convierte Serie de sets({'cardio',...}) -> matriz binaria (N, L) en el orden de 'labels'."""
    return np.array([[1 if lbl in s else 0 for lbl in labels] for s in series_of_sets], dtype=int)

def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas globales multilabel."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match_ratio": float((y_true == y_pred).all(axis=1).mean()),
    }

def compute_per_class(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    """Tabla por clase con P/R/F1 y soporte."""
    P, R, F1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    df = pd.DataFrame({
        "label": labels,
        "precision": np.round(P, 4),
        "recall": np.round(R, 4),
        "f1": np.round(F1, 4),
        "support": support.astype(int),
    })
    return df.sort_values("f1", ascending=False).reset_index(drop=True)

def plot_confusion_for_label(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], label_name: str):
    """Dibuja matriz de confusión 2x2 para una clase específica."""
    idx = labels.index(label_name)
    cm = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')  # sin estilo/colores específicos
    ax.set_title(f"Matriz de confusión — {label_name}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    # Anotar valores
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)


# -----------------------------
# Umbrales y helper de predictor
# -----------------------------

def load_thresholds(thresholds_path: str, expected_len: int) -> np.ndarray:
    """Carga umbrales desde .npy y corrige longitud si no coincide con labels."""
    try:
        th = np.load(thresholds_path)
        if len(th) != expected_len:
            st.warning(f"thresholds.npy len={len(th)} != #labels={expected_len}. Ajustando.")
            if len(th) > expected_len:
                th = th[:expected_len]
            else:
                th = np.pad(th, (0, expected_len - len(th)), constant_values=0.5)
        return th.astype(float)
    except Exception as e:
        st.error(f"No se pudo cargar thresholds en {thresholds_path}: {e}")
        return np.full(expected_len, 0.5, dtype=float)

def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binariza probas con umbrales por clase."""
    thr = thresholds.reshape(1, -1)
    return (probs >= thr).astype(int)

def format_labels(bits_row: np.ndarray, labels: List[str]) -> List[str]:
    """Convierte bits a lista de etiquetas activas."""
    return [labels[j] for j, b in enumerate(bits_row) if b]

@st.cache_resource(show_spinner=True)
def get_predictor(model_path: str, thresholds_path: str, labels_path: str, encoder: str, att_dim: int | None, max_len: int) -> Tuple[MedicalPredictor, List[str], np.ndarray]:
    """Crea y cachea el predictor con artefactos configurables; devuelve predictor, labels y thresholds."""
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception as e:
        st.error(f"No se pudo leer labels.json en {labels_path}: {e}")
        labels = ["cardiovascular", "hepatorenal", "neurological", "oncological"]

    thresholds = load_thresholds(thresholds_path, expected_len=len(labels))
    predictor = MedicalPredictor(
        model_path=model_path,
        thresholds_path=thresholds_path,
        labels_path=labels_path,
        encoder_name=encoder,
        att_dim=att_dim,
        default_max_length=max_len
    )
    return predictor, labels, thresholds


# -----------------------------
# Layout principal
# -----------------------------

st.set_page_config(page_title="Medical Multi-Label Classifier", layout="wide")
st.title("🧬 Clasificación Multietiqueta de Literatura Médica")

with st.sidebar:
    st.header("⚙️ Configuración")

    default_model = "outputs/b3/final_model.pt"
    default_thr = "outputs/b3/thresholds.npy"
    default_labels = "outputs/b3/labels.json"
    default_encoder = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  # o "src/model/local_pubmedbert"

    model_path = st.text_input("Ruta del modelo (.pt)", value=default_model)
    thresholds_path = st.text_input("Ruta de thresholds (.npy)", value=default_thr)
    labels_path = st.text_input("Ruta de labels (.json)", value=default_labels)
    encoder_name = st.text_input("Encoder (HF o ruta local)", value=default_encoder)

    colA, colB = st.columns(2)
    with colA:
        max_len = st.number_input("max_length", min_value=64, max_value=512, value=256, step=32)
    with colB:
        batch_size = st.number_input("batch_size", min_value=1, max_value=128, value=16, step=1)

    att_dim_override = st.number_input("att_dim (opcional)", min_value=0, max_value=2048, value=0, step=1,
                                       help="Déjalo en 0 para auto-detec. del checkpoint.")
    att_dim = int(att_dim_override) if att_dim_override > 0 else None

    st.markdown("---")
    st.subheader("🎚 Umbrales")
    use_custom_thr = st.checkbox("Usar umbrales personalizados en la sesión", value=False)
    custom_thr_vals = None

predictor, labels, file_thresholds = get_predictor(
    model_path=model_path,
    thresholds_path=thresholds_path,
    labels_path=labels_path,
    encoder=encoder_name,
    att_dim=att_dim,
    max_len=max_len
)

if use_custom_thr:
    st.sidebar.caption("Ajusta umbrales (0.01–0.99)")
    custom_thr_vals = []
    for i, lab in enumerate(labels):
        v = float(st.sidebar.slider(f"{lab}", min_value=0.01, max_value=0.99, value=float(file_thresholds[i]), step=0.01))
        custom_thr_vals.append(v)
    session_thresholds = np.array(custom_thr_vals, dtype=float)
else:
    session_thresholds = file_thresholds

st.sidebar.markdown("---")
st.sidebar.write(f"**Dispositivo**: {'cuda' if predictor.device.type == 'cuda' else 'cpu'}")
st.sidebar.write(f"**Etiquetas**: {', '.join(labels)}")

tab_single, tab_batch = st.tabs(["🔎 Predicción simple", "📦 Predicción por CSV + Métricas"])

# -----------------------------
# Pestaña: Predicción simple
# -----------------------------
with tab_single:
    st.subheader("Entrada de texto")
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Title", value="")
    with c2:
        abstract = st.text_area("Abstract", value="", height=180)

    go = st.button("🔮 Predecir", use_container_width=True)

    if go:
        if not title and not abstract:
            st.warning("Ingresa al menos el título o el abstract.")
        else:
            try:
                probs = predictor.predict_batch([f"{title} {abstract}"], batch_size=1, max_length=max_len, return_probs=True)[0]
                bits = apply_thresholds(probs.reshape(1, -1), session_thresholds)[0]
                pred_labels = format_labels(bits, labels)

                c3, c4 = st.columns([1, 1])
                with c3:
                    st.write("**Etiquetas activas**")
                    if pred_labels:
                        st.success(", ".join(pred_labels))
                    else:
                        st.info("Ninguna etiqueta superó su umbral.")

                with c4:
                    st.write("**Umbrales en uso**")
                    st.json({lab: float(t) for lab, t in zip(labels, session_thresholds)})

                st.write("**Probabilidades por clase**")
                dfp = pd.DataFrame({"label": labels, "prob": np.round(probs, 4), "active": bits.astype(bool)})
                st.dataframe(dfp.sort_values("prob", ascending=False), use_container_width=True)

            except Exception as e:
                st.error(f"Error en predicción: {e}")

# -----------------------------
# Pestaña: Predicción por CSV + Métricas
# -----------------------------
with tab_batch:
    st.subheader("Sube un CSV con columnas: title;abstract;group (group opcional, sep=';')")
    upl = st.file_uploader("Selecciona el archivo (.csv)", type=["csv"])

    if upl is not None:
        try:
            df_in = pd.read_csv(upl, sep=";")
            if not {"title", "abstract"}.issubset(df_in.columns):
                st.error("El CSV debe contener las columnas: 'title' y 'abstract'.")
            else:
                texts = (df_in["title"].fillna("") + " " + df_in["abstract"].fillna("")).tolist()
                probs = predictor.predict_batch(texts, batch_size=int(batch_size), max_length=max_len, return_probs=True)
                bits = apply_thresholds(probs, session_thresholds)
                preds = [format_labels(bits[i], labels) for i in range(bits.shape[0])]

                df_out = df_in.copy()
                df_out["predicted"] = [", ".join(p) for p in preds]
                for j, lab in enumerate(labels):
                    df_out[f"prob__{lab}"] = np.round(probs[:, j], 4)

                st.success(f"Listo. Filas procesadas: {len(df_out)}")
                st.dataframe(df_out.head(20), use_container_width=True)

                # Si trae ground-truth, calculamos métricas
                if "group" in df_in.columns:
                    st.markdown("### 📈 Métricas (si 'group' está en el CSV)")
                    y_true_sets = df_in["group"].astype(str).str.split("|").apply(set)
                    y_pred_sets = [set(p) for p in preds]
                    y_true = binarize_sets(y_true_sets, labels)
                    y_pred = binarize_sets(pd.Series(y_pred_sets), labels)

                    # Globales
                    g = compute_global_metrics(y_true, y_pred)
                    st.json({k: round(v, 4) for k, v in g.items()})

                    # Por clase
                    st.markdown("#### Por clase")
                    df_cls = compute_per_class(y_true, y_pred, labels)
                    st.dataframe(df_cls, use_container_width=True)

                    # Matriz de confusión interactiva (per-class)
                    st.markdown("#### Matriz de confusión por clase")
                    selected = st.selectbox("Selecciona etiqueta", labels, index=0)
                    plot_confusion_for_label(y_true, y_pred, labels, selected)

                # Descargar resultados
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Descargar resultados CSV",
                    data=csv_bytes,
                    file_name="predicciones.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"No se pudo procesar el CSV: {e}")

st.markdown("---")
st.caption("Modelo: PubMedBERT + Label-Wise Attention · Artefactos por defecto: outputs/b3/*. Si el CSV trae 'group', se calculan métricas y matriz de confusión por clase.")
