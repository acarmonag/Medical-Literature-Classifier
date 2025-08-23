from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models_transformer" / "model"
DEFAULT_MLB_PATH  = PROJECT_ROOT / "models_transformer" / "mlb.joblib"
DEFAULT_THS_PATH  = PROJECT_ROOT / "models_transformer" / "thresholds.json"

def _device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

@st.cache_resource(show_spinner="Loading model/tokenizer/labels...")
def load_artifacts(model_dir: Path, mlb_path: Path, ths_path: Path):
    # Tokenizer + Model (HF)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    mdl.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)

    # Label binarizer
    mlb = joblib.load(mlb_path)
    classes = mlb.classes_.tolist()

    # Thresholds por clase (opcional)
    thresholds: Dict[str, float] = {}
    if ths_path.exists():
        try:
            thresholds = json.loads(ths_path.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Could not read thresholds.json: {e}")

    return tok, mdl, device, classes, thresholds

def predict_texts(tok, mdl, device, texts: List[str], max_len: int = 512) -> np.ndarray:
    """Devuelve matriz de probabilidades (n_docs, n_classes)."""
    # Tokenización en lotes para eficiencia
    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs

def apply_mode(probs: np.ndarray, classes: List[str], mode: str, thresholds: Dict[str, float], fixed_thr: float, top_k: int | None):
    """Aplica la decisión según modo: thresholds.json | fixed threshold | top-k."""
    if mode == "Per-class thresholds.json":
        T = np.array([thresholds.get(c, 0.5) for c in classes], dtype=float)
        preds = (probs >= T).astype(int)
        return preds, T
    elif mode == "Fixed threshold":
        T = np.full(probs.shape[1], float(fixed_thr), dtype=float)
        preds = (probs >= T).astype(int)
        return preds, T
    else:  # Top-k
        T = np.zeros(probs.shape[1], dtype=float)
        preds = np.zeros_like(probs, dtype=int)
        k = int(top_k or 1)
        for i, row in enumerate(probs):
            idx = np.argsort(row)[::-1][:k]
            preds[i, idx] = 1
        return preds, T

def main():
    st.set_page_config(page_title="MedLit Transformer Classifier", layout="wide")
    st.title("🤖 MedLit Transformer Classifier")
    st.caption(f"Device: **{_device_name()}** — CUDA: **{torch.cuda.is_available()}**")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_dir = Path(st.text_input("Model directory", str(DEFAULT_MODEL_DIR)))
        mlb_path  = Path(st.text_input("Label binarizer (.joblib)", str(DEFAULT_MLB_PATH)))
        ths_path  = Path(st.text_input("thresholds.json (optional)", str(DEFAULT_THS_PATH)))

        max_len = st.number_input("Max sequence length", value=512, min_value=64, max_value=512, step=64)
        mode = st.radio("Decision mode", ["Per-class thresholds.json", "Fixed threshold", "Top-k (single/multi)"])
        fixed_thr = st.number_input("Fixed threshold", value=0.5, step=0.05, disabled=(mode != "Fixed threshold"))
        top_k = st.number_input("Top-k", value=1, min_value=1, max_value=5, step=1, disabled=(mode != "Top-k (single/multi)"))

    tok, mdl, device, classes, thresholds = load_artifacts(model_dir, mlb_path, ths_path)
    st.success(f"Loaded {len(classes)} classes: {classes}")

    # ---------- Single prediction ----------
    st.subheader("🔎 Single prediction")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", placeholder="Proteomic profiling reveals distinct phases...")
    with col2:
        abstract = st.text_area("Abstract", height=160, placeholder="Paste abstract here...")

    if st.button("Predict", type="primary", use_container_width=True):
        text = f"{title or ''} [SEP] {abstract or ''}"
        probs = predict_texts(tok, mdl, device, [text], max_len=max_len)[0]
        preds, T = apply_mode(probs.reshape(1, -1), classes, mode, thresholds, fixed_thr, top_k)
        labels = [c for c, m in zip(classes, preds[0]) if m]
        st.success(f"Predicted labels: {labels or '—'}")

        df_scores = pd.DataFrame({"class": classes, "prob": probs})
        if mode == "Per-class thresholds.json":
            df_scores["threshold"] = [thresholds.get(c, 0.5) for c in classes]
        st.dataframe(df_scores.sort_values("prob", ascending=False).reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # ---------- Batch prediction ----------
    st.subheader("📦 Batch prediction (CSV)")
    up = st.file_uploader("Upload CSV with columns: title, abstract", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up, sep=None, engine="python", dtype=str)
        except Exception:
            up.seek(0)
            df = pd.read_csv(up, dtype=str)
        for col in ("title", "abstract"):
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                st.stop()
            df[col] = df[col].fillna("")

        texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
        probs_all = predict_texts(tok, mdl, device, texts, max_len=max_len)
        preds_all, T = apply_mode(probs_all, classes, mode, thresholds, fixed_thr, top_k)

        labels_list = [[c for c, m in zip(classes, row) if m] for row in preds_all]
        out = df.copy()
        out["pred_labels"] = [", ".join(L) for L in labels_list]
        # añade probabilidades por clase (opcional)
        for j, c in enumerate(classes):
            out[f"prob_{c}"] = probs_all[:, j]

        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions_transformer.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
