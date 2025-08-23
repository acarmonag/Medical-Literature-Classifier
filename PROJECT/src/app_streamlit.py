from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-import custom transformer so joblib can resolve it during unpickle
from src.utils.text_cleaning import MedTextCleaner  # noqa: F401

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from typing import Dict, List, Tuple

DEFAULT_PIPE = PROJECT_ROOT / "models" / "tfidf_lr_med_pipeline.joblib"
DEFAULT_MLB  = PROJECT_ROOT / "models" / "label_binarizer.joblib"
DEFAULT_THS  = PROJECT_ROOT / "models" / "thresholds.json"

@st.cache_resource(show_spinner="Loading artifacts...")
def load_artifacts(pipe_path: Path, mlb_path: Path, ths_path: Path):
    try:
        pipe = joblib.load(pipe_path)
        mlb  = joblib.load(mlb_path)
    except Exception as e:
        st.error(
            "Failed to load artifacts. The unpickler must import "
            "`src.utils.text_cleaning.MedTextCleaner`.\n\n"
            f"sys.path[0] = `{sys.path[0]}`\n\nDetails: {type(e).__name__}: {e}"
        )
        st.stop()

    thresholds = {}
    if ths_path.exists():
        try:
            thresholds = json.loads(ths_path.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Could not read thresholds.json: {e}")
    return pipe, mlb, thresholds

def apply_thresholds(scores: np.ndarray, classes: list[str], thresholds: dict[str, float]) -> np.ndarray:
    T = np.array([thresholds.get(c, 0.0) for c in classes], dtype=float)
    return (scores >= T).astype(int)

def score_text(pipe, text: str) -> np.ndarray:
    decf = getattr(pipe, "decision_function", None)
    if decf is not None:
        return decf([text])[0]
    prob = getattr(pipe, "predict_proba", None)
    if prob is not None:
        return prob([text])[0]
    pred = pipe.predict([text])[0]
    return pred.astype(float)

# ---------- Explainability helpers ----------
def _get_linear_parts(pipe):
    clean = pipe.named_steps.get("clean")
    vect  = pipe.named_steps.get("tfidf")
    clf   = pipe.named_steps.get("clf")
    if vect is None or clf is None:
        return None, None, None
    return clean, vect, clf

def explain_doc(pipe, text: str, classes: List[str], top_terms: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Returns top positive-contributing TF-IDF terms present in the document for each class.
    """
    clean, vect, clf = _get_linear_parts(pipe)
    if vect is None or clf is None:
        return {}

    # Clean then vectorize the single document to know which terms it has
    if clean is not None:
        text_clean = clean.transform([text])[0]
    else:
        text_clean = text
    X_doc = vect.transform([text_clean])
    feature_names = vect.get_feature_names_out()
    X_arr = X_doc.toarray()[0]

    # OneVsRestClassifier of LogisticRegression
    ests = getattr(clf, "estimators_", None)
    if ests is None:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for j, cls in enumerate(classes):
        lr = ests[j]
        coef = lr.coef_.ravel()  # (n_features,)
        # Contribution of each term in the doc = tfidf_value * coef
        contrib = X_arr * coef
        # Keep only terms present (tfidf > 0)
        idx_present = np.where(X_arr > 0)[0]
        if idx_present.size == 0:
            out[cls] = pd.DataFrame(columns=["term", "tfidf", "weight", "contribution"])
            continue
        terms = feature_names[idx_present]
        tfidf_vals = X_arr[idx_present]
        weights = coef[idx_present]
        contrib_vals = contrib[idx_present]
        df = pd.DataFrame({
            "term": terms,
            "tfidf": tfidf_vals,
            "weight": weights,
            "contribution": contrib_vals
        }).sort_values("contribution", ascending=False).head(top_terms).reset_index(drop=True)
        out[cls] = df
    return out
# --------------------------------------------

def main():
    st.set_page_config(page_title="MedLit Classifier", layout="wide")
    st.title("📚 Medical Literature Classifier")

    with st.sidebar:
        st.header("⚙️ Settings")
        pipe_path = st.text_input("Pipeline .joblib", str(DEFAULT_PIPE))
        mlb_path  = st.text_input("Label binarizer .joblib", str(DEFAULT_MLB))
        ths_path  = st.text_input("thresholds.json (optional)", str(DEFAULT_THS))

        mode = st.radio("Decision mode", ["Per-class thresholds.json", "Fixed threshold", "Top-k (single label)"])
        fixed_thr = st.number_input("Fixed threshold", value=0.0, step=0.05, disabled=(mode != "Fixed threshold"))
        top_k = st.number_input("Top-k", value=1, min_value=1, max_value=5, step=1, disabled=(mode != "Top-k (single label)"))
        top_terms = st.number_input("Top terms per class (Explain)", value=10, min_value=3, max_value=30)

    pipe, mlb, thresholds = load_artifacts(Path(pipe_path), Path(mlb_path), Path(ths_path))
    classes = mlb.classes_.tolist()

    st.subheader("🔎 Single prediction")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", placeholder="Effects of IL-6 in COVID-19")
    with col2:
        abstract = st.text_area("Abstract", height=160, placeholder="Randomized trial shows...")

    if st.button("Predict", use_container_width=True):
        text = f"{title or ''} {abstract or ''}"
        scores = score_text(pipe, text)

        if mode == "Per-class thresholds.json":
            mask = apply_thresholds(scores, classes, thresholds)
        elif mode == "Fixed threshold":
            mask = (scores >= float(fixed_thr)).astype(int)
        else:
            idx = np.argsort(scores)[::-1][:int(top_k)]
            mask = np.zeros_like(scores, dtype=int)
            mask[idx] = 1

        labels = [c for c, m in zip(classes, mask) if m]
        st.success(f"Predicted labels: {labels or '—'}")

        # Scores table
        df_scores = pd.DataFrame({"class": classes, "score": scores})
        if mode == "Per-class thresholds.json":
            df_scores["threshold"] = [thresholds.get(c, 0.0) for c in classes]
        st.dataframe(df_scores.sort_values("score", ascending=False).reset_index(drop=True), use_container_width=True)

        # Explainability
        with st.expander("🔍 Explain (top TF-IDF contributions per class)"):
            explain_all = explain_doc(pipe, text, classes, top_terms=top_terms)
            if not explain_all:
                st.info("Explainability is available for linear OneVsRest(LogisticRegression) with TF-IDF.")
            else:
                # Show only predicted classes first, then others
                order = labels + [c for c in classes if c not in labels]
                for cls in order:
                    st.markdown(f"**{cls}**")
                    dfc = explain_all.get(cls)
                    if dfc is None or dfc.empty:
                        st.write("(no terms present)")
                    else:
                        st.dataframe(dfc, use_container_width=True)

    st.markdown("---")
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

        texts = (df["title"] + " " + df["abstract"]).tolist()
        decf = getattr(pipe, "decision_function", None)
        if decf is not None:
            scores_all = decf(texts)
        else:
            prob = getattr(pipe, "predict_proba", None)
            scores_all = prob(texts) if prob is not None else pipe.predict(texts).astype(float)

        if mode == "Per-class thresholds.json":
            preds = apply_thresholds(scores_all, classes, thresholds)
        elif mode == "Fixed threshold":
            preds = (scores_all >= float(fixed_thr)).astype(int)
        else:
            preds = np.zeros_like(scores_all, dtype=int)
            k = int(top_k)
            for i, row in enumerate(scores_all):
                idx = np.argsort(row)[::-1][:k]
                preds[i, idx] = 1

        labels_list = [[c for c, m in zip(classes, row) if m] for row in preds]
        df_out = df.copy()
        df_out["pred_labels"] = [", ".join(L) for L in labels_list]
        st.dataframe(df_out.head(20), use_container_width=True)
        st.download_button("Download predictions CSV", data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv", use_container_width=True)

if __name__ == "__main__":
    main()
