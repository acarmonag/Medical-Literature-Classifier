# Medical Literature Classifier

Multi-label classifier for biomedical abstracts across 4 clinical domains: **cardiology**, **hepato-renal**, **neurology**, and **oncology**.

Built on PubMedBERT with a custom Label-Wise Attention head and per-class threshold optimization using F1.5 sweep (recall-biased). Includes a Streamlit inference UI, CLI evaluation scripts, and reproducible training runs with versioned output artifacts.

---

## Architecture

```
Title + Abstract
      │
      ▼
PubMedBERT encoder (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
      │  hidden states (B, T, H)
      ▼
Linear projection → Tanh  (B, T, att_dim)
      │
      ▼
Label-Wise Attention        ← one attention vector per label
      │  attended context (B, L, H)
      ▼
Per-label linear classifier
      │  logits (B, L)
      ▼
BCEWithLogitsLoss + F1.5 threshold sweep
      │
      ▼
Multi-label predictions  [cardio | hepatorenal | neuro | onco]
```

**Key design decisions:**
- Label-Wise Attention allows each label to attend independently to different parts of the input — standard CLS pooling loses this signal in multi-label scenarios
- F1.5 threshold sweep over the validation set prioritizes recall over precision, appropriate for clinical literature triage
- Per-run artifact versioning (`outputs/b3/`) enables reproducible comparison across training configurations

---

## Stack

`Python 3.10` `PyTorch` `HuggingFace Transformers` `PubMedBERT` `Streamlit` `pandas` `numpy`

---

## Project structure

```
train.py                        # Training entry point
run_eval.py                     # Evaluation against annotated CSV
app.py                          # Streamlit inference UI
src/
  model/
    labelwise_attention.py      # Custom LabelWiseAttentionClassifier
    local_pubmedbert/           # Local encoder weights (offline support)
  data/
    dataset.py                  # Dataset + tokenization
    split_multilabel.py         # Stratified multi-label split
    medical_lit.csv             # Full dataset
    train.csv / val.csv / test.csv
  train/
    trainer.py                  # Training loop + early stopping
    calibration.py              # Post-hoc calibration
  eval/
    eval_predictor.py           # Batch evaluation
    threshold_tuning.py         # F1.5 sweep per class
  infer/
    predictor.py                # Inference wrapper
  utils/
    text_cleaning.py            # Text normalization
outputs/
  b3/                           # Best run artifacts
    best_model.pt
    final_model.pt
    thresholds.npy              # Per-class optimized thresholds
    labels.json
    eval_b3.csv                 # Holdout evaluation results
docs/
  block_diagram.pdf
  informe_final.pdf
```

---

## Quickstart

**Clone and pull LFS weights:**
```bash
git clone https://github.com/acarmonag/Medical-Literature-Classifier
cd Medical-Literature-Classifier
git lfs install && git lfs pull
```

**Install dependencies:**
```bash
# Unix/macOS
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Windows
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

**Run Streamlit UI:**
```bash
streamlit run app.py
```

Upload a CSV with `title` and `abstract` columns, or enter a single article manually. Predictions run automatically on CSV upload — no need to press Predict.

---

## Training

```bash
python train.py \
  --data_path src/data/train.csv \
  --epochs 6 \
  --early_stop 2 \
  --batch_size 16 \
  --lr 1e-5 \
  --att_dim 192 \
  --dropout 0.2 \
  --output_dir outputs/b3
```

Each run produces `best_model.pt`, `final_model.pt`, `thresholds.npy`, and `labels.json` under the specified output directory. Use distinct folder names per run (e.g. `b3-att192`) to track config changes.

---

## Evaluation

```bash
# Evaluate against annotated CSV
python run_eval.py \
  --csv src/data/medical_lit.csv \
  --batch_size 16 \
  --max_len 256 \
  --encoder src/model/local_pubmedbert \
  --model outputs/b3/final_model.pt \
  --thresholds outputs/b3/thresholds.npy \
  --labels outputs/b3/labels.json \
  --save eval_full.csv

# Holdout evaluation
python -m src.eval.eval_holdout \
  --csv src/data/test.csv \
  --model outputs/b3/final_model.pt \
  --thresholds outputs/b3/thresholds.npy \
  --labels outputs/b3/labels.json \
  --encoder src/model/local_pubmedbert \
  --batch_size 16 \
  --max_len 256
```

---

## Requirements

- Python 3.9+ (tested on 3.10)
- PyTorch, Transformers, Streamlit, pandas, numpy — see `requirements.txt`
- `iterstrat` optional for stratified multi-label splits
- Offline encoder available at `src/model/local_pubmedbert` — no HuggingFace connection required

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `att_dim` mismatch on load | Predictor auto-detects `att_dim` from checkpoint and loads with `strict=False` if needed |
| OOM on CPU | Reduce `--batch_size 8` and `--max_len 192` |
| Import errors | Use `python -m src.eval.eval_holdout` instead of direct script execution |
| UI crash on startup | Delete `outputs/` and replace with `output.zip` contents |
