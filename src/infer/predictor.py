# def/src/infer/predictor.py
# Predictor para inferencia con modelo multietiqueta entrenado (auto-infiere att_dim/num_labels del checkpoint).

from __future__ import annotations
import json
from typing import List, Iterable, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from src.model.labelwise_attention import LabelWiseAttentionClassifier
from src.utils import MedTextCleaner


def _infer_shapes_from_state(state_dict: dict) -> tuple[int, int]:
    """
    Devuelve (num_labels, att_dim) inferidos desde el checkpoint.
    - attention: (num_labels, att_dim)
    - project.bias: (att_dim,)
    """
    if "attention" in state_dict:
        num_labels, att_dim = state_dict["attention"].shape
        return int(num_labels), int(att_dim)
    if "project.bias" in state_dict:
        att_dim = state_dict["project.bias"].shape[0]
        # num_labels fallback (si no hay 'attention'): intenta 'output.weight' (H->1) no sirve; usa None
        return -1, int(att_dim)
    raise ValueError("No se pudo inferir att_dim/num_labels del checkpoint (faltan claves esperadas).")


class MedicalPredictor:
    def __init__(
        self,
        model_path: str,
        thresholds_path: str,
        labels_path: str,
        encoder_name: str,
        att_dim: Optional[int] = None,   # opcional: override manual
        dropout: float = 0.2,
        default_max_length: int = 512,
        dtype_half_on_cuda: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Etiquetas y umbrales
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)
        self.thresholds = np.load(thresholds_path)

        # Tokenizador y limpieza (permite paths locales o nombres HF)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.cleaner = MedTextCleaner()

        # Cargar checkpoint primero para inferir tamaños
        state = torch.load(model_path, map_location=self.device)
        if "state_dict" in state:  # por si lo guardaste envuelto
            state = state["state_dict"]

        ckpt_num_labels, ckpt_att_dim = _infer_shapes_from_state(state)
        # num_labels: por defecto usamos las etiquetas del JSON
        num_labels = len(self.labels) if ckpt_num_labels in (-1, len(self.labels)) else ckpt_num_labels

        # Validaciones y conciliación
        if num_labels != len(self.labels):
            # No rompemos: avisamos y recortamos/extendemos thresholds si fuera necesario
            print(f"[WARN] num_labels({num_labels}) != len(labels_json)({len(self.labels)}). Usando num_labels del checkpoint.")
        # Ajuste de thresholds si largo no coincide (seguridad)
        if len(self.thresholds) != len(self.labels):
            print(f"[WARN] thresholds.npy len={len(self.thresholds)} difiere de labels={len(self.labels)}; ajustando...")
            if len(self.thresholds) > len(self.labels):
                self.thresholds = self.thresholds[:len(self.labels)]
            else:
                self.thresholds = np.pad(self.thresholds, (0, len(self.labels)-len(self.thresholds)), constant_values=0.5)

        # Permitir override manual de att_dim desde CLI si lo pasas
        final_att_dim = int(att_dim) if att_dim is not None else int(ckpt_att_dim)

        # Instanciar modelo con tamaños correctos
        self.model = LabelWiseAttentionClassifier(
            encoder_model=encoder_name,
            num_labels=num_labels,
            att_dim=final_att_dim,
            dropout=dropout,
            freeze_encoder=False
        )
        # Cargar pesos
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Config
        self.max_length = int(default_max_length)
        self._use_fp16 = bool(dtype_half_on_cuda and self.device.type == "cuda")

    def _predict_chunk(self, texts: List[str], max_length: int) -> np.ndarray:
        cleaned = self.cleaner.transform(texts)
        enc = self.tokenizer(
            cleaned, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            if self._use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(input_ids, attention_mask)
            else:
                logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs

    def predict(self, title: str, abstract: str, max_length: Optional[int] = None) -> List[str]:
        max_len = int(max_length or self.max_length)
        return self.predict_batch([f"{title} {abstract}"], batch_size=1, max_length=max_len)[0]

    def predict_batch(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        max_length: Optional[int] = None,
        return_probs: bool = False,
    ):
        X = list(texts)
        if not X:
            return [] if not return_probs else np.zeros((0, len(self.labels)), dtype=float)
        max_len = int(max_length or self.max_length)

        all_probs = []
        for i in range(0, len(X), batch_size):
            chunk = X[i:i+batch_size]
            all_probs.append(self._predict_chunk(chunk, max_len))
        probs_full = np.vstack(all_probs)

        if return_probs:
            return probs_full
        preds_bin = (probs_full >= self.thresholds).astype(int)
        return [[self.labels[j] for j, bit in enumerate(row) if bit] for row in preds_bin]
