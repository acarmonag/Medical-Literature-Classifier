# def/src/train/trainer.py
# Entrenador modular para clasificación multietiqueta con umbrales adaptativos y early stopping.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    # pos_weight = (#neg / #pos) por clase
    num_pos = labels.sum(axis=0)
    num_neg = labels.shape[0] - num_pos
    weight = num_neg / (num_pos + 1e-6)
    return torch.tensor(weight, dtype=torch.float)


def threshold_optimization(y_true, y_pred_logits, beta=1.5):
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1 = -1
        best_t = 0.5
        for t in thresholds:
            preds = (y_pred_logits[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0, beta=beta)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    return np.array(best_thresholds)


def evaluate(model, dataloader, device, thresholds=None):
    model.eval()
    y_true, y_pred_logits = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids, attention_mask).cpu().numpy()
            y_pred_logits.append(logits)
            y_true.append(labels)

    y_true = np.vstack(y_true)
    y_pred_logits = np.vstack(y_pred_logits)

    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)

    y_pred = (y_pred_logits >= thresholds).astype(int)
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics, y_true, y_pred_logits


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=2e-5,
        epochs=10,
        early_stop=3,
        use_pos_weight=True,
        beta=1.5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.early_stop = early_stop
        self.beta = beta

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        all_labels = np.vstack([b["labels"].numpy() for b in train_loader])
        if use_pos_weight:
            pos_weight = compute_pos_weights(all_labels).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def train(self):
        best_score = 0
        patience = self.early_stop
        best_thresholds = np.full(len(self.train_loader.dataset[0]["labels"]), 0.5)

        for epoch in range(self.epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            total_loss = 0

            for batch in loop:
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            val_metrics, y_true, y_logits = evaluate(self.model, self.val_loader, self.device)
            thresholds = threshold_optimization(y_true, y_logits, beta=self.beta)
            val_metrics_opt, _, _ = evaluate(self.model, self.val_loader, self.device, thresholds)

            macro_f1 = val_metrics_opt["macro_f1"]
            print(f"Epoch {epoch+1} val_macro_f1 (opt): {macro_f1:.4f}")

            if macro_f1 > best_score:
                best_score = macro_f1
                best_thresholds = thresholds
                patience = self.early_stop
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered.")
                    break

        return best_score, best_thresholds
