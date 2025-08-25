# def/src/data/dataset.py
# Dataset y DataLoader para clasificación multietiqueta en literatura médica con PubMedBERT.

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.utils.text_cleaning import MedTextCleaner


class MedicalLitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def load_dataset(
    path="def/data/medical_lit.csv",
    encoder_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    test_size=0.2,
    random_state=42,
    max_length=512
):
    df = pd.read_csv(path, sep=';')
    df.dropna(subset=["title", "abstract", "group"], inplace=True)

    # Limpiar texto
    cleaner = MedTextCleaner()
    df["clean_text"] = cleaner.transform(df["title"] + " " + df["abstract"])

    # Convertir etiquetas (multi-label -> list of str)
    df["label_list"] = df["group"].apply(lambda x: x.split("|"))
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["label_list"])

    # División simple (estratificación multilabel requiere workaround)
    X_train, X_val, Y_train, Y_val = train_test_split(
        df["clean_text"], Y, test_size=test_size, random_state=random_state
    )

    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    train_ds = MedicalLitDataset(X_train.tolist(), Y_train, tokenizer, max_length)
    val_ds = MedicalLitDataset(X_val.tolist(), Y_val, tokenizer, max_length)
    return train_ds, val_ds, mlb


def build_dataloaders(train_ds, val_ds, batch_size=16, num_workers=2):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
