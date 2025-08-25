# Script principal para entrenar el clasificador multietiqueta médico con PubMedBERT + atención.

import os
import torch
from src.data.dataset import load_dataset, build_dataloaders
from src.model.labelwise_attention import LabelWiseAttentionClassifier
from src.train.trainer import Trainer
import argparse
import numpy as np
import json


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_ds, val_ds, mlb = load_dataset(
        path=args.data_path,
        encoder_model=args.encoder,
        test_size=args.test_size
    )
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=args.batch_size)

    model = LabelWiseAttentionClassifier(
        encoder_model=args.encoder,
        num_labels=len(mlb.classes_),
        att_dim=args.att_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        early_stop=args.early_stop,
        beta=args.beta
    )

    best_score, thresholds = trainer.train()

    print(f"Mejor macro-F1: {best_score:.4f}")
    print("Umbrales por clase:", dict(zip(mlb.classes_, thresholds.round(2).tolist())))

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    np.save(os.path.join(args.output_dir, "thresholds.npy"), thresholds)
    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(mlb.classes_.tolist(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="src/data/medical_lit.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--encoder", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--att_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
