# Crea splits train/val/test estratificados para multilabel.
# - Usa iterstrat si está disponible
# - Fallback robusto con train_test_split y una clave estratificada simple
# - Guarda CSVs con separador ';'

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def stratified_split_indices(Y: np.ndarray, train_size: float, val_size: float, test_size: float, seed: int = 42):
    """
    Devuelve (train_idx, val_idx, test_idx) como arrays de índices (np.ndarray).
    Intentará usar MultilabelStratifiedShuffleSplit; si no, fallback con claves estratificadas.
    """
    N = Y.shape[0]
    idx = np.arange(N)

    # Intento 1: iterstrat
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(msss1.split(idx, Y))

        rel_val = val_size / (train_size + val_size)
        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_rel, va_rel = next(msss2.split(train_val_idx, Y[train_val_idx]))
        train_idx = train_val_idx[tr_rel]
        val_idx = train_val_idx[va_rel]
        return train_idx, val_idx, test_idx
    except Exception:
        # Fallback: estratificar por (n_labels, label_mayoritaria)
        label_counts = Y.sum(axis=1)
        major = Y.argmax(axis=1)
        key = (label_counts * 10 + major).astype(int)

        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=key
        )
        rel_val = val_size / (train_size + val_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=rel_val, random_state=seed, stratify=key[train_val_idx]
        )
        return train_idx, val_idx, test_idx


def main(args):
    df = pd.read_csv(args.csv, sep=";")
    df["label_list"] = df["group"].astype(str).str.split("|")

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["label_list"])

    train_idx, val_idx, test_idx = stratified_split_indices(
        Y,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        seed=args.seed
    )

    df.iloc[train_idx].to_csv(args.out_train, index=False, sep=";")
    df.iloc[val_idx].to_csv(args.out_val, index=False, sep=";")
    df.iloc[test_idx].to_csv(args.out_test, index=False, sep=";")
    print(f"Saved:\n  {args.out_train} ({len(train_idx)})\n  {args.out_val} ({len(val_idx)})\n  {args.out_test} ({len(test_idx)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="def/data/medical_lit.csv")
    p.add_argument("--train", type=float, default=0.72)
    p.add_argument("--val", type=float, default=0.08)
    p.add_argument("--test", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_train", default="def/data/train.csv")
    p.add_argument("--out_val", default="def/data/val.csv")
    p.add_argument("--out_test", default="def/data/test.csv")
    args = p.parse_args()
    main(args)
