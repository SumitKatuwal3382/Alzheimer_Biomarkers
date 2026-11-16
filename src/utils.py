"""Shared paths, seed and data loaders for notebooks and scripts.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("../src")
    from utils import load_dataset, SEED
    X, clin = load_dataset("GSE120584_serum", split="train")
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
REFERENCE = ROOT / "data" / "reference"
FIGURES = ROOT / "results" / "figures"


def list_datasets():
    """Summary table of all processed datasets."""
    return pd.read_csv(PROCESSED / "dataset_summary.csv")


def load_dataset(name, split="all", diagnoses=None):
    """Load one processed dataset.

    name:      e.g. "GSE120584_serum", "GSE215789_serum" (see list_datasets()).
    split:     "all", "train" or "test". Only GSE120584_serum has a held-out split;
               use split="train" for all EDA / model development on it.
    diagnoses: optional list to keep, e.g. ["AD", "NC"].

    Returns (X, clin): X = samples x features, clin = clinical table (same index).
    """
    X = pd.read_csv(PROCESSED / f"{name}_expr.csv", index_col=0)
    clin = pd.read_csv(PROCESSED / f"{name}_clinical.csv", index_col=0)

    if split != "all":
        split_file = SPLITS / f"{name}_split.csv"
        if not split_file.exists():
            raise ValueError(f"No train/test split defined for {name}; use split='all'.")
        s = pd.read_csv(split_file, index_col=0)["split"]
        clin = clin[clin.index.map(s) == split]

    if diagnoses is not None:
        clin = clin[clin["diagnosis"].isin(diagnoses)]

    return X.loc[clin.index], clin


def load_known_ncrnas():
    """Literature reference list of AD-associated ncRNAs (data/reference/known_ad_ncrnas.csv)."""
    return pd.read_csv(REFERENCE / "known_ad_ncrnas.csv")


def set_seed(seed=SEED):
    np.random.seed(seed)
    return seed
