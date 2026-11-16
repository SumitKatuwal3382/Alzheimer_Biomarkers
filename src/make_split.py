"""Create the fixed, held-out test split for the training dataset (GSE120584 serum).

Run ONCE. The resulting file (data/splits/GSE120584_serum_split.csv) is committed to git
and must not be regenerated after analysis starts. The test samples are not to be used
for EDA, feature selection, tuning or model choice - only for the final evaluation.
"""
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import PROCESSED, SEED, SPLITS

NAME = "GSE120584_serum"
TEST_SIZE = 0.20

out = SPLITS / f"{NAME}_split.csv"
if out.exists() and "--force" not in sys.argv:
    sys.exit(f"{out} already exists - refusing to overwrite (the test set must stay fixed).")

clin = pd.read_csv(PROCESSED / f"{NAME}_clinical.csv", index_col=0)
# Stratify on diagnosis x sex so both are balanced between train and test.
strata = clin["diagnosis"] + "_" + clin["sex"]
train_idx, test_idx = train_test_split(clin.index, test_size=TEST_SIZE, stratify=strata, random_state=SEED)

split = pd.Series("train", index=clin.index, name="split")
split[test_idx] = "test"
SPLITS.mkdir(parents=True, exist_ok=True)
split.to_frame().to_csv(out)

summary = pd.crosstab(clin["diagnosis"], split)
summary["test_%"] = (100 * summary["test"] / summary.sum(axis=1)).round(1)
print(summary.to_string())
print("\nMean age:", clin.groupby(split)["age"].mean().round(1).to_dict())
print(f"\nSaved {out}")
