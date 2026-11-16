"""Improvement round for the AD vs NC model (results analysed in notebooks/02b_improve_GSE120584.ipynb).

Runs nested CV for feature set C (ncRNA + clinical) with larger hyperparameter grids, two
detection filters (>=80% and >=50%), and stores inner-fold scores so decision thresholds and
ensembles can be chosen on training data only. Each run is cached separately, so the script can
be stopped and resumed.

    .venv/bin/python src/run_improve.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid thread oversubscription (one thread per CV worker)

import time
import warnings

import joblib
import pandas as pd

from modeling import detection_rate, make_features, nested_cv
from utils import ROOT, load_dataset

OUT = ROOT / "results" / "models" / "02b_runs"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"

X, clin = load_dataset("GSE120584_serum", split="train", diagnoses=["AD", "NC"])
clin["detect_rate"] = detection_rate(X)
y = (clin["diagnosis"] == "AD").astype(int)

# Detection filters are computed on the whole training split (all diagnoses), label-free.
Xall, _ = load_dataset("GSE120584_serum", split="train")
det = Xall.sub(Xall.min(axis=1), axis=0).gt(1e-6).mean(axis=0)
FILTERS = {"det80": det[det >= 0.8].index.tolist(), "det50": det[det >= 0.5].index.tolist()}
pd.DataFrame({"miRNA": FILTERS["det50"], "detect_rate": det[FILTERS["det50"]].values}).to_csv(
    TAB / "02b_filtered_mirnas_det50.csv", index=False)

RUNS = [(f, a) for f in ["det80", "det50"] for a in ["LogReg", "GradBoost", "XGBoost"]]

for filt, algo in RUNS:
    path = OUT / f"C_{filt}_{algo}.joblib"
    if path.exists():
        print(f"cached   {filt} {algo}")
        continue
    cols = FILTERS[filt]
    F = make_features(X[cols], clin)
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds, oof, inner = nested_cv(F, y, cols, "C_ncrna+clinical", algo,
                                      big_grid=True, k_grid=[100, 300, "all"])
    joblib.dump({"filter": filt, "algo": algo, "n_features": len(cols),
                 "folds": folds, "oof": oof, "inner": inner}, path)
    print(f"done     {filt} {algo:10s} AUC {folds.AUC.mean():.3f}  acc@0.5 {folds.accuracy.mean():.3f}"
          f"  ({time.time() - t:.0f}s)", flush=True)
