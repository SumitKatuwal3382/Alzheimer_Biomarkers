"""Sensitivity analyses for notebook 06: does the lncRNA AD signal survive removal of the global shift?

    A median   : per-sample median-centering
    B quantile : per-sample quantile normalisation
    C no_outl  : original data without the 13 correlation outliers flagged in notebook 06
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time

import joblib

from secondary import load_secondary, repeated_nested_cv
from utils import ROOT

OUT = ROOT / "results" / "models" / "06_runs"
OUTLIERS = ["GSM7913176", "GSM7913177", "GSM7913178", "GSM7913184", "GSM7913191", "GSM7913196", "GSM7913197",
            "GSM7913200", "GSM7913202", "GSM7913217", "GSM7913242", "GSM7913246", "GSM7913252"]
VARIANTS = {"median": dict(norm="median"), "quantile": dict(norm="quantile"), "no_outl": dict(drop=OUTLIERS)}

if __name__ == "__main__":
    for v, kw in VARIANTS.items():
        X, y, info = load_secondary("lncRNA", **kw)
        for algo in ["RandomForest", "LogReg"]:
            p = OUT / f"sens_{v}_{algo}.joblib"
            if p.exists():
                continue
            t = time.time()
            joblib.dump(repeated_nested_cv(X, y, algo, n_repeats=5), p)
            print(f"{v} {algo}: {time.time() - t:.0f}s  n={info['n']}", flush=True)
    print("all done", flush=True)
