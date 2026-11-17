"""Heavy computations for notebook 06 (lncRNA / circRNA, GSE248417). Results cached per run.

    python src/run_secondary.py          # skips runs that are already cached
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time

import joblib
import numpy as np
import pandas as pd

from biomarkers import stability_ranking
from secondary import ALGOS, load_secondary, repeated_nested_cv
from utils import ROOT, SEED

OUT = ROOT / "results" / "models" / "06_runs"
OUT.mkdir(parents=True, exist_ok=True)
N_PERM = 20


def cached(name, fn):
    p = OUT / f"{name}.joblib"
    if p.exists():
        return joblib.load(p)
    t = time.time()
    r = fn()
    joblib.dump(r, p)
    print(f"{name}: {time.time() - t:.0f}s", flush=True)
    return r


def feature_sets():
    Xl, yl, il = load_secondary("lncRNA")
    Xc, yc, ic = load_secondary("circRNA")
    common = Xc.index.intersection(Xl.index)
    Xb = pd.concat([Xl.loc[common], Xc.loc[common]], axis=1)
    return {"lncRNA": (Xl, yl, il), "circRNA": (Xc, yc, ic),
            "lncRNA+circRNA": (Xb, yl.loc[common], {"n": len(common), "features_used": Xb.shape[1]})}


if __name__ == "__main__":
    sets = feature_sets()
    # 1. repeated nested CV for every feature set x algorithm
    for fs, (X, y, _) in sets.items():
        for algo in ALGOS:
            cached(f"cv_{fs}_{algo}", lambda: repeated_nested_cv(X, y, algo, n_repeats=5))

    # 2. label-permutation null for the best algorithm per feature set (is the AUC above chance?)
    for fs, (X, y, _) in sets.items():
        best = max(ALGOS, key=lambda a: np.mean(joblib.load(OUT / f"cv_{fs}_{a}.joblib")["aucs"]))
        def perm():
            from joblib import Parallel, delayed
            rng = np.random.default_rng(SEED)
            perms = [pd.Series(rng.permutation(y.values), index=y.index) for _ in range(N_PERM)]
            null = Parallel(n_jobs=-1)(delayed(lambda yp: repeated_nested_cv(X, yp, best, n_repeats=1, n_jobs=1)["aucs"][0])(yp)
                                       for yp in perms)
            return {"algo": best, "null_aucs": list(null)}
        cached(f"perm_{fs}", perm)

    # 3. stability ranking (LASSO / RFE / RF / SHAP) for lncRNA and circRNA separately
    for fs in ["lncRNA", "circRNA"]:
        X, y, _ = sets[fs]
        cached(f"stability_{fs}", lambda: dict(zip(["freq", "mean_rank"],
                                                   stability_ranking(X, y, list(X.columns)))))
    print("all done", flush=True)
