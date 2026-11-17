"""Notebook 07: is the GSE120584 serum ncRNA signal specific to AD, or shared by other dementias?

Tasks (positive class first):
    AD_vs_DLB      AD vs dementia with Lewy bodies
    AD_vs_VaD      AD vs vascular dementia
    AD_vs_other    AD vs (DLB + VaD)
    Dem_vs_NC      any dementia (AD + DLB + VaD) vs controls  (reference: "dementia detector")

Phase 1 (training split only):  python src/run_specificity.py
    nested CV for every task x feature set (A2 clinical+quality, C ncRNA+clinical+quality) x algorithm,
    and stability ranking of miRNAs for AD_vs_other.
Phase 2 (one-time test):        python src/run_specificity.py --final
    picks the model per task from CV results only, writes the locked config (timestamped), refits on the
    full training split, picks thresholds on training OOF predictions, then scores the held-out test split.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from biomarkers import stability_ranking
from modeling import (FEATURE_SETS, K_GRID, build_pipeline, classifiers, detection_rate, make_features,
                      nested_cv, pick_threshold)
from utils import ROOT, SEED, load_dataset

OUT = ROOT / "results" / "models" / "07_runs"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = ROOT / "results" / "models" / "07_locked_config.json"
TAB = ROOT / "results" / "tables"

TASKS = {
    "AD_vs_DLB":   (["AD"], ["DLB"]),
    "AD_vs_VaD":   (["AD"], ["VaD"]),
    "AD_vs_other": (["AD"], ["DLB", "VaD"]),
    "Dem_vs_NC":   (["AD", "DLB", "VaD"], ["NC"]),
}
SETS = ["A2_clinical+quality", "C_ncrna+clinical"]
ALGOS = ["LogReg", "XGBoost"]


def features(split):
    """All samples of a split with features; ncRNAs = the 817 miRNAs chosen in notebook 01 (training data)."""
    ncrna = pd.read_csv(TAB / "01_eda_filtered_mirnas.csv")["miRNA"].tolist()
    X, clin = load_dataset("GSE120584_serum", split=split)
    clin = clin.copy()
    clin["detect_rate"] = detection_rate(X)          # per-sample, uses all features of that sample only
    return make_features(X[ncrna], clin), clin, ncrna


def task_data(F, clin, task):
    pos, neg = TASKS[task]
    m = clin.diagnosis.isin(pos + neg)
    return F[m], clin.diagnosis[m].isin(pos).astype(int)


def cached(name, fn):
    p = OUT / f"{name}.joblib"
    if p.exists():
        return joblib.load(p)
    t = time.time()
    r = fn()
    joblib.dump(r, p)
    print(f"{name}: {time.time() - t:.0f}s", flush=True)
    return r


def phase1():
    F, clin, ncrna = features("train")
    for task in TASKS:
        Ft, y = task_data(F, clin, task)
        for fs in SETS:
            for algo in ALGOS:
                def run():
                    folds, oof, inner = nested_cv(Ft, y, ncrna, fs, algo)
                    return {"folds": folds, "oof": oof, "inner": inner,
                            "auc": roc_auc_score(y, oof["score"].astype(float))}
                cached(f"cv_{task}_{fs}_{algo}", run)
    Ft, y = task_data(F, clin, "AD_vs_other")
    cached("stability_AD_vs_other",
           lambda: dict(zip(["freq", "mean_rank"], stability_ranking(Ft, y, ncrna, n_splits=5, n_repeats=5, top_n=20))))
    print("phase 1 done", flush=True)


def final_fit(Ft, y, ncrna, fs, algo):
    """Tune on the full training data (5-fold GridSearchCV), return fitted model + training OOF scores."""
    cols = (ncrna if FEATURE_SETS[fs]["ncrna"] else []) + FEATURE_SETS[fs]["covariates"]
    clf, grid = classifiers(y)[algo]
    pipe = build_pipeline(clf, ncrna if FEATURE_SETS[fs]["ncrna"] else [], FEATURE_SETS[fs]["covariates"])
    if FEATURE_SETS[fs]["ncrna"]:
        grid = {**grid, "features__ncrna__select__k": K_GRID}
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv5, n_jobs=-1).fit(Ft[cols], y)
    oof = cross_val_predict(clone(gs.best_estimator_), Ft[cols], y, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
    return gs.best_estimator_, cols, oof, gs.best_params_


def phase2():
    if LOCK.exists():
        sys.exit(f"{LOCK} exists: the test set has already been used for notebook 07. Not re-running.")
    # choose per task from CV only: best algorithm for each feature set
    choice = {}
    for task in TASKS:
        choice[task] = {fs: max(ALGOS, key=lambda a: joblib.load(OUT / f"cv_{task}_{fs}_{a}.joblib")["auc"])
                        for fs in SETS}
    lock = {"locked_at": datetime.now().isoformat(timespec="seconds"),
            "rule": "per task and feature set, the algorithm with the higher nested-CV AUC on the training split; "
                    "refit with 5-fold GridSearchCV on the full training split; thresholds chosen on training OOF "
                    "predictions (rule: balanced accuracy)",
            "primary_endpoint": "test AUC for AD_vs_other, C_ncrna+clinical vs A2_clinical+quality",
            "choice": choice}
    json.dump(lock, open(LOCK, "w"), indent=2)
    print("locked:", json.dumps(choice), flush=True)

    F, clin, ncrna = features("train")
    Fte_all, clin_te, _ = features("test")
    out = {}
    for task in TASKS:
        Ft, y = task_data(F, clin, task)
        Fte, yte = task_data(Fte_all, clin_te, task)
        for fs in SETS:
            algo = choice[task][fs]
            model, cols, oof, params = final_fit(Ft, y, ncrna, fs, algo)
            thr = pick_threshold(y, oof, "balanced")
            out[(task, fs)] = {"algo": algo, "params": params, "train_oof_auc": roc_auc_score(y, oof),
                               "threshold": thr, "y_test": yte,
                               "test_score": pd.Series(model.predict_proba(Fte[cols])[:, 1], index=Fte.index),
                               "diagnosis_test": clin_te.diagnosis.loc[Fte.index]}
            print(f"{task} {fs} {algo}: train OOF AUC {out[(task, fs)]['train_oof_auc']:.3f}, "
                  f"test AUC {roc_auc_score(yte, out[(task, fs)]['test_score']):.3f}", flush=True)
    joblib.dump(out, ROOT / "results" / "models" / "07_final_test.joblib")
    print("phase 2 done", flush=True)


if __name__ == "__main__":
    phase2() if "--final" in sys.argv else phase1()
