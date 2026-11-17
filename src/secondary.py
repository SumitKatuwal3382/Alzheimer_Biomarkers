"""lncRNA / circRNA analysis on GSE248417 (notebook 06).

GSE248417: whole-blood total RNA-seq, 49 AD vs 49 "matched" controls. No age, sex or APOE4 is
available, so no covariates can be modelled; results are ncRNA-only and exploratory.

Reuses the classifiers and pipeline from modeling.py (covariates = none). Because n is small,
nested CV is repeated with different fold assignments and the AUC is averaged over repeats.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modeling import build_pipeline, classifiers
from utils import SEED, load_dataset

K_GRID_SEC = [20, 100, 500]          # ncRNAs kept by univariate selection inside each fold
ALGOS = ["LogReg", "RandomForest", "SVM", "XGBoost"]


def gene_of(feature):
    """'ENST000...|MALAT1-012' -> 'MALAT1'; circRNA ids are returned unchanged."""
    if "|" not in feature:
        return feature
    name = feature.split("|", 1)[1]
    return name.rsplit("-", 1)[0] if "-" in name else name


def normalize(X, norm):
    """Per-sample, label-free normalisation of log2 data (sensitivity analyses).
    'median'  : subtract each sample's median  -> removes global level shifts
    'quantile': give every sample the same distribution (mean of sorted values)"""
    if norm is None:
        return X
    if norm == "median":
        return X.sub(X.median(axis=1), axis=0)
    if norm == "quantile":
        ref = np.sort(X.values, axis=1).mean(axis=0)
        ranks = X.rank(axis=1, method="first").astype(int).values - 1
        return pd.DataFrame(ref[ranks], index=X.index, columns=X.columns)
    raise ValueError(norm)


def load_secondary(kind, min_detect=0.5, top_var=5000, norm=None, drop=None):
    """Load lncRNA or circRNA, apply label-free filters and log2(x+1).

    - circRNA: drop failed samples (total TPM < 1e5, i.e. essentially empty)
    - keep features with value > 1 in >= `min_detect` of samples
    - optional per-sample normalisation (`norm`) and sample exclusion (`drop`), for sensitivity analyses
    - keep the `top_var` most variable features (on the log scale)
    These filters never look at the diagnosis, so they cannot leak label information.
    Returns X (log2), y (1 = AD), info dict.
    """
    X, clin = load_dataset(f"GSE248417_{kind}")
    info = {"samples_raw": len(X), "features_raw": X.shape[1]}
    if kind == "circRNA":
        ok = X.sum(axis=1) >= 1e5
        info["failed_samples"] = clin.loc[~ok, "diagnosis"].value_counts().to_dict()
        X, clin = X[ok], clin[ok]
    X = X.loc[:, (X > 1).mean() >= min_detect]
    info["features_detected"] = X.shape[1]
    X = normalize(np.log2(X + 1), norm)
    if drop is not None:
        keep = ~X.index.isin(list(drop))
        X, clin = X[keep], clin[keep]
    if top_var and X.shape[1] > top_var:
        X = X[X.var().sort_values(ascending=False).index[:top_var]]
    info["features_used"] = X.shape[1]
    y = (clin["diagnosis"] == "AD").astype(int)
    info.update(n=len(y), AD=int(y.sum()), NC=int((1 - y).sum()))
    return X, y, info


def _one_repeat(X, y, algo, rep, outer_k, inner_k, k_grid):
    seed = SEED + rep
    clf, grid = classifiers(y)[algo]
    pipe = build_pipeline(clf, list(X.columns), [])
    grid = {**grid, "features__ncrna__select__k": [k for k in k_grid if k == "all" or k < X.shape[1]]}
    oof = pd.Series(np.nan, index=X.index)
    params = []
    for tr, te in StratifiedKFold(outer_k, shuffle=True, random_state=seed).split(X, y):
        gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc", n_jobs=1,
                          cv=StratifiedKFold(inner_k, shuffle=True, random_state=seed)).fit(X.iloc[tr], y.iloc[tr])
        m = gs.best_estimator_
        oof.iloc[te] = (m.predict_proba(X.iloc[te])[:, 1] if hasattr(m, "predict_proba")
                        else m.decision_function(X.iloc[te]))
        params.append(gs.best_params_)
    return oof, params


def repeated_nested_cv(X, y, algo, n_repeats=5, outer_k=5, inner_k=3, k_grid=K_GRID_SEC, n_jobs=-1):
    """Returns dict: aucs (per repeat), oof (DataFrame, one column per repeat), params."""
    res = Parallel(n_jobs=n_jobs)(delayed(_one_repeat)(X, y, algo, r, outer_k, inner_k, k_grid)
                                  for r in range(n_repeats))
    oof = pd.concat([r[0] for r in res], axis=1, keys=range(n_repeats))
    return {"aucs": [roc_auc_score(y, oof[r]) for r in oof], "oof": oof,
            "params": [p for r in res for p in r[1]]}
