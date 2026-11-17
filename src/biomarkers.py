"""Biomarker (ncRNA) ranking methods for Objective 3.

Four complementary methods rank the ncRNAs on a given training set. Clinical covariates are always
included in the models, so ncRNAs are ranked by what they add *beyond* age, sex, APOE4 and sample quality.

    1. LASSO        |coefficient| of an L1 logistic regression        (Aim 3.1)
    2. RFE          recursive feature elimination with L2 logistic   (Aim 3.1)
    3. RF           random forest impurity importance                 (Aim 3.1)
    4. SHAP         mean |SHAP| of an XGBoost model on held-out data  (Aim 3.2)

`stability_ranking` repeats this on many resamples of the training data; an ncRNA's score for each
method is how often it lands in that method's top-N ("selection frequency").
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import SEED

METHODS = ["LASSO", "RFE", "RF", "SHAP"]


def _scale(Ftr, Fte=None):
    sc = StandardScaler().fit(Ftr)
    out = pd.DataFrame(sc.transform(Ftr), index=Ftr.index, columns=Ftr.columns)
    if Fte is None:
        return out
    return out, pd.DataFrame(sc.transform(Fte), index=Fte.index, columns=Fte.columns)


def rank_once(Ftr, ytr, Fte, ncrna_cols, seed=SEED, rfe_keep=20):
    """Importance of every ncRNA under each method, fit on (Ftr, ytr); SHAP evaluated on Fte.
    Returns DataFrame (ncRNA x method) of importance scores (higher = more important)."""
    Ztr, Zte = _scale(Ftr, Fte)
    cols = list(Ftr.columns)
    out = pd.DataFrame(index=ncrna_cols, columns=METHODS, dtype=float)

    lasso = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=0.1, class_weight="balanced",
                               max_iter=5000).fit(Ztr, ytr)
    out["LASSO"] = pd.Series(np.abs(lasso.coef_[0]), index=cols)[ncrna_cols]

    rfe = RFE(LogisticRegression(solver="liblinear", C=0.1, class_weight="balanced", max_iter=5000),
              n_features_to_select=rfe_keep, step=0.1).fit(Ztr, ytr)
    out["RFE"] = -pd.Series(rfe.ranking_, index=cols)[ncrna_cols].astype(float)   # rank 1 = best

    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=seed, n_jobs=1).fit(Ztr, ytr)
    out["RF"] = pd.Series(rf.feature_importances_, index=cols)[ncrna_cols]

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.5, scale_pos_weight=spw, eval_metric="logloss",
                        random_state=seed, n_jobs=1, verbosity=0).fit(Ztr, ytr)
    sv = shap.TreeExplainer(xgb).shap_values(Zte)
    out["SHAP"] = pd.Series(np.abs(sv).mean(axis=0), index=cols)[ncrna_cols]
    return out


def stability_ranking(F, y, ncrna_cols, n_splits=5, n_repeats=5, top_n=20, seed=SEED, n_jobs=-1):
    """Repeat rank_once over repeated stratified K-fold splits of the given data.

    Returns:
        freq   ncRNA x method: fraction of resamples where the ncRNA is in that method's top_n
        ranks  ncRNA x method: mean rank (1 = most important) across resamples
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(cv.split(F, y))
    results = Parallel(n_jobs=n_jobs)(
        delayed(rank_once)(F.iloc[tr], y.iloc[tr], F.iloc[te], ncrna_cols, seed + i)
        for i, (tr, te) in enumerate(splits))
    ranks = [r.rank(ascending=False, method="min") for r in results]
    freq = sum((rk <= top_n).astype(float) for rk in ranks) / len(ranks)
    mean_rank = sum(ranks) / len(ranks)
    return freq, mean_rank


def consensus(freq, mean_rank, n_top=10, min_freq=0.5, min_methods=2):
    """Consensus table. An ncRNA qualifies if it is in the top-N in >= min_freq of resamples for
    at least `min_methods` methods; qualifying ncRNAs are ordered by mean selection frequency."""
    t = pd.DataFrame({f"freq_{m}": freq[m] for m in freq.columns})
    t["mean_freq"] = freq.mean(axis=1)
    t["n_methods_stable"] = (freq >= min_freq).sum(axis=1)
    t["mean_rank"] = mean_rank.mean(axis=1)
    t["qualifies"] = t["n_methods_stable"] >= min_methods
    t = t.sort_values(["qualifies", "mean_freq", "mean_rank"], ascending=[False, False, True])
    t["consensus_rank"] = np.arange(1, len(t) + 1)
    return t, t[t.qualifies].head(n_top).index.tolist()
