"""Model pipelines, hyperparameter grids and nested cross-validation.

Everything that learns from data (scaling, feature selection, the classifier) lives inside a
sklearn Pipeline, so it is re-fit inside every CV fold and no information leaks between folds.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils import SEED

CLINICAL = ["age", "female", "apoe4"]
QUALITY = ["detect_rate"]

# Feature sets compared in the modeling notebook
FEATURE_SETS = {
    "A_clinical":          {"covariates": CLINICAL,           "ncrna": False},
    "A2_clinical+quality": {"covariates": CLINICAL + QUALITY, "ncrna": False},
    "B_ncrna":             {"covariates": [],                 "ncrna": True},
    "C_ncrna+clinical":    {"covariates": CLINICAL + QUALITY, "ncrna": True},
}

K_GRID = [50, 200, "all"]   # number of ncRNAs kept by univariate selection (inside CV)


def detection_rate(X):
    """Per-sample fraction of features above that sample's minimum ('not detected' filler value).
    Uses only that sample's own values, so it cannot leak information between samples."""
    return X.sub(X.min(axis=1), axis=0).gt(1e-6).mean(axis=1)


def make_features(X, clin):
    """Combine ncRNA matrix with numeric covariates into one DataFrame."""
    cov = pd.DataFrame({
        "age": clin["age"],
        "female": (clin["sex"] == "female").astype(int),
        "apoe4": clin["apoe4"] if "apoe4" in clin else np.nan,
        "detect_rate": clin["detect_rate"] if "detect_rate" in clin else np.nan,
    }, index=clin.index)
    return pd.concat([X, cov], axis=1)


def classifiers(y_train=None):
    """Classifier + hyperparameter grid for each algorithm. Class imbalance handled with class weights."""
    spw = 1.0 if y_train is None else (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "LogReg": (LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000),
                   {"clf__C": [0.01, 0.1, 1], "clf__l1_ratio": [0.0, 1.0]}),   # 0 = L2 (ridge), 1 = L1 (LASSO)
        "RandomForest": (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=1),
                         {"clf__max_features": ["sqrt", 0.1], "clf__min_samples_leaf": [1, 5]}),
        "SVM": (SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                {"clf__C": [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(weights="distance"),
                {"clf__n_neighbors": [5, 15, 31]}),
        "GradBoost": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                      {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None]}),
        "XGBoost": (XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.5,
                                  scale_pos_weight=spw, eval_metric="logloss",
                                  random_state=SEED, n_jobs=1, verbosity=0),
                    {"clf__max_depth": [2, 4], "clf__learning_rate": [0.05, 0.1]}),
    }


def build_pipeline(clf, ncrna_cols, covariates):
    """Scale everything; select top-k ncRNAs (k tuned); covariates are always kept."""
    parts = []
    if ncrna_cols:
        parts.append(("ncrna", Pipeline([("scale", StandardScaler()),
                                         ("select", SelectKBest(f_classif, k="all"))]), list(ncrna_cols)))
    if covariates:
        parts.append(("cov", StandardScaler(), list(covariates)))
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", clf)])


def scores(model, X):
    """Continuous score for AUC (probability if available, else decision function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics(y, score, pred):
    return {"AUC": roc_auc_score(y, score),
            "accuracy": accuracy_score(y, pred),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "sensitivity": recall_score(y, pred, pos_label=1),
            "specificity": recall_score(y, pred, pos_label=0),
            "F1": f1_score(y, pred)}


def pick_threshold(y, score, rule="accuracy"):
    """Choose a decision threshold on TRAINING data only.
    rule='accuracy' maximizes accuracy; rule='balanced' maximizes balanced accuracy (Youden's J)."""
    y, score = np.asarray(y), np.asarray(score)
    cands = np.unique(score)
    if len(cands) > 400:
        cands = np.quantile(score, np.linspace(0, 1, 401))
    best_t, best_v = 0.5, -1
    for t in cands:
        p = (score >= t).astype(int)
        v = accuracy_score(y, p) if rule == "accuracy" else balanced_accuracy_score(y, p)
        if v > best_v:
            best_t, best_v = t, v
    return best_t


# Larger grids used in the "improve" round (notebook 02b)
BIG_GRIDS = {
    "LogReg": (lambda spw: LogisticRegression(solver="saga", class_weight="balanced", max_iter=3000, tol=1e-3),
               {"clf__C": [0.03, 0.1, 0.3], "clf__l1_ratio": [0.0, 0.5, 1.0]}),
    "XGBoost": (lambda spw: XGBClassifier(subsample=0.8, colsample_bytree=0.5, scale_pos_weight=spw,
                                          eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0),
                {"clf__n_estimators": [300, 600], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.06]}),
    "GradBoost": (lambda spw: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                  {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None], "clf__l2_regularization": [0.0, 1.0]}),
}


def nested_cv(F, y, ncrna_cols, feature_set, algo, outer_k=5, inner_k=3, seed=SEED, n_jobs=-1,
              big_grid=False, k_grid=None):
    """Nested CV for one (feature set, algorithm).

    Returns per-fold metrics and out-of-fold predictions. Also returns, per outer fold, the inner
    out-of-fold scores of the tuned model on the outer-TRAINING data, so decision thresholds (and
    ensembles) can be chosen without ever looking at the outer test fold.
    """
    fs = FEATURE_SETS[feature_set]
    cols = (list(ncrna_cols) if fs["ncrna"] else []) + fs["covariates"]
    outer = StratifiedKFold(outer_k, shuffle=True, random_state=seed)
    fold_rows, oof = [], pd.DataFrame(index=F.index, columns=["score", "pred", "fold"], dtype=float)
    inner_scores = {}
    for fold, (tr, te) in enumerate(outer.split(F, y)):
        ytr = y.iloc[tr]
        if big_grid:
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            make, grid = BIG_GRIDS[algo]
            clf = make(spw)
        else:
            clf, grid = classifiers(ytr)[algo]
        pipe = build_pipeline(clf, ncrna_cols if fs["ncrna"] else [], fs["covariates"])
        if fs["ncrna"]:
            kg = k_grid or K_GRID
            grid = {**grid, "features__ncrna__select__k": [k for k in kg if k == "all" or k < len(ncrna_cols)]}
        inner = StratifiedKFold(inner_k, shuffle=True, random_state=seed + fold)
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs, refit=True)
        gs.fit(F.iloc[tr][cols], ytr)
        # inner OOF scores of the chosen configuration (training data only)
        method = "predict_proba" if hasattr(gs.best_estimator_, "predict_proba") else "decision_function"
        s_in = cross_val_predict(clone(gs.best_estimator_), F.iloc[tr][cols], ytr, cv=inner,
                                 method=method, n_jobs=n_jobs)
        s_in = s_in[:, 1] if s_in.ndim == 2 else s_in
        inner_scores[fold] = pd.Series(s_in, index=F.index[tr])
        s = scores(gs.best_estimator_, F.iloc[te][cols])
        p = gs.best_estimator_.predict(F.iloc[te][cols])
        oof.iloc[te] = np.c_[s, p, np.full(len(te), fold)]
        fold_rows.append({"fold": fold, **metrics(y.iloc[te], s, p),
                          "best_params": gs.best_params_})
    return pd.DataFrame(fold_rows), oof, inner_scores


def evaluate_with_threshold(y, oof, inner_scores, rule="accuracy"):
    """Apply, in each outer fold, a threshold chosen on that fold's inner (training) scores."""
    pred = pd.Series(index=oof.index, dtype=float)
    for fold, s_in in inner_scores.items():
        t = pick_threshold(y.loc[s_in.index], s_in, rule)
        idx = oof.index[oof["fold"] == fold]
        pred[idx] = (oof.loc[idx, "score"] >= t).astype(int)
    return metrics(y, oof["score"], pred.astype(int)), pred


def bootstrap_auc_ci(y, score, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], score[i]))
    return np.percentile(vals, [2.5, 97.5])


def bootstrap_auc_diff(y, score_a, score_b, n=2000, seed=SEED):
    """Paired bootstrap of AUC(b) - AUC(a) on the same samples. Returns (diff, ci_low, ci_high, p_one_sided)."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, score_a, score_b))
    d0 = roc_auc_score(y, b) - roc_auc_score(y, a)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    diffs = np.array(diffs)
    return d0, *np.percentile(diffs, [2.5, 97.5]), (diffs <= 0).mean()
