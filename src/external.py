"""External validation helpers (notebook 05).

The locked GSE120584 model was trained on 3D-Gene array values plus APOE4 and an array-specific
sample-quality covariate (detect_rate). The external cohorts use other technologies (small RNA-seq,
TaqMan qPCR, Affymetrix) and lack APOE4, so the trained model cannot be applied to them directly.
What *can* be transferred without refitting is the training **signature**: which miRNAs differ
between AD and NC, and in which direction. These helpers test whether that signature replicates.

Everything that defines the signature comes from the training split only.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import ROOT, SEED, load_dataset

TAB = ROOT / "results" / "tables"

# Platform-specific preprocessing and "detected" rule (decided before looking at any labels).
PLATFORM = {
    "GSE215789_plasma": "seq_counts",
    "GSE215789_serum":  "seq_counts",
    "GSE46579_blood":   "seq_normalized",
    "GSE305625_plasma": "qpcr",
    "GSE305625_csf":    "qpcr",
    "GSE157239_brain":  "array_log2",
}
QPCR_CONTROLS = ["cel-miR-39-3p", "hsa-miR-647", "hsa-miR-499a-3p", "hsa-miR-16-5p"]  # spike-in, negatives, normalizer


def training_signature(fdr_col="fdr_adj_q", beta_col="beta_adj_q", alpha=0.05):
    """AD-vs-NC miRNAs from the training split (adjusted for age, sex, APOE4, detect_rate).
    Returns DataFrame indexed by miRNA with columns beta (training effect) and sign (+1 up in AD)."""
    t = pd.read_csv(TAB / "01_eda_AD_vs_NC_univariate.csv").set_index("miRNA")
    sig = t[t[fdr_col] < alpha].copy()
    sig["beta"] = sig[beta_col]
    sig["sign"] = np.sign(sig["beta"]).astype(int)
    return sig[["beta", "sign", fdr_col]].sort_values(fdr_col)


def prepare(name, min_frac=0.5):
    """Load an external dataset, put it on a log scale and flag detected miRNAs.
    Returns (X_log, clin, detected: list of miRNAs)."""
    X, clin = load_dataset(name)
    kind = PLATFORM[name]
    if kind == "seq_counts":
        cpm = X.div(X.sum(axis=1), axis=0) * 1e6
        detected = X.columns[(X >= 5).mean() >= min_frac]
        Xl = np.log2(cpm + 1)
    elif kind == "seq_normalized":
        detected = X.columns[(X >= 5).mean() >= min_frac]
        Xl = np.log2(X + 1)
    elif kind == "qpcr":                      # values already log2-like, higher = more expression
        X = X.drop(columns=[c for c in QPCR_CONTROLS if c in X.columns])
        detected = X.columns[X.isna().mean() <= 0.2]
        Xl = X.fillna(X.median())
    elif kind == "array_log2":
        detected = X.columns[(X > 3).mean() >= min_frac]
        Xl = X
    else:
        raise ValueError(kind)
    return Xl, clin, list(detected)


def zscore(X):
    """Within-dataset standardisation (label-free)."""
    return (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)


def signature_score(Xl, signs):
    """Locked score: mean of sign * z-score over the signature miRNAs given (no refitting)."""
    Z = zscore(Xl[signs.index])
    return (Z * signs).mean(axis=1, skipna=True)


def perm_auc_p(y, s, n=2000, seed=SEED):
    """Two-sided permutation p-value for |AUC - 0.5| under shuffled labels
    (an AUC well below 0.5 means the training direction is reversed in this cohort)."""
    rng = np.random.default_rng(seed)
    obs = abs(roc_auc_score(y, s) - 0.5)
    null = np.array([abs(roc_auc_score(rng.permutation(y), s) - 0.5) for _ in range(n)])
    return (1 + (null >= obs).sum()) / (n + 1)


def ad_nc_effects(Xl, y):
    """Per-miRNA external effect: mean difference AD - NC and Mann-Whitney p."""
    a, b = Xl[y == 1], Xl[y == 0]
    p = [stats.mannwhitneyu(a[c].dropna(), b[c].dropna()).pvalue for c in Xl.columns]
    return pd.DataFrame({"ext_diff": a.mean() - b.mean(), "ext_p": p}, index=Xl.columns)


def refit_cv_auc(F, y, n_splits=5, n_repeats=10, seed=SEED, C=0.1):
    """Mean out-of-fold AUC of an L2 logistic regression refit *within* the external cohort."""
    k = int(min(n_splits, y.value_counts().min()))
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    aucs = []
    for r, (splits) in enumerate(_chunk(list(cv.split(F, y)), k)):
        pred = pd.Series(np.nan, index=F.index)
        for tr, te in splits:
            pred.iloc[te] = model.fit(F.iloc[tr], y.iloc[tr]).predict_proba(F.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return float(np.mean(aucs))


def _chunk(lst, k):
    return [lst[i:i + k] for i in range(0, len(lst), k)]
