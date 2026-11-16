# Analysis Plan (draft, fix before modeling)

**Project:** Can ncRNAs be biomarkers for early detection of Alzheimer's disease? An AI approach
**Status:** DRAFT. Items marked **[DECIDE]** need confirmation with the advisor.
**Date fixed:** ____________

## 1. Questions
| # | Question | Data | Primary metric |
|---|---|---|---|
| Q1 (primary) | Do serum miRNAs distinguish AD from cognitively normal controls (NC)? | GSE120584 serum (train split) | ROC AUC |
| Q2 (early detection) | Does the Q1 model / signature separate MCI from NC? | GSE215789 serum + plasma (MCI vs NC) | ROC AUC |
| Q3 (specificity) | Is the signature specific to AD vs other dementias (DLB, VaD)? | GSE120584 serum | ROC AUC |
| Q4 (replication) | Do the top miRNAs show the same direction of change in independent cohorts? | GSE305625 plasma, GSE46579 blood, GSE215789 | Direction agreement, per-miRNA AUC |
| Q5 (exploratory) | Do blood lncRNAs/circRNAs separate AD from NC? | GSE248417 | Cross-validated AUC (no external validation available) |

## 2. Data splits (fixed)
- GSE120584 serum: 80% train / 20% held-out test, stratified by diagnosis x sex, seed 42
  (`data/splits/GSE120584_serum_split.csv`). Test set used **once**, for the final model only.
- All EDA, feature filtering, feature selection, tuning and model choice use the train split only.
- External datasets (GSE215789, GSE305625, GSE46579) are never used for training or tuning.

## 3. Preprocessing (fit on training folds only)
- Remove low-detection miRNAs: **[DECIDE]** threshold, e.g. expressed above background in >= 50% of training samples.
- GSE120584 is already log2-normalized by the submitters; standardize (z-score) features inside CV.
- Sequencing data (GSE215789, GSE46579): log2(CPM + 1); scale within each dataset.
- No batch correction across datasets; each external dataset is scaled on its own.

## 4. Confounders
- Controls are younger than AD (GSE120584: 71.7 vs 79.2 y; GSE215789: 62 vs 72 y).
- **[DECIDE]** Approach (recommended: report all three models):
  - Model A: age + sex + APOE4 only (baseline; APOE4 available in GSE120584)
  - Model B: miRNAs only
  - Model C: miRNAs + age + sex + APOE4. The added value of miRNAs = AUC(C) - AUC(A).
- Sensitivity analysis: age-matched AD vs NC subset.

## 5. Models and validation
- Models: Logistic Regression (L1/L2), Random Forest, SVM, KNN, Gradient Boosting, XGBoost.
- Nested CV on the train split: outer stratified 5-fold (performance), inner 5-fold (tuning). Seed 42.
- Feature selection (LASSO / RFE / RF importance) runs **inside** the CV folds.
- Class imbalance: class weights; report AUC, balanced accuracy, sensitivity, specificity, F1.
- Model choice: highest mean outer-fold AUC; ties go to the simpler model.

## 6. Definition of the "top 10 ncRNAs"
**[DECIDE]** Proposed: rank miRNAs by selection frequency across outer CV folds and agreement across
>= 2 methods (LASSO coefficients, RF importance, SHAP). Top 10 by mean rank.
Then: direction and AUC checked in GSE215789, GSE305625, GSE46579; compared to `data/reference/known_ad_ncrnas.csv`.

## 7. Success criteria (set in advance)
- **[DECIDE]** e.g. held-out test AUC >= 0.80 for AD vs NC, and miRNA model beats age/sex baseline.
- Early detection: MCI vs NC AUC in GSE215789 reported regardless of result (expected lower).
- Benchmark: original GSE120584 publication, Shigemizu et al., *Commun Biol* 2019
  (doi:10.1038/s42003-019-0324-7, PMC6389908). Their setup: ~50/50 discovery/validation split
  (786 / 783; 144 NC each), MCI kept out as a separate cohort. Reported for AD (validation):

  | Model | AD AUC | Notes |
  |---|---|---|
  | Clinical only (age, sex, APOE), logistic regression | 0.857 | |
  | 78-miRNA model (supervised PCA, 2-step) | 0.874 | accuracy 0.873, sensitivity 0.933, **specificity 0.660** |
  | LASSO on all miRNAs | 0.898 | reported in passing |
  | MCI cohort (n = 32, 10 converters) | | sensitivity 1.0, **specificity 0.18** |

  Main lesson: miRNAs added only about +0.02 AUC over clinical variables, and specificity was low. Our
  contribution should be measured as **gain over the age/sex/APOE baseline** (Section 4, Model A vs C),
  better specificity, and external/MCI validation, not raw AUC alone.
  **[TODO]** Confirm their exact control group (AD vs NC, or AD vs all non-AD) and whether the miRNA model
  included age/sex/APOE; get their 78-miRNA list from Supplementary Data 2 to compare with our top 10.

## 8. Reporting
- All metrics reported with 95% CIs (bootstrap) and fold-level spread.
- Negative or null results reported as well.
