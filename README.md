# ncRNA Biomarkers for Early Alzheimer's Detection (AI approach)

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data pipeline
```bash
.venv/bin/python src/download_geo.py     # -> data/raw/<GSE>/   (skips files already present)
.venv/bin/python src/build_datasets.py   # -> data/processed/<dataset>_expr.csv / _clinical.csv
```

One-time setup (already done; do NOT re-run, the test set must stay fixed):
```bash
.venv/bin/python src/make_split.py          # -> data/splits/GSE120584_serum_split.csv (80/20)
.venv/bin/python src/check_known_ncrnas.py  # -> data/reference/known_ncrna_coverage.csv
```

Load data in notebooks with `src/utils.py`:
```python
import sys; sys.path.append("../src")
from utils import load_dataset, SEED
X, clin = load_dataset("GSE120584_serum", split="train", diagnoses=["AD", "NC"])
```

The analysis plan is in [docs/analysis_plan.md](docs/analysis_plan.md).

Every processed dataset has:
- `<name>_expr.csv`: samples x features
- `<name>_clinical.csv`: sample, gsm, dataset, tissue, diagnosis (AD / MCI / NC / other), age, sex

`data/processed/dataset_summary.csv` lists sizes, value types and available covariates.

## Datasets and roles
| Dataset | Tissue | ncRNA | AD / MCI / NC | Role |
|---|---|---|---|---|
| GSE120584 | serum | miRNA (array) | 1021 / 32 / 288 (+DLB, VaD) | Discovery / training |
| GSE215789 | serum + plasma | miRNA (seq) | 28 / 63 / 21 (serum) | External validation, MCI (early detection) |
| GSE305625 | plasma + CSF | 64 candidate miRNAs (qPCR) | 71 / 0 / 74 (plasma) | External validation of top miRNAs |
| GSE46579 | whole blood | miRNA (seq) | 48 / 0 / 22 | External validation |
| GSE248417 | whole blood | lncRNA + circRNA (seq) | 49 / 0 / 49 | Separate lncRNA / circRNA analysis |
| GSE157239 | brain cortex | miRNA (array) | 8 / 0 / 8 | Biological support only (not a blood test) |

Dropped from the original list of 10: GSE161199 (PBMC mRNA, raw FASTQ only, 6 AD),
GSE248423 (SuperSeries wrapping GSE248417, no extra data), GSE212623 (33 CSF miRNAs, old names)
and GSE218605 (6 AD vs 6 controls).

Value scales differ (log2 array, raw counts, normalized counts, qPCR, TPM),
so each dataset must be normalized/scaled separately before any cross-dataset comparison.
