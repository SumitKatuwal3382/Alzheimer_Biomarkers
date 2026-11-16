"""Harmonize the project's GEO datasets into a common format.

For each dataset/tissue this writes to data/processed/:
    <name>_expr.csv      samples x features (feature = miRNA / lncRNA / circRNA name)
    <name>_clinical.csv  sample, gsm, dataset, tissue, diagnosis (AD/MCI/NC/other), age, sex, ...
plus data/processed/dataset_summary.csv.

Run src/download_geo.py first.
"""
import gzip
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DIAG_MAP = {
    "ad": "AD", "alzheimer's disease": "AD", "alzheimer patient": "AD",
    "mci": "MCI",
    "nc": "NC", "control": "NC", "healthy control": "NC",
}


# ------------------------------------------------------------------ helpers
def harmonize_dx(v):
    return np.nan if pd.isna(v) else DIAG_MAP.get(str(v).strip().lower(), str(v).strip())


def parse_age(v):
    m = re.search(r"\d+(\.\d+)?", str(v)) if pd.notna(v) else None
    return float(m.group()) if m else np.nan


def geo_metadata(gse):
    """Per-sample metadata from a series matrix; characteristics parsed by key."""
    fields = {}
    with gzip.open(RAW / gse / f"{gse}_series_matrix.txt.gz", "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
                fields.setdefault(parts[0][8:], []).append(parts[1:])
    rows = []
    for i in range(len(fields["geo_accession"][0])):
        r = {"gsm": fields["geo_accession"][0][i],
             "title": fields["title"][0][i],
             "descriptions": [d[i] for d in fields.get("description", [])]}
        for ch in fields.get("characteristics_ch1", []):
            if ": " in ch[i]:
                k, v = ch[i].split(": ", 1)
                r[k.strip().lower()] = v.strip()
        rows.append(r)
    return pd.DataFrame(rows)


def matrix_table(gse):
    """Expression table embedded in a series matrix (features x GSM)."""
    with gzip.open(RAW / gse / f"{gse}_series_matrix.txt.gz", "rt", errors="replace") as f:
        text = f.read().split("!series_matrix_table_begin\n")[1].split("!series_matrix_table_end")[0]
    return pd.read_csv(StringIO(text), sep="\t", index_col=0)


def read_platform(path):
    with open(path) as f:
        skip = [i for i, l in enumerate(f) if l.startswith(("!", "#", "^"))]
    return pd.read_csv(path, sep="\t", skiprows=skip)


def clinical(meta, dataset, tissue, dx_key, sample_ids, extra=None):
    sex = meta["sex"] if "sex" in meta else meta.get("gender")
    c = pd.DataFrame({
        "gsm": meta["gsm"].values,
        "dataset": dataset,
        "tissue": tissue,
        "diagnosis": meta[dx_key].map(harmonize_dx).values,
        "age": meta["age"].map(parse_age).values if "age" in meta else np.nan,
        "sex": sex.str.lower().values if sex is not None else np.nan,
    }, index=pd.Index(sample_ids, name="sample"))
    for k, v in (extra or {}).items():
        c[k] = v
    return c


def save(name, expr, clin, feature_type, value_type):
    expr = expr.loc[clin.index]
    expr.index.name = "sample"
    expr.to_csv(OUT / f"{name}_expr.csv")
    clin.to_csv(OUT / f"{name}_clinical.csv")
    counts = clin["diagnosis"].value_counts().to_dict()
    print(f"{name:20s} {expr.shape[0]:5d} samples x {expr.shape[1]:6d} features  {counts}")
    return {"name": name, "dataset": clin["dataset"].iloc[0], "tissue": clin["tissue"].iloc[0],
            "feature_type": feature_type, "value_type": value_type,
            "n_samples": expr.shape[0], "n_features": expr.shape[1],
            **{f"n_{k}": counts.get(k, 0) for k in ["AD", "MCI", "NC"]},
            "n_other": sum(v for k, v in counts.items() if k not in ("AD", "MCI", "NC")),
            "has_age": clin["age"].notna().any(), "has_sex": clin["sex"].notna().any()}


summary = []

# ---------------------------------------------------------------- GSE120584  (training)
m = geo_metadata("GSE120584")
d = matrix_table("GSE120584")
ann = read_platform(RAW / "GSE120584" / "GPL21263_annotation.txt")
id2name = dict(zip(ann["ID"], ann["miRNA_ID_LIST"]))
d.index = [id2name.get(i, i) for i in d.index]
c = clinical(m, "GSE120584", "serum", "diagnosis", m["gsm"].values,
             {"apoe4": pd.to_numeric(m["apoe4"]).values})
summary.append(save("GSE120584_serum", d[m["gsm"]].T, c, "miRNA", "log2 normalized (3D-Gene array)"))

# ---------------------------------------------------------------- GSE215789  (MCI validation)
m = geo_metadata("GSE215789")
m["file_col"] = m["descriptions"].str[0]
for tissue in ["Plasma", "Serum"]:
    counts = pd.read_csv(RAW / "GSE215789" / f"GSE215789_miR.Counts_{tissue.lower()}.csv.gz", index_col=0)
    mm = m[(m["tissue"] == tissue) & m["file_col"].isin(counts.columns)]
    expr = counts[mm["file_col"]].T
    expr.index = mm["gsm"].values
    c = clinical(mm, "GSE215789", tissue.lower(), "disease state", mm["gsm"].values,
                 {"subject": mm["subject"].values})
    summary.append(save(f"GSE215789_{tissue.lower()}", expr, c, "miRNA", "raw read counts (small RNA-seq)"))

# ---------------------------------------------------------------- GSE305625  (qPCR validation)
m = geo_metadata("GSE305625")
m["file_col"] = m["descriptions"].str[1]
d = pd.read_csv(RAW / "GSE305625" / "GSE305625_normalized_data.txt.gz", sep="\t")
d = d.set_index("mirBaseV21 ID")   # headers are shifted: this column actually holds the miRNA name
d = d[[col for col in d.columns if col.startswith("JSAUG")]]
for tissue_raw, tissue in [("Plasma", "plasma"), ("Cerebral Spinal Fluid", "csf")]:
    mm = m[(m["tissue"] == tissue_raw) & m["file_col"].isin(d.columns)]
    expr = d[mm["file_col"]].T
    expr.index = mm["gsm"].values
    keep = expr.notna().any(axis=1).values   # 32 samples excluded by the authors are all-NaN
    mm, expr = mm[keep], expr[keep]
    c = clinical(mm, "GSE305625", tissue, "subject status", mm["gsm"].values,
                 {"donor_id": mm["donor id"].values})
    summary.append(save(f"GSE305625_{tissue}", expr, c, "miRNA (64 candidates)",
                        "normalized RT-qPCR (TaqMan array); NaN = missing/censored"))

# ---------------------------------------------------------------- GSE46579  (blood validation)
m = geo_metadata("GSE46579")
x = pd.read_excel(RAW / "GSE46579" / "GSE46579_AD_ngs_data_summarized.xls",
                  sheet_name="normalized", index_col=0)
x.index = x.index.str.split(":").str[-1]          # 'hsa-mir-30a:hsa-miR-30a-3p' -> mature name
x = x.groupby(level=0).sum()
# Columns are 'AD, AD.1, ..., control, control.1, ...' in the same order as the GEO sample titles.
grp = x.columns.str.replace(r"\.\d+$", "", regex=True)
ad, ctl = m[m["group"] == "alzheimer patient"], m[m["group"] == "control"]
assert (grp == "AD").sum() == len(ad) and (grp == "control").sum() == len(ctl)
mm = pd.concat([ad, ctl])
expr = x.T
expr.index = mm["gsm"].values
c = clinical(mm, "GSE46579", "whole blood", "group", mm["gsm"].values)
c[["age", "sex"]] = np.nan   # can't be reliably matched to data columns
summary.append(save("GSE46579_blood", expr, c, "miRNA", "normalized read counts (small RNA-seq)"))

# ---------------------------------------------------------------- GSE248417  (lncRNA + circRNA)
m = geo_metadata("GSE248417")
m["file_col"] = m["descriptions"].str[0]
c = clinical(m, "GSE248417", "whole blood", "disease state", m["gsm"].values)

d = pd.read_csv(RAW / "GSE248417" / "GSE248417_ADvsControl.lncRNA.transcript_level.deg.txt.gz", sep="\t")
d.index = d["transcript_id"] + "|" + d["transcript_name"].astype(str)
expr = d[m["file_col"]].T.astype(float)
expr.index = m["gsm"].values
summary.append(save("GSE248417_lncRNA", expr, c, "lncRNA transcripts", "FPKM/TPM-like (RNA-seq)"))

x = pd.read_excel(RAW / "GSE248417" / "GSE248417_circRNA_ADvsControl_deg.xlsx")
x = x.dropna(subset=["circRNA"]).set_index("circRNA")   # sheet has ~1M trailing empty rows
expr = x[[f"{s}_tpm" for s in m["file_col"]]].astype(float).T
expr.index = m["gsm"].values
summary.append(save("GSE248417_circRNA", expr, c, "circRNA", "TPM (RNA-seq)"))

# ---------------------------------------------------------------- GSE157239  (brain; biological support)
m = geo_metadata("GSE157239")
d = matrix_table("GSE157239")
ann = pd.read_csv(RAW / "GSE157239" / "GPL21572_human_miRNA.csv")
d = d.loc[d.index.intersection(ann["ID"])]
d.index = ann.set_index("ID").loc[d.index, "miRNA"].values
m["dx"] = np.where(m["title"].str.startswith("AD"), "AD", "NC")
c = clinical(m, "GSE157239", "brain (temporal cortex)", "dx", m["gsm"].values)
summary.append(save("GSE157239_brain", d.T, c, "miRNA (mature, human)", "log2 RMA (Affymetrix miRNA 4.1)"))

pd.DataFrame(summary).to_csv(OUT / "dataset_summary.csv", index=False)
print(f"\nSummary written to {OUT / 'dataset_summary.csv'}")
