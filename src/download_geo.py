"""Download every raw GEO file this project uses into data/raw/<GSE>/.

Usage:
    .venv/bin/python src/download_geo.py            # all datasets
    .venv/bin/python src/download_geo.py GSE215789  # just one
Files that already exist are skipped.
"""
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Supplementary files needed per dataset (series matrix is always downloaded).
SUPPL = {
    "GSE120584": [],
    "GSE215789": ["GSE215789_miR.Counts_plasma.csv.gz", "GSE215789_miR.Counts_serum.csv.gz"],
    "GSE305625": ["GSE305625_normalized_data.txt.gz"],
    "GSE46579": ["GSE46579_AD_ngs_data_summarized.xls.gz"],
    "GSE248417": ["GSE248417_ADvsControl.lncRNA.transcript_level.deg.txt.gz",
                  "GSE248417_circRNA_ADvsControl_deg.xlsx"],
    "GSE157239": [],
}
# Platform annotations needed to map probe IDs to miRNA names.
PLATFORMS = {"GSE120584": "GPL21263", "GSE157239": "GPL21572"}


def fetch(url, dest):
    if dest.exists():
        print(f"  exists  {dest.relative_to(ROOT)}")
        return
    print(f"  get     {dest.relative_to(ROOT)}")
    urllib.request.urlretrieve(url, dest)
    time.sleep(1)  # be polite to NCBI


def download(gse):
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}"
    d = RAW / gse
    d.mkdir(parents=True, exist_ok=True)
    print(gse)
    fetch(f"{base}/matrix/{gse}_series_matrix.txt.gz", d / f"{gse}_series_matrix.txt.gz")
    for f in SUPPL[gse]:
        if f.endswith(".xls.gz") and (d / f[:-3]).exists():
            print(f"  exists  {(d / f[:-3]).relative_to(ROOT)}")
            continue
        fetch(f"{base}/suppl/{f}", d / f)
        if f.endswith(".xls.gz"):  # xlrd can't read gzipped Excel
            import gzip, shutil
            out = d / f[:-3]
            if not out.exists():
                with gzip.open(d / f) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            (d / f).unlink()
    gpl = PLATFORMS.get(gse)
    if gpl and not (d / "GPL21572_human_miRNA.csv").exists():
        ann = d / f"{gpl}_annotation.txt"
        fetch(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&form=text&view=data", ann)
        if gpl == "GPL21572":  # 97 MB multi-species file -> keep only human miRNAs
            with open(ann) as fh:
                skip = [i for i, l in enumerate(fh) if l.startswith(("!", "#", "^"))]
            a = pd.read_csv(ann, sep="\t", skiprows=skip,
                            usecols=["ID", "Accession", "Transcript ID(Array Design)",
                                     "Sequence Type", "Species Scientific Name"])
            a = a[(a["Species Scientific Name"] == "Homo sapiens") & (a["Sequence Type"] == "miRNA")]
            a.rename(columns={"Transcript ID(Array Design)": "miRNA"})[["ID", "Accession", "miRNA"]] \
                .to_csv(d / "GPL21572_human_miRNA.csv", index=False)
            ann.unlink()


if __name__ == "__main__":
    for g in sys.argv[1:] or SUPPL:
        download(g)
