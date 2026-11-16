"""Check which literature AD ncRNAs are measured in each processed dataset.

Writes data/reference/known_ncrna_coverage.csv (rows = known ncRNAs, columns = datasets).
miRNA names are matched exactly or, for old-style names (e.g. hsa-miR-107), against the -5p/-3p arm.
lncRNA/circRNA names are matched against gene/transcript names in GSE248417 feature IDs.
"""
import re

import pandas as pd

from utils import PROCESSED, REFERENCE, list_datasets, load_known_ncrnas

known = load_known_ncrnas()
cov = pd.DataFrame(index=known["name"])

for name in list_datasets()["name"]:
    feats = pd.read_csv(PROCESSED / f"{name}_expr.csv", nrows=0).columns[1:]
    fset = set(feats)
    hits = []
    for n, t in zip(known["name"], known["type"]):
        if t == "miRNA":
            cands = [n, n + "-5p", n + "-3p", re.sub(r"-[35]p$", "", n)]
            hit = next((c for c in cands if c in fset), "")
            if not hit:  # merged features in seq data, e.g. 'hsa-miR-103a-3p/107'
                short = n.replace("hsa-", "").replace("miR-", "")
                hit = next((f for f in feats if "/" in f and short in
                            [p.replace("hsa-", "").replace("miR-", "") for p in f.split("/")]), "")
        else:
            pat = re.compile(rf"(^|\|){re.escape(n)}($|[-|])", re.I)
            hit = next((f for f in feats if pat.search(f)), "")
        hits.append(hit)
    cov[name] = hits

cov.to_csv(REFERENCE / "known_ncrna_coverage.csv")
present = (cov != "").astype(int)
print(present.rename(columns=lambda c: c.replace("_", "\n")).to_string())
print("\nFound per dataset:\n" + present.sum().to_string())
