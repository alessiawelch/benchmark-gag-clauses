#!/usr/bin/env python3
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
MIN_DOCS = 150


rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "documents.jsonl"), encoding="utf-8")]
d = pd.DataFrame([{"spciq_id": r["spciq_id"], "clause": bool(r["clauses"])} for r in rows if r["product_terms"]])
ci = pd.read_csv(os.path.join(ROOT, "data", "capiq", "capiq_company_info.csv"), encoding="utf-8-sig", dtype={"spciq_id": str},
                 keep_default_na=False, na_values=[""])
d = d.merge(ci[["spciq_id", "iq_industry"]].rename(columns={"iq_industry": "industry"}), on="spciq_id", how="left")
d["matched"] = d.spciq_id.notna()


def row(name, s):
    return {"industry": name, "docs": len(s), "pct_docs": 100 * len(s) / len(d),
            "pct_clause_docs": 100 * s.clause.sum() / d.clause.sum(), "pct_with_clause": 100 * s.clause.mean(),
            "with_clause": int(s.clause.sum())}


size = d.industry.value_counts()
big = size[size >= MIN_DOCS].index
rest = d.industry.notna() & ~d.industry.isin(big)
t = pd.DataFrame(sorted([row(i, d[d.industry == i]) for i in big], key=lambda r: -r["pct_with_clause"]) +
                 [row(f"Other industries ({d[rest].industry.nunique()})", d[rest]),
                  row("Matched, no GICS industry", d[d.matched & d.industry.isna()]),
                  row("Matched to Capital IQ", d[d.matched]), row("Not matched to Capital IQ", d[~d.matched]),
                  row("All product terms", d)])
GICS = {"Electronic Equipment, Instruments and Components": "Electronic Equipment and Instruments"}
t["industry"] = [GICS.get(s, s).replace(" and ", " & ") for s in t.industry]       
t.to_csv(os.path.join(OUT, "table_3_industry_prevalence.csv"), index=False)

print(f"{'Industry':34s}{'Docs (n)':>10s}{'% of Docs':>11s}{'% of Clause Docs':>18s}{'% with Clause (n)':>20s}")
print("-" * 93)
for _, r in t.iterrows():
    print(f"{r.industry:34s}{r.docs:>10,}{r.pct_docs:>11.1f}{r.pct_clause_docs:>18.1f}{f'{r.pct_with_clause:.1f} ({r.with_clause:,})':>20s}")
