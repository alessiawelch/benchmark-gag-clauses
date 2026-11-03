#!/usr/bin/env python3
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

CLAUSES = {"Benchmarking Results": "Benchmarking and Results Restrictions",
           "Competitive Analysis": "Competitive Analysis Restrictions",
           "Evaluation Confidentiality": "Evaluation-Related Confidentiality Restrictions",
           "Service Publicity": "Service-Related Publicity Restrictions",
           "Conditional Benchmarking": "Conditional Benchmarking Permissions"}
PRIO = ["Service", "Mixed", "Site-only", "Sales terms"]
TYPES = [("Service", ["Service"]), ("Mixed", ["Mixed"]), ("Product terms", ["Service", "Mixed"]),
         ("Website-only", ["Site-only"]), ("Sales terms", ["Sales terms"]), ("All", PRIO)]
COLS = ["Any Clause"] + list(CLAUSES)

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "documents.jsonl"), encoding="utf-8")]
docs = pd.DataFrame([{"company": r["company_domain"], "scope": r["scope"]} for r in rows])
for col, name in CLAUSES.items():
    docs[col] = [any(c["label"] == name for c in r["clauses"]) for r in rows]
docs["prio"] = docs.scope.map({s: i for i, s in enumerate(PRIO)})
firms = docs.groupby("company").agg(prio=("prio", "min"), **{c: (c, "max") for c in CLAUSES}).reset_index()
firms["scope"] = firms.prio.map(dict(enumerate(PRIO)))

# ---------------------------------------------------------------- table
out = []
for typ, sc in TYPES:
    for unit, u in [("Docs", docs), ("Firms", firms)]:
        s = u[u.scope.isin(sc)]
        r = {"type": typ, "unit": unit, "N": len(s), "Any Clause": int(s[list(CLAUSES)].any(axis=1).sum())}
        r.update({c: int(s[c].sum()) for c in CLAUSES})
        out.append(r)
t = pd.DataFrame(out)
t.to_csv(os.path.join(OUT, "table_2_scope_prevalence.csv"), index=False)

cell = lambda k, n: "0" if k == 0 else f"{k:,} ({100 * k / n:.2f})"
print("Table 6. Prevalence of the five clauses by document type, for documents and for firms (companies).\n")
head = f"{'Type':15s}{'Unit':6s}{'N':>8s}" + "".join(f"{c:>{max(len(c), 13) + 2}s}" for c in COLS)
print(head); print("-" * len(head))
for _, r in t.iterrows():
    print(f"{r.type if r.unit == 'Docs' else '':15s}{r.unit:6s}{r.N:>8,}" + "".join(f"{cell(r[c], r.N):>{max(len(c), 13) + 2}s}" for c in COLS))
