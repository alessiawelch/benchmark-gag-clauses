import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
CUT = pd.Timestamp("2026-09-22")                 # the TermTrove copy and every archive date stop here

PAPER_NAME = {
    "Core DeWitt Clause": "Benchmarking and Results Restrictions",
    "DeWitt Embrace Clause": "Conditional Benchmarking Permissions",
    "Competitive Ban": "Competitive Analysis Restrictions",
    "Publicity Gag Clause": "Service-Related Publicity Restrictions",
    "Confidential Information Restriction": "Evaluation-Related Confidentiality Restrictions",
}
LABELS = list(PAPER_NAME)
DB_NAME = {v: k for k, v in PAPER_NAME.items()}


def _read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def documents(runs=None):
    """Documents with one bool column per label (clause in the latest version). runs: keep only clauses from these runs."""
    rows = _read("documents.jsonl")
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "clauses"} for r in rows])
    for l in LABELS:
        d[l] = [any(DB_NAME[c["label"]] == l and (runs is None or c["run"] in runs) for c in r["clauses"]) for r in rows]
    return d


def history(sources=("wayback", "termtrove")):
    """Archived observations (dated <= CUT) with one bool column per label."""
    rows = [r for r in _read("history.jsonl") if r["source"] in sources]
    h = pd.DataFrame(rows)
    for l in LABELS:
        h[l] = [PAPER_NAME[l] in r["labels"] for r in rows]
    h["date"] = pd.to_datetime(h.date)
    return h.drop(columns="labels")


def observed_years(docs):
    """Document x year: does any version that year (a Wayback capture or a TermTrove version) carry each clause?"""
    h = history()
    h = h[h.document_id.isin(set(docs))].copy()
    h["year"] = h.date.dt.year
    return h


def year_status(docs):
    """Document x year: the clauses in the document's latest version that year (a Wayback capture or a TermTrove version)."""
    h = observed_years(docs)
    last = h.groupby(["document_id", "year"]).date.transform("max")
    return h[h.date == last].groupby(["document_id", "year"])[LABELS].any()


def capiq():
    p = os.path.join(DATA, "capiq", "capiq_company_info.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Capital IQ file not found: {p}\nThe industry and revenue outputs need it; it cannot be redistributed.")
    ci = pd.read_csv(p, encoding="utf-8-sig", dtype={"spciq_id": str}, usecols=["spciq_id", "iq_industry", "iq_total_rev"],
                     keep_default_na=False, na_values=[""])                # "NA" etc. stay text
    return ci.rename(columns={"iq_total_rev": "revenue"})               # total revenue, thousands of US dollars
