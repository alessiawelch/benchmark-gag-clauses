#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import LABELS, documents, year_status, capiq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
MIN_DOCS = 150
BINS = [0, 10, 100, 1000, 10000, np.inf]                             
REV = ["Under $10M", "$10M–$100M", "$100M–$1B", "$1B–$10B", "Over $10B"]

docs = documents()
docs["Any"] = docs[LABELS].any(axis=1)
ci = capiq()
docs = docs.merge(ci[["spciq_id", "iq_industry", "revenue"]].rename(columns={"iq_industry": "industry"}), on="spciq_id", how="left")
prod = docs[docs.product_terms].copy()
prod["matched"] = prod.spciq_id.notna()

# ---------------------------------------------------------------- panel A: industry, 2019 vs 2026
st = year_status(dict(zip(docs.document_id, docs.scope)))       # clauses in each document's latest version that year
yr = st[st.index.get_level_values("year").isin([2019, 2026])].any(axis=1).unstack("year")
yr = yr[yr[2019].notna() & yr[2026].notna()].astype(bool)              
pa = prod[prod.document_id.isin(yr.index)].copy()
pa["c19"], pa["c26"] = yr.loc[pa.document_id, 2019].values, yr.loc[pa.document_id, 2026].values


def change(s):
    a, b = int((~s.c19 & s.c26).sum()), int((s.c19 & ~s.c26).sum())
    return dict(documents=len(s), with_clause_2019=int(s.c19.sum()), with_clause_2026=int(s.c26.sum()), added=a, dropped=b,
                mcnemar_p=binomtest(a, a + b, 0.5).pvalue if a + b else 1.0)


size = prod.industry.value_counts()
industries = size[size >= MIN_DOCS].index                                # the 13 rows of the industry table
p_adj = dict(zip(industries, multipletests([change(pa[pa.industry == i])["mcnemar_p"] for i in industries], method="fdr_bh")[1]))
A = []
for label, s, adj in [("Software", pa[pa.industry == "Software"], p_adj["Software"]),
                      ("Media", pa[pa.industry == "Media"], p_adj["Media"]),
                      ("Interactive Media & Services", pa[pa.industry == "Interactive Media and Services"], p_adj["Interactive Media and Services"]),
                      ("Matched, excl. software", pa[pa.industry.notna() & pa.industry.ne("Software")], None),   # matched and with a GICS industry
                      ("All product terms", pa, None)]:
    r = change(s)
    r["pct_2019"], r["pct_2026"] = round(100 * r["with_clause_2019"] / r["documents"], 1), round(100 * r["with_clause_2026"] / r["documents"], 1)
    r["change_pp"] = round(r["pct_2026"] - r["pct_2019"], 1)            
    r["p_reported"] = r["mcnemar_p"] if adj is None else adj
    r["stars"] = "***" if r["p_reported"] < 0.01 else "**" if r["p_reported"] < 0.05 else "*" if r["p_reported"] < 0.10 else ""
    A.append({"group": label, **r})
A = pd.DataFrame(A)

# ---------------------------------------------------------------- panel B: companies by revenue
co = prod[prod.industry.notna()].groupby("company_domain").agg(clause=("Any", "max"), spciq_id=("spciq_id", "first"),
                                                               industry=("industry", "first")).reset_index()
co = co.merge(ci.loc[ci.revenue > 0, ["spciq_id", "revenue"]], on="spciq_id")
co = co.groupby("spciq_id").agg(clause=("clause", "max"), software=("industry", lambda s: s.iloc[0] == "Software"),
                                revenue=("revenue", "first")).reset_index()      
co["bin"] = pd.cut(co.revenue / 1000, BINS, labels=REV)                 
Bp = []
for grp, m in [("Software", co.software), ("All other industries", ~co.software)]:
    for b in REV:
        x = co[m & co.bin.eq(b)]
        Bp.append(dict(group=grp, bin=b, companies=len(x), with_clause=int(x.clause.sum()), pct=round(100 * x.clause.mean(), 2)))
Bp = pd.DataFrame(Bp)

# ---------------------------------------------------------------- figure
BLUE, BLUE_LT, ORANGE = "#2a78d6", "#a9c9f0", "#eb6834"
INK, INK2, GRID, SEP = "#000000", "#52514e", "#e6e5e0", "#b8b6b0"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "text.color": INK})
SHOW = {"Under $10M": "Under\n\\$10M", "$10M–$100M": "\\$10M–\n\\$100M", "$100M–$1B": "\\$100M–\n\\$1B",
        "$1B–$10B": "\\$1B–\n\\$10B", "Over $10B": "Over\n\\$10B"}             # matplotlib reads $...$ as math
Y = [4.0, 3.1, 2.05, 0.6, -0.6]                                           # dumbbell rows (3-line labels get more room), gap before the pools
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.8, 3.45), gridspec_kw={"width_ratios": [1, 1.4]})

for y, (_, r) in zip(Y, A.iterrows()):
    axA.plot([r.pct_2019, r.pct_2026], [y, y], color=BLUE_LT, lw=2.2, solid_capstyle="round", zorder=2)
    axA.scatter(r.pct_2019, y, s=36, color=BLUE_LT, edgecolor=BLUE, linewidth=1.1, zorder=3)
    axA.scatter(r.pct_2026, y, s=36, color=BLUE, edgecolor=BLUE, linewidth=1.1, zorder=4)
    axA.annotate(f"{r.change_pp:+.1f}{r.stars}", (max(r.pct_2019, r.pct_2026), y), xytext=(5, 0), textcoords="offset points",
                 ha="left", va="center", color=INK)
axA.axhline((Y[2] + Y[3]) / 2, color=SEP, lw=0.8, ls=(0, (4, 3)), zorder=1)
axA.set_yticks(Y); axA.set_yticklabels([])
NAMES = {"Interactive Media & Services": "Interactive Media\n& Services", "Matched, excl. software": "Matched,\nexcl. software",
         "All product terms": "All product\nterms"}
LH = plt.rcParams["font.size"] * 1.2                                    
for y, g, n in zip(Y, A.group, A.documents):
    name = NAMES.get(g, g); k = name.count("\n") + 1                    
    for txt, dy, wt in [(name, LH / 2, "bold"), (f"(n = {n:,})", -k * LH / 2, "normal")]:
        axA.annotate(txt, (0, y), xycoords=("axes fraction", "data"), xytext=(-6, dy), textcoords="offset points",
                     ha="right", va="center", fontweight=wt, color=INK)
axA.tick_params(axis="y", length=0)
axA.set_ylim(min(Y) - 0.65, max(Y) + 0.45); axA.set_xlim(0, 26); axA.set_xticks(range(0, 26, 5))
axA.set_xlabel("Product-term documents with any clause (%)")
axA.xaxis.grid(True, color=GRID, lw=0.6); axA.set_axisbelow(True)
axA.legend(handles=[Line2D([], [], ls="", marker="o", ms=6.2, mfc=BLUE_LT, mec=BLUE, mew=1.1, label="2019"),
                    Line2D([], [], ls="", marker="o", ms=6.2, mfc=BLUE, mec=BLUE, mew=1.1, label="2026")],
           frameon=False, loc="lower right", handletextpad=0.3)
axA.set_title("A. Change by industry, 2019–2026", loc="left", fontsize=10.5, fontweight="bold", color=INK)

x = np.arange(len(REV)); w = 0.34
for off, (grp, col) in zip((-w / 2 - 0.06, w / 2 + 0.06), [("Software", BLUE), ("All other industries", ORANGE)]):
    s = Bp[Bp.group.eq(grp)].set_index("bin").loc[REV]
    axB.bar(x + off, s.pct, w, color=col, label=grp, zorder=3)
    for xi, (p, k, n) in zip(x + off, zip(s.pct, s.with_clause, s.companies)):
        axB.annotate(f"{k}/{n}", (xi, p), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, color=INK)
        axB.annotate(f"{p:.0f}%", (xi, p), xytext=(0, 12), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.4, color=INK, fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels([SHOW[b] for b in REV])
axB.set_xlabel("Annual revenue (S&P Capital IQ)"); axB.set_ylabel("Companies with any clause (%)")
axB.set_ylim(0, max(50, Bp.pct.max() + 12))
axB.yaxis.grid(True, color=GRID, lw=0.6); axB.set_axisbelow(True)
axB.legend(frameon=False, loc="upper left")
axB.set_title("B. By company revenue", loc="left", fontsize=10.5, fontweight="bold", color=INK)

fig.tight_layout(w_pad=3.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure_3_industry_revenue.{ext}"), dpi=300, bbox_inches="tight")
plt.show()
