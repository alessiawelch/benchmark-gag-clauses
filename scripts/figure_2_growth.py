#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import CUT, documents, year_status

LABELS = ["Core DeWitt Clause", "Competitive Ban", "Confidential Information Restriction", "Publicity Gag Clause", "DeWitt Embrace Clause"]
SHORTNAME = {"Core DeWitt Clause": "Bench-\nmarking\n& Results", "Competitive Ban": "Compet-\nitive\nAnalysis",
             "Confidential Information Restriction": "Evaluation\nConfiden-\ntiality", "Publicity Gag Clause": "Service\nPublicity",
             "DeWitt Embrace Clause": "Condi-\ntional\nBench."}
YEARS = list(range(2019, 2027))
XS = YEARS[:-1] + [2026 + (CUT.dayofyear - 1) / 365.25]       # the last point sits where the data end
BLUE, ORANGE, BLUE_LT, ORANGE_LT = "#2a78d6", "#eb6834", "#a9c9f0", "#f6bc9e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
STAR = {"Snyk": "#1baf7a", "Resemble AI": "#8a1c7c", "Academia.edu": "#2b2b2b"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


docs = documents()
scope = dict(zip(docs.document_id, docs.scope))
per = year_status(scope)                                  # clauses in each document's latest version that year
per = per[per.index.get_level_values("year").isin(YEARS)].copy()
per["Any"] = per[LABELS].any(axis=1)

# panel: product terms seen in both 2019 and 2026
ends = per.reset_index().groupby("document_id").year.agg(lambda y: {2019, 2026} <= set(y))
panel = [d for d in sorted(ends[ends].index) if scope[d] in ("Service", "Mixed")]
K = LABELS + ["Any"]
grid = per.loc[panel].reset_index().set_index(["document_id", "year"])[K].unstack("year")

# Panel A
m = grid["Any"].reindex(columns=YEARS).astype("float").ffill(axis=1)
pct, n = 100 * m.mean(), m.sum().astype(int)

# Panel B
any0, any1 = grid[("Any", 2019)].astype(bool), grid[("Any", 2026)].astype(bool)
B = {}
for k in LABELS:
    a0, a1 = grid[(k, 2019)].astype(bool), grid[(k, 2026)].astype(bool)
    B[k] = dict(added=int((~a0 & a1).sum()), added_new=int((~a0 & a1 & ~any0).sum()), added_swap=int((~a0 & a1 & any0).sum()),
                removed=int((a0 & ~a1).sum()), removed_all=int((a0 & ~a1 & ~any1).sum()), removed_swap=int((a0 & ~a1 & any1).sum()))

# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"width_ratios": [1, 1.45]})
axA.plot(XS, pct.values, color=BLUE, lw=2, marker="o", ms=5.5, mec="white", mew=1.2, zorder=3)
for y, dx, ha in ((2019, -4, "left"), (2026, 0, "center")):
    axA.annotate((f"{CUT:%b %Y}: " + rf"$\bf{{{pct[y]:.1f}\%}}$" if y == 2026 else f"{pct[y]:.1f}%") + f"\n({n[y]:,} docs)", (XS[YEARS.index(y)], pct[y]), xytext=(dx, 9), textcoords="offset points", ha=ha, va="bottom",
                 fontsize=7.5, color=INK)
# first archived snapshot containing the example companies' clauses 
for xe, lab, off, col in [(2021 + 183 / 365.25, "Snyk\n(Jul 2021)", (0, -34), STAR["Snyk"]),
                          (2024 + 123 / 365.25, "Resemble AI\n(May 2024)", (-6, -52), STAR["Resemble AI"]),
                          (2026.0, "Academia.edu\n(Jan 2026)", (-8, -22), STAR["Academia.edu"])]:
    ye = float(np.interp(xe, XS, pct.values))
    axA.plot([xe], [ye], marker="*", ms=7.5, color=col, mec="white", mew=0.5, zorder=4)
    axA.annotate(lab, (xe, ye), xytext=off, textcoords="offset points", ha="center", va="top", fontsize=6, color=col,
                 fontweight="bold", linespacing=1.05,
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, shrinkA=1, shrinkB=4, mutation_scale=6))
axA.set_xticks(YEARS + [2027]); axA.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS + [2027]], fontsize=7.5)
axA.set_xlim(2018.6, 2027.3); axA.set_ylim(0, pct.max() * 1.35)
axA.set_ylabel("Product-term documents with any clause,\n2019 to 2026 (%)")
axA.yaxis.grid(True, color=GRID, lw=0.6); axA.set_axisbelow(True)
axA.set_title("A. Overall growth", loc="left", fontsize=9, fontweight="bold", color=INK)

x = np.arange(len(LABELS)); w = 0.36
get = lambda f: np.array([B[l][f] for l in LABELS])
xa, xr = x - w / 2 - 0.01, x + w / 2 + 0.01
axB.bar(xa, get("added_new"), w, color=BLUE, label="Added (no clause in 2019)", zorder=3)
axB.bar(xa, get("added_swap"), w, bottom=get("added_new"), color=BLUE_LT, label="Added (swapped in for another clause)", zorder=3)
axB.bar(xr, get("removed_all"), w, color=ORANGE, label="Removed (no clause in 2026)", zorder=3)
axB.bar(xr, get("removed_swap"), w, bottom=get("removed_all"), color=ORANGE_LT, label="Removed (swapped for another clause)", zorder=3)
top = max(get("added").max(), get("removed").max())
for xs, vals in ((xa, get("added")), (xr, get("removed"))):
    for xi, v in zip(xs, vals): axB.text(xi, v + top * 0.015, f"{v}", ha="center", va="bottom", fontsize=7.5, color=INK)
axB.set_xticks(x); axB.set_xticklabels([SHORTNAME[l] for l in LABELS], fontsize=6.6)
axB.set_ylim(0, top * 1.15); axB.set_ylabel("Product-term documents adding or removing\nthe clause, 2019 to 2026")
axB.yaxis.grid(True, color=GRID, lw=0.6); axB.set_axisbelow(True)
axB.legend(frameon=False, fontsize=6.6, loc="upper right")
axB.set_title("B. Changes within each clause", loc="left", fontsize=9, fontweight="bold", color=INK)

fig.suptitle("Benchmark Gag Clauses", x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK, y=0.995)
fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.985))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure_2_growth.{ext}"), dpi=300, bbox_inches="tight")
plt.show()
