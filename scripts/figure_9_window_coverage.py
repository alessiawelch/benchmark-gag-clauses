#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import documents, observed_years

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
STARTS, ENDS = list(range(2015, 2026)), list(range(2016, 2027))

# ---------------------------------------------------------------- years each product-term document was observed
docs = documents()
obs = observed_years(set(docs[docs.product_terms].document_id))
years = obs.groupby("year").document_id.agg(set)
g = pd.DataFrame([[len(years[s] & years[e]) if e > s else np.nan for e in ENDS] for s in STARTS], index=STARTS, columns=ENDS)

# ---------------------------------------------------------------- figure
INK, INK2 = "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("blues", ["#dbe8f8", "#2a78d6", "#123f78"])
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.0, 5.4))
v = g.values.astype(float)
lo, hi = np.nanmin(v) * 0.6, np.nanmax(v)
ax.imshow(np.ma.masked_invalid(v), cmap=CMAP, vmin=lo, vmax=hi, aspect="equal")
for i in range(len(STARTS)):
    for j in range(len(ENDS)):
        if np.isnan(v[i, j]): continue
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.2))
        light = v[i, j] < lo + 0.40 * (hi - lo)                          # dark text on light cells
        ax.text(j, i, f"{v[i, j] / 1000:.1f}k", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK if light else "white")
i, j = STARTS.index(2019), ENDS.index(2026)
ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2.2, zorder=5))
ax.set_xticks(range(len(ENDS))); ax.set_xticklabels([f"'{str(e)[2:]}" for e in ENDS], fontsize=8, color=INK2)
ax.set_yticks(range(len(STARTS))); ax.set_yticklabels([str(s) for s in STARTS], fontsize=8, color=INK2)
ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
ax.set_xlabel("End year", fontsize=8.5, color=INK2, labelpad=6); ax.set_ylabel("Start year", fontsize=8.5, color=INK2)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Product-term documents observed in both the start and end year", x=0.02, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure_9_window_coverage.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.show()
