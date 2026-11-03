#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PAPER_NAME, CUT, documents, history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
LABELS = ["Core DeWitt Clause", "Competitive Ban", "Confidential Information Restriction", "Publicity Gag Clause", "DeWitt Embrace Clause"]
SLUG = dict(zip(LABELS, ["benchmarking", "competitive", "confidentiality", "publicity", "conditional"]))
NUM = dict(zip(LABELS, [4, 5, 6, 7, 8]))                     # figure numbers in the paper
XMIN, XTODAY = 2007.6, 2027.5
BLUE, ORANGE, PLUM, SPAN, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a1c7c", "#cfcdc6", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

D = documents()
D = D[D.product_terms]
docs = set(D.document_id)


def yfrac(t):
    t = pd.Timestamp(t)
    return t.year + (t.dayofyear - 1) / 365.25


# ---------------------------------------------------------------- observations: clause -> document -> {fractional year: present}
obs = defaultdict(lambda: defaultdict(dict))
h = history()
h = h[h.document_id.isin(docs)]
wb = h[h.source == "wayback"]
tt = h[(h.source == "termtrove") & ~h.latest.astype(bool)]
for part in (wb, tt):
    for l in LABELS:
        for d, t, s in zip(part.document_id, part.date, part[l]):
            if pd.Timestamp(t) > CUT: continue
            x = max(round(yfrac(t), 3), XMIN + 0.05); o = obs[l][int(d)]; o[x] = o.get(x, False) or bool(s)
obs = {l: dict(v) for l, v in obs.items()}
today = {l: dict(zip(D[D.version_id.notna()].document_id.astype(int), D[D.version_id.notna()][l].astype(bool))) for l in LABELS}


def build(l):
    A, B = [], []
    for d in set(obs[l]) | {d for d, v in today[l].items() if v}:
        o = obs[l].get(d, {}); td = today[l].get(d)
        if not (any(o.values()) or td): continue
        red = sorted(t for t, p in o.items() if p)
        r = dict(d=d, o=o, td=td, first=red[0] if red else XTODAY, last=red[-1] if red else XTODAY, n=len(red),
                 oth=any(today[k].get(d) for k in LABELS if k != l))
        (A if td else B).append(r)
    A.sort(key=lambda r: (r["first"], -r["n"])); B.sort(key=lambda r: (not r["oth"], r["last"], -r["n"]))
    return A, B


# ---------------------------------------------------------------- figure
W, H = 7.0, 8.5                                    
Y0, Y1 = 2008, 2027; NB = (Y1 - Y0) * 4           
CMAP = ListedColormap(["white", SPAN, BLUE, ORANGE, PLUM])
RANK = {0: 0, 1: 1, 2: 2, 4: 3, 3: 4}               
qb = lambda t: int(np.clip(int((t - Y0) * 4), 0, NB - 1))


def grid(l, rows, other_colour):
    """codes: 0 blank, 1 between observations, 2 clause absent, 3 clause present, 4 clause absent but another of the five present"""
    g = np.zeros((len(rows), NB + 3), dtype=int)
    others = [k for k in LABELS if k != l]
    for i, r in enumerate(rows):
        ts = sorted(r["o"])
        if ts: g[i, qb(ts[0]):qb(ts[-1]) + 1] = 1
        for t in ts:
            code = 3 if r["o"][t] else (4 if other_colour and any(obs[k].get(r["d"], {}).get(t) for k in others) else 2)
            if RANK[code] > RANK[g[i, qb(t)]]: g[i, qb(t)] = code
        if r["td"] is not None:
            g[i, NB + 1:NB + 3] = 3 if r["td"] else (4 if other_colour and any(today[k].get(r["d"]) for k in others) else 2)
    return g


def panel(ax, l, rows, title, other_colour):
    n = len(rows)
    ax.imshow(grid(l, rows, other_colour), aspect="auto", cmap=CMAP, vmin=0, vmax=4, interpolation="nearest", extent=(0, NB + 3, n, 0))
    for y in range(Y0 + 2, Y1, 2): ax.axvline((y - Y0) * 4, color=GRID, lw=0.4, zorder=0, alpha=0.6)
    ax.axvline(NB + 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(0, NB + 3); ax.set_ylim(n, 0)
    ax.set_xticks([(y - Y0) * 4 for y in range(Y0, Y1, 2)] + [NB + 2])
    ax.set_xticklabels([str(y) for y in range(Y0, Y1, 2)] + ["today"], fontsize=7, color=INK2); ax.tick_params(axis="x", length=0, pad=2)
    step = 100 if n > 400 else (50 if n > 100 else (10 if n > 30 else 5)); yt = list(range(0, n + 1, step))
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:,}" for v in yt], fontsize=6.5, color=INK2); ax.tick_params(axis="y", length=2, color=GRID)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=4)


def draw(l, A, B):
    RA, RB = len(A), len(B)
    top, gap, bot = 0.48, 0.42, 0.66
    avail = H - top - gap - bot
    ha, hb = avail * RA / (RA + RB), avail * RB / (RA + RB)
    if RB and hb < 0.45: hb = 0.45; ha = avail - hb
    fig = plt.figure(figsize=(W, H)); left, right = 0.62, 0.15
    for k, (rows, h, y_in, title, oc) in enumerate([
            (A, ha, top, f"A.  Present today: {RA:,} documents, ordered by when the clause is first seen", False),
            (B, hb, top + ha + gap, f"B.  No longer present today: {RB:,} documents ({sum(r['oth'] for r in B):,} still carry another clause, shown first)", True)]):
        if not rows: continue
        ax = fig.add_axes([left / W, 1 - (y_in + h) / H, (W - left - right) / W, h / H])
        panel(ax, l, rows, title, oc)
        if k == 0: ax.set_ylabel("Documents", fontsize=7.5, color=INK2, labelpad=2)
    fig.text(left / W, 1 - 0.18 / H, PAPER_NAME[l], ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    leg = [mp.Patch(color=ORANGE, label="Clause present"), mp.Patch(color=BLUE, label="Clause not present"),
           mp.Patch(color=PLUM, label="Clause not present, but another clause in the set is present"), mp.Patch(color=SPAN, label="Time between observations")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.04 / H), handlelength=1.1, columnspacing=1.2)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"figure_{NUM[l]}_clause_adoption_{SLUG[l]}.{ext}"), dpi=300)


for l in LABELS:
    A, B = build(l)
    draw(l, A, B)
plt.show()
