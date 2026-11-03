#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "data", "validation")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

LABELS = ["Core", "CompBan", "ConfInfo", "Publicity", "Embrace"]           
N_PERM, SEED = 10_000, 20261025

df = pd.read_csv(os.path.join(VAL, "validation_labels.csv"))
MODELS = [c for c in df.columns if c not in ("document", "gold")]      


def to_matrix(col):
    rows = []
    for v in col.fillna("(none)"):
        s = {p.strip() for p in str(v).split(";")} - {"(none)", ""}
        assert s <= set(LABELS), f"unknown label(s) {s - set(LABELS)}"
        rows.append([int(l in s) for l in LABELS])
    return np.array(rows, dtype=np.int8)


G = to_matrix(df.gold)
P = np.stack([to_matrix(df[c]) for c in MODELS])                         
N, K = G.shape[0], len(MODELS)


def micro_f1(X):
    tp = (G & X).sum(); fp = ((1 - G) & X).sum(); fn = (G & (1 - X)).sum()
    return 2 * tp / (2 * tp + fp + fn)


def macro_f1(X, drop_embrace=False):
    tp = (G & X).sum(0); fp = ((1 - G) & X).sum(0); fn = (G & (1 - X)).sum(0); den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)              
    return (f[:-1] if drop_embrace else f).mean()


SCORES = {"micro_f1": micro_f1, "macro_f1": macro_f1, "macro_f1_4": lambda X: macro_f1(X, True)}
exact = (P == G[None]).all(2)                                          
t = pd.DataFrame({"model": MODELS, "exact_match": exact.mean(1),
                  **{k: [f(P[i]) for i in range(K)] for k, f in SCORES.items()}})

# exact match: Cochran's Q (statsmodels)
p = {"exact_match": cochrans_q(exact.T.astype(int)).pvalue}

# F1 scores: paired permutation test (scipy).
BITS = 1 << np.arange(len(LABELS))
codes = [(P[i] * BITS).sum(1) for i in range(K)]                         
decode = lambda c: ((np.asarray(c)[:, None] & BITS) > 0).astype(np.int8)
warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)   
for k, f in SCORES.items():
    res = permutation_test(codes, lambda *s, f=f: np.var([f(decode(x)) for x in s]), permutation_type="samples",
                           alternative="greater", n_resamples=N_PERM, random_state=SEED)
    p[k] = res.pvalue
p_bh = dict(zip(p, multipletests(list(p.values()), method="fdr_bh")[1]))

tests = pd.DataFrame([{"model": "Test across models", **{k: ("Cochran's Q" if k == "exact_match" else "Permutation") for k in p}},
                      {"model": "p (raw)", **p}, {"model": "BH-adjusted p", **p_bh}])
out = pd.concat([t, tests], ignore_index=True)
out.to_csv(os.path.join(OUT, "table_1_model_performance.csv"), index=False)

# print
fmt = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
print(f"{'Model':46s} {'Exact match':>12s} {'Micro-F1':>9s} {'Macro-F1':>9s} {'Macro-F1 (4)':>13s}")
for r in t.itertuples():
    print(f"{r.model:46s} {100 * r.exact_match:11.1f}% {r.micro_f1:9.3f} {r.macro_f1:9.3f} {r.macro_f1_4:13.3f}")
print(f"{'Test across models':46s} {'Cochran Q':>12s} {'Permutation':>9s} {'Permutation':>9s} {'Permutation':>13s}")
print(f"{'BH-adjusted p':46s} {fmt(p_bh['exact_match']):>12s} {fmt(p_bh['micro_f1']):>9s} {fmt(p_bh['macro_f1']):>9s} {fmt(p_bh['macro_f1_4']):>13s}")
print(f"\nwrote {os.path.join(OUT, 'table_1_model_performance.csv')}")
