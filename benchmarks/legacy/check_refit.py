"""
Quick reproducibility check: refit the configuration selected for a stored
outer fold and compare its test C-index and IBS with results/<dataset>.json.
This skips the inner search (the expensive part), so it takes seconds to minutes.

Usage: python check_refit.py <dataset> <method> <seed> <fold> [<seed> <fold> ...]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import json
import os
import sys

import numpy as np
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import run_benchmark as RB


def main():
    ds, method, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if method.endswith("(library defaults)"):
        RB.LIBRARY_DEFAULTS = True
    base = method.replace(" (library defaults)", "")
    res = json.load(open(os.path.join(RB.HERE, "..", "results", f"{ds}.json")))
    X, dur, ev = RB.load(ds)
    for seed, fold in zip(map(int, rest[::2]), map(int, rest[1::2])):
        tr, te = list(KFold(5, shuffle=True, random_state=seed).split(X))[fold]
        sc = StandardScaler().fit(X.values[tr])
        Xtr, Xte = sc.transform(X.values[tr]), sc.transform(X.values[te])
        ttr, etr, tte, ete = dur[tr], ev[tr], dur[te], ev[te]
        grid = np.linspace(tte.min(), tte.max(), 100)
        stored = res[f"{method}|{seed}|{fold}"]
        risk, surv = RB.FITTERS[base](stored["config"], Xtr, ttr, etr, Xte, grid)
        c = RB.safe_c(tte, risk, ete)
        ibs = EvalSurv(surv, tte, ete, censor_surv="km").integrated_brier_score(grid)
        print(
            f"{ds} {method} seed {seed} fold {fold}: C {c:.4f} (stored {stored['c']:.4f}), "
            f"IBS {ibs:.4f} (stored {stored['ibs']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
