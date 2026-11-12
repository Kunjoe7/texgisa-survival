"""
Tables from results/<dataset>.json: mean C-index and IBS over the outer test
folds, and paired fold-by-fold comparisons of TexGISa with each baseline.

Usage: python summarize.py [--markdown results/summary.md]
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["metabric", "support", "flchain", "gbsg"]
METHODS = ["TexGISa", "Cox", "CoxTime", "DeepSurv", "DeepHit"]
LIB = ["Cox (library defaults)", "DeepSurv (library defaults)"]


def load(ds):
    return json.load(open(os.path.join(HERE, "results", f"{ds}.json")))


def folds(res, methods):
    """(seed, fold) pairs present for every method."""
    keys = {tuple(k.split("|")[1:]) for k in res if k.split("|")[0] == methods[0]}
    return sorted(k for k in keys if all(f"{m}|{k[0]}|{k[1]}" in res for m in methods))


def values(res, m, keys, metric):
    return np.array([res[f"{m}|{s}|{f}"][metric] for s, f in keys])


def tables():
    lines = []
    for metric, name, better in (("c", "C-index", max), ("ibs", "Integrated Brier score", min)):
        lines += [f"### {name} (mean over outer test folds; best in bold)", ""]
        lines += ["| Dataset | folds | " + " | ".join(METHODS) + " |"]
        lines += ["|---|---|" + "---|" * len(METHODS)]
        for ds in DATASETS:
            res = load(ds)
            keys = folds(res, METHODS)
            means = {m: values(res, m, keys, metric).mean() for m in METHODS}
            best = better(means.values())
            cells = [
                f"**{v:.3f}**" if round(v, 3) == round(best, 3) else f"{v:.3f}"
                for v in means.values()
            ]
            lines += [f"| {ds.upper()} | {len(keys)} | " + " | ".join(cells) + " |"]
        lines += [""]

    lines += [
        "### TexGISa minus baseline, C-index (mean difference; folds where TexGISa is higher)"
    ]
    lines += ["", "| Dataset | " + " | ".join(METHODS[1:]) + " |", "|---|" + "---|" * 4]
    for ds in DATASETS:
        res = load(ds)
        keys = folds(res, METHODS)
        t = values(res, "TexGISa", keys, "c")
        cells = []
        for m in METHODS[1:]:
            d = t - values(res, m, keys, "c")
            cells.append(f"{d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})")
        lines += [f"| {ds.upper()} | " + " | ".join(cells) + " |"]
    lines += [""]

    lines += ["### Baselines with the libraries' default settings (C-index / IBS)", ""]
    lines += ["Folds where the library-default runs exist (outer seeds 31415 and 27182).", ""]
    lines += [
        "| Dataset | folds | TexGISa | Cox | Cox (library defaults) | DeepSurv | "
        "DeepSurv (library defaults) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        res = load(ds)
        ms = ["TexGISa", "Cox", LIB[0], "DeepSurv", LIB[1]]
        keys = folds(res, ms)
        cells = [
            f"{values(res, m, keys, 'c').mean():.3f} / {values(res, m, keys, 'ibs').mean():.3f}"
            for m in ms
        ]
        lines += [f"| {ds.upper()} | {len(keys)} | " + " | ".join(cells) + " |"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markdown", default=None)
    a = p.parse_args()
    text = tables()
    print(text)
    if a.markdown:
        open(a.markdown, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
