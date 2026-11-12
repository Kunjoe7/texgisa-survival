"""Corrected nested-CV benchmark; results are separate from the archived table.

Preprocessing fits on optimizer-training observations only, including each
candidate's early-stopping split. The evaluation grid uses training durations.
The search spaces and budgets are preserved. Full runs may take hours.
Archived table reproduction is in legacy/; never overwrite those records.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.integrate
import torch

warnings.filterwarnings("ignore")
if not hasattr(scipy.integrate, "simps"):  # pycox 0.3 predates SciPy's rename to simpson
    scipy.integrate.simps = lambda y, x: scipy.integrate.simpson(y, x=x)

import torchtuples as tt  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import concordance_index as ll_cindex  # noqa: E402
from pycox import datasets as pds  # noqa: E402
from pycox.evaluation import EvalSurv  # noqa: E402
from pycox.models import CoxPH, CoxTime, DeepHitSingle  # noqa: E402
from pycox.models.cox_time import MLPVanillaCoxTime  # noqa: E402
from pycox.models.loss import cox_ph_loss  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline  # noqa: E402

from texgisa_survival import TexGISa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = json.load(open(os.path.join(HERE, "configs.json")))
LIBRARY_DEFAULTS = False


# ---------------------------------------------------------------- data
def load(name):
    """Features, durations and events; FLCHAIN from scikit-survival, others from pycox."""
    if name == "flchain":
        from sksurv.datasets import load_flchain

        X, y = load_flchain()
        X = X.drop(columns=["chapter"])  # cause of death: leaks the outcome
        X["creatinine"] = X["creatinine"].astype(float)
        dur, ev = y["futime"].astype(float), y["death"].astype(int)
    else:
        df = getattr(pds, name).read_df()
        dur, ev = df.pop("duration").values.astype(float), df.pop("event").values.astype(int)
        X = df
    return X.reset_index(drop=True), np.asarray(dur), np.asarray(ev)


def preprocess(Xtr, Xte, fit_rows=None):
    """Learn imputations, categories and scale from actual fitting rows only."""
    Xtr, Xte = pd.DataFrame(Xtr), pd.DataFrame(Xte)
    numeric = Xtr.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = [col for col in Xtr if col not in numeric]
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent'),
             OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)), categorical)
    ])
    transform.fit(Xtr if fit_rows is None else Xtr.iloc[fit_rows])
    return transform.transform(Xtr).astype('float32'), transform.transform(Xte).astype('float32')


def es_split(n, seed=0):
    """Random 80/20 split for early stopping."""
    idx = np.random.default_rng(seed).permutation(n)
    k = n // 5
    return idx[k:], idx[:k]


def safe_c(t, risk, e):
    risk = np.asarray(risk, dtype=float)
    if not np.all(np.isfinite(risk)):
        return -1.0
    return ll_cindex(t, -risk, e)


# ---------------------------------------------------------------- fitters
# Each returns (risk scores on Xte, survival DataFrame on grid, or None if grid is None)
def fit_texgisa(c, Xtr, ttr, etr, Xte, grid):
    a, b = es_split(len(ttr))
    Xtr, Xte = preprocess(Xtr, Xte, fit_rows=a)
    m = TexGISa(
        hidden_layers=c["hidden"],
        num_time_bins=c["bins"],
        dropout=c["dropout"],
        activation=c["activation"],
        time_binning=c["binning"],
        risk_score="expected_time",
        weight_decay=c["wd"],
        architecture=c["arch"],
        norm=c["norm"],
        es_metric=c["es_metric"],
        likelihood=c["likelihood"],
        lambda_cox=c["lambda_cox"],
        calibrate=c["calibrate"],
        ema_decay=c["ema_decay"],
        input_noise=c["input_noise"],
        input_gate=c["input_gate"],
        random_state=42,
        device="cpu",
        verbose=0,
    )
    m.fit(
        Xtr[a],
        ttr[a],
        etr[a],
        validation_data=(Xtr[b], ttr[b], etr[b]),
        epochs=300,
        batch_size=c["batch"],
        learning_rate=c["lr"],
        early_stopping=True,
        patience=15,
        verbose=0,
    )
    surv = None
    if grid is not None:
        surv = pd.DataFrame(m.predict_survival(Xte, times=grid).T, index=grid)
    return m.predict_risk(Xte), surv


class SafeCoxPHLoss(torch.nn.Module):
    """pycox's Cox partial likelihood, returning 0 for a mini-batch without events."""

    def forward(self, log_h, durations, events):
        if events.sum() < 1:
            return log_h.sum() * 0.0
        return cox_ph_loss(log_h, durations, events)


def _net(d, out, c):
    return tt.practical.MLPVanilla(d, c["hidden"], out, batch_norm=True, dropout=c["dropout"])


def _opt(c):
    if c["wd"] > 0:
        return tt.optim.AdamW(lr=c["lr"], decoupled_weight_decay=c["wd"])
    return tt.optim.Adam(c["lr"])


def fit_deepsurv(c, Xtr, ttr, etr, Xte, grid):
    torch.manual_seed(42)
    np.random.seed(42)
    a, b = es_split(len(ttr))
    Xtr, Xte = preprocess(Xtr, Xte, fit_rows=a)
    Xtr, Xte = Xtr.astype("float32"), Xte.astype("float32")

    def y(i):
        return ttr[i].astype("float32"), etr[i].astype("float32")

    if LIBRARY_DEFAULTS:
        m = CoxPH(_net(Xtr.shape[1], 1, c), _opt(c))
    else:
        m = CoxPH(_net(Xtr.shape[1], 1, c), _opt(c), loss=SafeCoxPHLoss())
    m.fit(
        Xtr[a],
        y(a),
        batch_size=c["batch"],
        epochs=300,
        verbose=False,
        callbacks=[tt.callbacks.EarlyStopping(patience=15)],
        val_data=(Xtr[b], y(b)),
    )
    m.compute_baseline_hazards()
    return m.predict(Xte).ravel(), (None if grid is None else m.predict_surv_df(Xte))


def fit_deephit(c, Xtr, ttr, etr, Xte, grid):
    torch.manual_seed(42)
    np.random.seed(42)
    a, b = es_split(len(ttr))
    Xtr, Xte = preprocess(Xtr, Xte, fit_rows=a)
    Xtr, Xte = Xtr.astype("float32"), Xte.astype("float32")
    lab = DeepHitSingle.label_transform(c["bins"])
    ya, yb = lab.fit_transform(ttr[a], etr[a]), lab.transform(ttr[b], etr[b])
    m = DeepHitSingle(
        _net(Xtr.shape[1], lab.out_features, c),
        _opt(c),
        alpha=c["alpha"],
        sigma=c["sigma"],
        duration_index=lab.cuts,
    )
    m.fit(
        Xtr[a],
        ya,
        batch_size=c["batch"],
        epochs=300,
        verbose=False,
        callbacks=[tt.callbacks.EarlyStopping(patience=15)],
        val_data=(Xtr[b], yb),
    )
    surv = m.interpolate(10).predict_surv_df(Xte)
    risk = -scipy.integrate.trapezoid(surv.values, surv.index.values, axis=0)  # minus expected time
    return risk, (None if grid is None else surv)


def fit_coxtime(c, Xtr, ttr, etr, Xte, grid):
    torch.manual_seed(42)
    np.random.seed(42)
    a, b = es_split(len(ttr))
    Xtr, Xte = preprocess(Xtr, Xte, fit_rows=a)
    Xtr, Xte = Xtr.astype("float32"), Xte.astype("float32")
    lab = CoxTime.label_transform()
    ya = lab.fit_transform(ttr[a].astype("float32"), etr[a].astype("float32"))
    yb = lab.transform(ttr[b].astype("float32"), etr[b].astype("float32"))
    net = MLPVanillaCoxTime(Xtr.shape[1], c["hidden"], batch_norm=True, dropout=c["dropout"])
    m = CoxTime(net, _opt(c), labtrans=lab)
    m.fit(
        Xtr[a],
        ya,
        batch_size=c["batch"],
        epochs=300,
        verbose=False,
        callbacks=[tt.callbacks.EarlyStopping(patience=15)],
        val_data=(Xtr[b], yb),
    )
    m.compute_baseline_hazards()
    surv = m.predict_surv_df(Xte)
    risk = -scipy.integrate.trapezoid(surv.values, surv.index.values, axis=0)  # minus expected time
    return risk, (None if grid is None else surv)


def fit_cox(c, Xtr, ttr, etr, Xte, grid):
    Xtr, Xte = preprocess(Xtr, Xte)
    df = pd.DataFrame(Xtr)
    df["T"], df["E"] = ttr, etr
    m = CoxPHFitter(penalizer=c["pen"], l1_ratio=c["l1"])
    if LIBRARY_DEFAULTS:
        m.fit(df, "T", "E")
    else:
        m.fit(df, "T", "E", fit_options={"step_size": 0.25, "max_steps": 2000})
    risk = m.predict_partial_hazard(pd.DataFrame(Xte)).values
    surv = None if grid is None else m.predict_survival_function(pd.DataFrame(Xte), times=grid)
    return risk, surv


FITTERS = {
    "TexGISa": fit_texgisa,
    "Cox": fit_cox,
    "CoxTime": fit_coxtime,
    "DeepSurv": fit_deepsurv,
    "DeepHit": fit_deephit,
}


# ---------------------------------------------------------------- nested CV
def run(dataset, seeds, methods, out):
    res = json.load(open(out)) if os.path.exists(out) else {}
    X, dur, ev = load(dataset)
    suffix = " (library defaults)" if LIBRARY_DEFAULTS else ""
    for seed in seeds:
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ttr, etr, tte, ete = dur[tr], ev[tr], dur[te], ev[te]
            itr, iva = train_test_split(
                np.arange(len(ttr)), test_size=0.2, random_state=0, stratify=etr
            )
            grid = np.linspace(*np.percentile(ttr, [10, 80]), 100)
            for meth in methods:
                key = f"{meth}{suffix}|{seed}|{f}"
                if key in res:
                    continue
                t0 = time.time()
                space, fitter = CONFIGS[meth], FITTERS[meth]
                inner = []
                for c in space:
                    try:
                        r, _ = fitter(c, Xtr.iloc[itr], ttr[itr], etr[itr], Xtr.iloc[iva], None)
                        inner.append(safe_c(ttr[iva], r, etr[iva]))
                    except Exception:  # e.g. a Cox fit that does not converge
                        inner.append(-1.0)
                if max(inner) < 0:
                    raise RuntimeError(f"All candidates failed for {meth}; inspect data and fits")
                best = int(np.argmax(inner))
                risk, surv = fitter(space[best], Xtr, ttr, etr, Xte, grid)
                c_te = safe_c(tte, risk, ete)
                ibs = EvalSurv(surv, tte, ete, censor_surv="km").integrated_brier_score(grid)
                res[key] = {
                    "protocol": "split-local-preprocessing-2.1.1",
                    "grid": grid.tolist(),
                    "n_failed_candidates": sum(v < 0 for v in inner),
                    "c": float(c_te),
                    "ibs": float(ibs),
                    "inner_best": float(inner[best]),
                    "n_configs": len(space),
                    "config": space[best],
                    "sec": time.time() - t0,
                }
                json.dump(res, open(out, "w"), indent=1)
                print(
                    f"{dataset} seed {seed} fold {f} {meth + suffix:30s} C={c_te:.4f} "
                    f"IBS={ibs:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )


def main():
    global LIBRARY_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", choices=["metabric", "support", "flchain", "gbsg"])
    p.add_argument("--seeds", type=int, nargs="+", default=[31415, 27182, 16180, 14142])
    p.add_argument("--methods", nargs="+", default=list(FITTERS), choices=list(FITTERS))
    p.add_argument("--library-defaults", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None, help="results file (default results/<dataset>.json)")
    a = p.parse_args()
    LIBRARY_DEFAULTS = a.library_defaults
    torch.set_num_threads(a.threads)
    out = a.out or os.path.join(HERE, "corrected_results", f"{a.dataset}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run(a.dataset, a.seeds, a.methods, out)


if __name__ == "__main__":
    main()
