# Copyright (c) 2026 DHAI Lab
# Licensed under the MIT License; see LICENSE.

"""GBSG2 expert-feedback study with five fixed outer folds and paired training.

This evaluates conditional directions, prediction and runtime; GBSG2 (686
patients) is a separate dataset from the 2,232-patient GBSG benchmark.
No configuration or rule is chosen using test outcomes. Run from repository root:
python benchmarks/expert_feedback.py --out benchmarks/results/expert_feedback.json
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lifelines.datasets import load_gbsg2
from sklearn.model_selection import KFold, train_test_split
from texgisa_survival import TexGISa, concordance_index, integrated_brier_score
from texgisa_survival.texgi_utils import texgi_time_series


def main(out):
    torch.set_num_threads(2)
    df = load_gbsg2()
    df['horTh'] = (df['horTh'] == 'yes').astype(float)
    df['menostat'] = (df['menostat'] == 'Post').astype(float)
    df['tgrade'] = df['tgrade'].map({'I': 1., 'II': 2., 'III': 3.})
    durations, events = df.pop('time').to_numpy(float), df.pop('cens').to_numpy(int)
    X = df.astype(float)
    protocol = dict(dataset='GBSG2', n=len(X), p=X.shape[1], outer_seed=20261015,
                    folds=5, training_seed=42, inner_seed=0,
                    epochs=300, patience=15, batch_size=128, learning_rate=.003,
                    lambda_expert=5., min_magnitude=.01, ig_steps=20,
                    rules=[dict(feature='pnodes', relation='>', threshold='median', sign=1)],
                    evaluation='all intervals; held-out subjects above training median pnodes',
                    grid='100 points from training 10th to 80th duration percentiles',
                    censoring_weights='Kaplan-Meier estimated on test fold by library metric')
    results = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=20261015).split(X)):
        a, b = train_test_split(tr, test_size=.2, stratify=events[tr], random_state=0)
        grid = np.linspace(*np.percentile(durations[a], [10, 80]), 100)
        for guided in (False, True):
            model = TexGISa(hidden_layers=[256, 128], num_time_bins=20,
                            lambda_expert=5. if guided else 0., ig_steps=20,
                            random_state=42, device='cpu', verbose=0)
            if guided:
                model.add_expert_rule('pnodes', '>', 'median', sign=1,
                                      min_magnitude=.01)
            start = time.perf_counter()
            model.fit(X.iloc[a], durations[a], events[a],
                      validation_data=(X.iloc[b], durations[b], events[b]),
                      epochs=300, patience=15, batch_size=128,
                      learning_rate=.003, verbose=0)
            fit_sec = time.perf_counter() - start
            inner = model.model
            xt = torch.as_tensor(inner.scaler.transform(X.iloc[te].to_numpy('float32')),
                                 dtype=torch.float32)
            inner.model.eval()
            start = time.perf_counter()
            gradients = texgi_time_series(inner.model, xt, inner._reference_baseline(xt),
                                         M=20, multiply_by_inputs=False).detach().numpy()
            explain_sec = time.perf_counter() - start
            feat = list(X.columns).index('pnodes')
            mask = X.iloc[te]['pnodes'].to_numpy() > X.iloc[a]['pnodes'].median()
            g = gradients[:, mask, feat]
            phi = model.explain(X.iloc[te], n_steps=20)
            row = dict(fold=fold, guided=guided, n_test=len(te), n_condition=int(mask.sum()),
                       c=float(concordance_index(durations[te], model.predict_risk(X.iloc[te]), events[te])),
                       ibs=float(integrated_brier_score(durations[te],
                           model.predict_survival(X.iloc[te], times=grid), events[te], eval_times=grid)),
                       sign_violation=float((g < 0).mean()),
                       margin_shortfall=float(np.maximum(.01-g, 0).mean()),
                       attribution_temporal_l1_mean=float(np.abs(np.diff(phi, axis=0)).mean()),
                       fit_sec=fit_sec, explain_sec=explain_sec,
                       mean_path_derivative_by_interval=g.mean(axis=1).tolist())
            results.append(row)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(protocol=protocol, results=results), indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='mean_path_derivative_by_interval'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('benchmarks/results/expert_feedback.json'))
    main(p.parse_args().out)
