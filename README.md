# texgisa-survival

Interpretable and interactive deep survival analysis in Python.

`texgisa-survival` implements **TexGISa**, the deep discrete-time survival framework with
**Time-dependent EXtreme Gradient Integration (TEXGI)** from Qin et al. (ICDM 2025). TEXGI
explains the prediction of every time interval against an adversarially generated, high-risk
baseline, and these explanations act as **attribution priors** during training, so that features
identified by domain experts shape the model.

The package also provides:

- a **prior-finder skill** (`skills/texgisa-prior-finder`) that turns published sources into
  expert priors for a new data set;
- **one interface to 20 established survival models** from `lifelines`, `scikit-survival`,
  `pycox` and original author code, run unchanged for comparison.

## Installation

```bash
pip install texgisa-survival                 # TexGISa and TEXGI
pip install "texgisa-survival[recipes]"      # + SurvivalRecipe (needs scikit-survival)
pip install "texgisa-survival[baselines]"    # + lifelines / scikit-survival / pycox comparisons
```

From source: `pip install ".[recipes,baselines]"`. Python 3.10 or newer is required.

## Quick start

The examples load the public WHAS500 data from `scikit-survival` (installed with `[recipes]`).

### TexGISa with an expert prior and TEXGI explanations

```python
import torch
from sksurv.datasets import load_whas500
from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
from texgisa_survival.models.texgisa_advanced import (
    MySATrainer, build_supervision, make_intervals, texgi_time_series)

X, y = load_whas500()                     # load data, form ten intervals
X = (X.astype(float) - X.astype(float).mean()) / X.astype(float).std()
d = make_intervals(X.assign(duration=y["lenfol"], event=y["fstat"] * 1), n_bins=10)
x = torch.tensor(X.values).float()
lab, msk = map(torch.tensor, build_supervision(d.interval_number, d.event, 10))

prior = {"important_features": ["age", "hr", "sysbp", "chf"]}  # expert list
net = MultiTaskSurvivalNet(x.shape[1], [64, 32], num_time_bins=10)
trainer = MySATrainer(net, lambda_smooth=0.01, lambda_expert=0.1, expert_rules=prior,
                      X_train_ref=x, feat2idx={c: i for i, c in enumerate(X)})
for epoch in range(50):         # interval loss + smoothness and expert priors
    trainer.step(x, lab, msk)

phi = texgi_time_series(net, x[:5], trainer.G, trainer.ref_stats)  # (interval, patient, feature)
```

`MySATrainer` minimizes the interval loss plus `lambda_smooth * Omega_smooth` (temporal
smoothness of the attributions) and `lambda_expert * Omega_expert`, which raises the attribution
norm of each listed feature to at least the average and shrinks the others.

### Default configuration with separate outputs

```python
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_whas500
from texgisa_survival.recipes import SurvivalRecipe

X, y = load_whas500()
X = X.astype(float)
a, b = train_test_split(range(len(X)), test_size=0.25, random_state=271, stratify=y["fstat"])
model = SurvivalRecipe(random_state=271)
model.fit(X.iloc[a], y["lenfol"][a], y["fstat"][a])
risk = model.predict_risk(X.iloc[b])                       # larger = earlier event
survival = model.predict_survival(X.iloc[b], times=[30.0, 180.0, 365.0])  # subjects x times
point_time = model.predict_time(X.iloc[b])                 # positive point estimate
phi = model.explain(X.iloc[b[:2]], integration_steps=20, time_samples=4)  # time x subject x feature
```

Risk, survival curve and point time are separate outputs and are not interchangeable.

## Finding priors for your own data

`skills/texgisa-prior-finder/SKILL.md` is a procedure that a person or an AI assistant follows to
retrieve domain knowledge for a new data set:

1. collect the meaning of the outcome and of every column;
2. for each named feature, search the data set documentation, the original study, clinical or
   domain guidelines and curated knowledge bases, in that order;
3. encode each supported finding as an important-feature list (`Omega_expert`) or a directional
   rule (`TexGISa.add_expert_rule`), and record its source in `priors.json`;
4. never derive priors from outcomes or model rankings, and give anonymous columns no rule;
5. finish with a review by a domain expert.

Lists from clinicians are entered in the same format. The benchmark priors in
`src/texgisa_survival/priors.json` were produced with this procedure.

## Comparison methods

```python
from texgisa_survival.backends.runner import available_methods, fit_predict

print(available_methods())
out = fit_predict("RandomSurvivalForest", X_train, time_train, event_train, X_test,
                  config={"n_estimators": 100, "min_samples_leaf": 5, "max_depth": 8})
out["risk"], out["curve"], out["curve_times"], out["point"], out["metadata"]
```

| Source | Methods |
|---|---|
| `lifelines` | Cox, Weibull AFT, log-normal AFT, log-logistic AFT |
| `scikit-survival` | Coxnet, random survival forest, gradient boosted Cox |
| `pycox` | DeepSurv, Cox-Time, Cox-CC, DeepHit, logistic hazard, PC-hazard, PMF, MTLR, N-MTLR |
| Wang et al. (ICDM 2021) | HarrellNMSE, HarrellNMAE, UnoNMSE, UnoNMAE |
| Author programs (`run_author_cli`) | sample-split DRO, survival kernets, SODEN |

Each method runs its own code unchanged in a fresh process; the adapter only converts inputs and
outputs (with a training-only `StandardScaler`). Outputs a method does not provide are `None`,
and upstream failures are reported, not repaired. Baselines never receive TEXGI, priors or other
TexGISa components. See `BACKENDS.md` for configurations and runtime requirements (the ICDM 2021
models need the original author checkout and a NumPy 1.x interpreter).

## Documentation

- `API_REFERENCE.md`: estimators, recipes and utilities
- `BACKENDS.md`: comparison interface
- `SCIENTIFIC_NOTES.md`: definitions of outputs and metrics
- `EXTENSIONS_SINCE_ICDM.md`: changes relative to the ICDM 2025 code
- `examples/`: runnable scripts
- `benchmarks/`: benchmark configurations, per-fold results and summaries

## Development

```bash
pip install ".[dev,recipes,baselines]"
pytest tests --override-ini="addopts="
python -m build
```

## Citation

```bibtex
@inproceedings{qin2025interpretable,
  author    = {Qin, Xinyu and Yu, Ruiheng and Khayati, Armin and Qiu, Zixiao and Zou, Gengyi and Li, Yan and Wang, Lu},
  title     = {Interpretable and Interactive Deep Survival Analysis with Time-dependent {EXtreme} Gradient Integration},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  pages     = {673--682},
  year      = {2025},
  doi       = {10.1109/ICDM65498.2025.00075}
}
```

## License

MIT. Optional dependencies keep their own licenses; original author code used by the comparison
interface is obtained separately and is not bundled.
