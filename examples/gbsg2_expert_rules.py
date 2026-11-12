"""
Real-data example: German Breast Cancer Study Group 2 (GBSG2, 686 patients).

The data ship with lifelines, a dependency of texgisa-survival, so the example
needs no download. It
  1. fits TexGISa with and without two expert rules,
  2. compares C-index and integrated Brier score on a held-out split,
  3. prints per-interval TEXGI attributions for the number of positive nodes.

Run:  python examples/gbsg2_expert_rules.py
"""

import numpy as np
from lifelines.datasets import load_gbsg2
from sklearn.model_selection import train_test_split

from texgisa_survival import TexGISa, concordance_index, integrated_brier_score

# ----------------------------------------------------------------- data
df = load_gbsg2()
df["horTh"] = (df["horTh"] == "yes").astype(float)  # hormonal therapy
df["menostat"] = (df["menostat"] == "Post").astype(float)  # post-menopausal
df["tgrade"] = df["tgrade"].map({"I": 1.0, "II": 2.0, "III": 3.0})
time, event = df.pop("time").to_numpy(float), df.pop("cens").to_numpy(int)
X = df.astype(float)

X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
    X, time, event, test_size=0.25, random_state=0, stratify=event
)
X_fit, X_val, t_fit, t_val, e_fit, e_val = train_test_split(
    X_tr, t_tr, e_tr, test_size=0.2, random_state=0, stratify=e_tr
)
grid = np.linspace(np.percentile(t_te, 5), np.percentile(t_te, 80), 20)


def fit(with_rules: bool) -> TexGISa:
    model = TexGISa(
        hidden_layers=[32, 16],
        num_time_bins=10,
        dropout=0.1,
        lambda_expert=0.1 if with_rules else 0.0,
        ig_steps=10,
        random_state=0,
        verbose=0,
    )
    if with_rules:
        # Clinical knowledge: many positive lymph nodes raise the hazard,
        # hormonal therapy lowers it.
        model.add_expert_rule("pnodes", ">", "median", sign=1, weight=1.0)
        model.add_expert_rule("horTh", "==", 1, sign=-1, weight=1.0)
    model.fit(
        X_fit,
        t_fit,
        e_fit,
        validation_data=(X_val.to_numpy(), t_val, e_val),
        epochs=100,
        batch_size=64,
        patience=10,
        verbose=0,
    )
    return model


for with_rules in (False, True):
    model = fit(with_rules)
    c = concordance_index(t_te, model.predict_risk(X_te), e_te)
    ibs = integrated_brier_score(
        t_te, model.predict_survival(X_te, times=grid), e_te, eval_times=grid
    )
    label = "with expert rules   " if with_rules else "without expert rules"
    print(f"{label}: C-index = {c:.3f}, integrated Brier score = {ibs:.3f}")

# --------------------------------------------- time-resolved explanation
phi = model.explain(X_te.iloc[:50])  # (intervals, patients, features)
j = list(X.columns).index("pnodes")
print("\nMean TEXGI attribution of 'pnodes' per time interval (last model):")
for k, v in enumerate(phi[:, :, j].mean(axis=1)):
    print(f"  interval {k + 1:2d}: {v:+.4f}")

print("\nGlobal feature ranking:")
print(model.get_feature_importance().to_string(index=False))
