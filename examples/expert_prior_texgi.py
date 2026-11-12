"""TexGISa with an expert prior and time-resolved TEXGI explanations (Figure 1 of the MLOSS paper)."""
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
phi = texgi_time_series(net, x[:5], trainer.G, trainer.ref_stats)  # TEXGI
print("TEXGI attributions:", tuple(phi.shape), "(interval, patient, feature)")
