"""
Regression tests: attribution priors and the extreme-baseline generator must
actually influence training.

Each test fails on texgisa-survival 2.0.0, where
- TEXGI attributions were computed with create_graph=False, so the smoothness
  and expert penalties sent no gradient to the model weights;
- the generator's risk term was computed under torch.no_grad(), so the
  generator only learned to copy its input;
- the high-level expert loss was only active after fit() had finished.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from texgisa_survival import TexGISa
from texgisa_survival.texgi_utils import (
    attribution_temporal_l1,
    expert_penalty,
    texgi_time_series,
)
from texgisa_survival.models.texgisa import TEXGISAModel
from texgisa_survival.models.texgisa_advanced import (
    _destandardize,
    _standardize_apply,
    sample_extreme_code,
    train_adv_generator,
)


def _small_net(d=6, t=8):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, t), nn.Sigmoid())


def _param_grad_norm(net):
    return sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("penalty", ["smooth", "expert"])
def test_attribution_priors_reach_model_weights(penalty):
    net = _small_net()
    X = torch.randn(32, 6)
    phi = texgi_time_series(net, X, M=5, create_graph=True)
    loss = attribution_temporal_l1(phi) if penalty == "smooth" else expert_penalty(phi, [0, 1])
    net.zero_grad()
    loss.backward()
    assert _param_grad_norm(net) > 0


def test_default_attributions_carry_no_weight_gradient():
    # Post-hoc explanation (the default) should stay cheap and detached from the weights.
    net = _small_net()
    phi = texgi_time_series(net, torch.randn(16, 6), M=5)
    assert phi.shape == (8, 16, 6)
    assert not phi.requires_grad


def test_generator_raises_risk(tmp_path):
    net = _small_net()
    X = torch.randn(128, 6)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True)
    G, _ = train_adv_generator(
        net,
        X,
        mu,
        sd,
        epochs=60,
        batch_size=64,
        device="cpu",
        alpha_dist=0.1,
        checkpoint_path=str(tmp_path / "gen.pt"),
    )
    with torch.no_grad():
        z = torch.randn(128, 16)
        e = sample_extreme_code(128)
        xb = _destandardize(G(_standardize_apply(X, mu, sd), z, e), mu, sd)
        assert net(xb).sum(1).mean() > net(X).sum(1).mean()
    # model weights must be trainable again afterwards
    assert all(p.requires_grad for p in net.parameters())


def test_high_level_expert_loss_is_active_during_fit(monkeypatch):
    seen = []
    original = TEXGISAModel._compute_expert_loss

    def spy(self, X, hazards):
        out = original(self, X, hazards)
        seen.append((out.detach().item(), out.requires_grad))
        return out

    monkeypatch.setattr(TEXGISAModel, "_compute_expert_loss", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    time = rng.exponential(10, 200)
    event = rng.binomial(1, 0.7, 200)

    model = TexGISa(
        hidden_layers=[16],
        num_time_bins=6,
        lambda_expert=1.0,
        ig_steps=5,
        random_state=0,
        verbose=0,
    )
    model.add_expert_rule(0, ">", "mean", sign="-1")
    model.fit(X, time, event, epochs=2, batch_size=64, verbose=0)

    assert seen, "expert loss was never evaluated"
    assert any(v > 0 and g for v, g in seen), "expert loss never carried gradient"


def test_evaluate_matches_metric_functions():
    # evaluate() used to pass (time, event, risk) to concordance_index(time, scores, event).
    from texgisa_survival import concordance_index

    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    time = rng.exponential(5, 150)
    event = rng.binomial(1, 0.6, 150)
    model = TexGISa(hidden_layers=[16], num_time_bins=5, random_state=0, verbose=0)
    model.fit(X, time, event, epochs=2, verbose=0)
    scores = model.evaluate(X, time, event, metrics=["c-index", "brier_score", "ibs"])
    assert scores["c-index"] == pytest.approx(concordance_index(time, model.predict_risk(X), event))
    assert 0.0 <= scores["brier_score"] <= 1.0
    assert np.isfinite(scores["ibs"])


def test_concordance_index_takes_risk_scores():
    from texgisa_survival import concordance_index

    rng = np.random.default_rng(3)
    risk = rng.normal(size=400)
    time = rng.exponential(np.exp(-risk))  # higher risk -> earlier event
    event = np.ones(400)
    assert concordance_index(time, risk, event) > 0.6


def test_named_expert_rule_reaches_training():
    import pandas as pd

    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["age", "tumor_size", "grade"])
    time = rng.exponential(10, 200)
    event = rng.binomial(1, 0.7, 200)

    def fitted_hazards(lambda_expert):
        m = TexGISa(
            hidden_layers=[16],
            num_time_bins=5,
            lambda_expert=lambda_expert,
            ig_steps=5,
            random_state=0,
            verbose=0,
        )
        m.add_expert_rule("tumor_size", ">", "median", sign=-1, weight=5.0)
        m.fit(X, time, event, epochs=3, batch_size=64, verbose=0)
        return m.predict_survival(X)

    assert not np.allclose(fitted_hazards(0.0), fitted_hazards(1.0))


def test_unknown_feature_in_rule_raises():
    rng = np.random.default_rng(5)
    import pandas as pd

    X = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    m = TexGISa(hidden_layers=[8], num_time_bins=3, lambda_expert=1.0, verbose=0)
    m.add_expert_rule("typo", ">", "mean", sign=1)
    with pytest.raises(ValueError):
        m.fit(X, rng.exponential(1, 50), np.ones(50, dtype=int), epochs=1, verbose=0)


def _fitted(n=150, p=4, bins=5, seed=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    time = rng.exponential(5, n)
    event = rng.binomial(1, 0.6, n)
    m = TexGISa(hidden_layers=[16], num_time_bins=bins, ig_steps=5, random_state=0, verbose=0)
    m.fit(X, time, event, epochs=2, verbose=0)
    return m, X, time, event


def test_survival_interpolation_starts_at_one_and_matches_bin_edges():
    m, X, _, _ = _fitted()
    edges = m.model.time_bins
    S_bins = m.predict_survival(X[:3])
    assert np.allclose(m.predict_survival(X[:3], times=[edges[0]]), 1.0)
    assert np.allclose(m.predict_survival(X[:3], times=edges[1:]), S_bins)
    mid = 0.5 * (edges[0] + edges[1])
    assert np.allclose(m.predict_survival(X[:3], times=[mid])[:, 0], 0.5 * (1 + S_bins[:, 0]))


def test_predict_hazard_without_times_returns_model_hazards():
    m, X, _, _ = _fitted()
    h = m.model.predict_hazard(X[:4])
    assert h.shape == (4, 5)
    assert np.allclose(np.cumprod(1 - h, axis=1), m.predict_survival(X[:4]), atol=1e-6)


def test_save_load_round_trip_keeps_predictions_and_explanations(tmp_path):
    from texgisa_survival import load_model

    m, X, _, _ = _fitted(bins=7)
    path = tmp_path / "m.pkl"
    m.save(str(path))
    m2 = load_model(str(path), verbose=0)
    assert np.allclose(m.predict_survival(X[:5]), m2.predict_survival(X[:5]), atol=1e-6)
    assert m2.model.num_time_bins == 7
    assert m2.explain(X[:2]).shape == (7, 2, 4)
    assert len(m2.get_feature_importance()) == 4


def test_brier_rejects_misaligned_survival_matrix():
    from texgisa_survival import integrated_brier_score

    m, X, time, event = _fitted()
    with pytest.raises(ValueError):
        integrated_brier_score(time, m.predict_survival(X), event)  # 5 columns vs all event times
    grid = np.linspace(time.min(), np.percentile(time, 80), 10)
    ibs = integrated_brier_score(time, m.predict_survival(X, times=grid), event, eval_times=grid)
    assert 0 <= ibs <= 1


def test_mysatrainer_step_with_both_priors(tmp_path, monkeypatch):
    from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
    from texgisa_survival.models.texgisa_advanced import MySATrainer, build_supervision

    monkeypatch.chdir(tmp_path)  # generator checkpoint is written to the working directory
    torch.manual_seed(0)
    n, d, t = 64, 5, 6
    X = torch.randn(n, d)
    intervals = torch.randint(0, t, (n,)).numpy()
    events = torch.randint(0, 2, (n,)).numpy()
    y, m = build_supervision(intervals, events, t)
    net = MultiTaskSurvivalNet(input_dim=d, hidden_layers=[16], num_time_bins=t)
    names = [f"f{i}" for i in range(d)]
    trainer = MySATrainer(
        net,
        device="cpu",
        lambda_smooth=0.01,
        lambda_expert=0.1,
        expert_rules={"important_features": ["f0", "f1"]},
        feat2idx={k: i for i, k in enumerate(names)},
        ig_steps=4,
        ig_batch_samples=16,
        gen_epochs=2,
        gen_batch=32,
        X_train_ref=X,
    )
    before = [p.detach().clone() for p in net.parameters()]
    main, smooth, expert = trainer.step(
        X, torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(m, dtype=torch.float32)
    )
    assert np.isfinite([main, smooth, expert]).all() and smooth > 0 and expert > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_numeric_rule_threshold_is_in_original_units():
    rng = np.random.default_rng(7)
    raw = np.column_stack([rng.integers(0, 2, 120), rng.normal(60, 10, 120)]).astype(np.float32)
    m = TexGISa(hidden_layers=[8], num_time_bins=4, verbose=0, random_state=0)
    m.fit(raw, rng.exponential(5, 120), rng.binomial(1, 0.6, 120), epochs=1, verbose=0)
    inner = m.model
    X_std = torch.as_tensor(inner.scaler.transform(raw), dtype=torch.float32)
    # binary feature == 1 must select exactly the rows whose raw value is 1
    thr = inner._compute_threshold(1, X_std, 0)
    assert np.array_equal((torch.abs(X_std[:, 0] - thr) < 1e-4).numpy(), raw[:, 0] == 1)
    # age > 65 must select the rows whose raw age exceeds 65
    thr = inner._compute_threshold(65, X_std, 1)
    assert np.array_equal((X_std[:, 1] > thr).numpy(), raw[:, 1] > 65)


def test_quantile_binning_and_expected_time_risk(tmp_path):
    from texgisa_survival import load_model

    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    time = rng.lognormal(2, 1, 300)  # skewed times: quantile bins differ from uniform ones
    event = rng.binomial(1, 0.7, 300)
    m = TexGISa(
        hidden_layers=[8],
        num_time_bins=6,
        time_binning="quantile",
        risk_score="expected_time",
        lambda_cox=0.0,  # without the Cox head the risk comes from the hazards
        weight_decay=1e-3,
        random_state=0,
        verbose=0,
    )
    m.fit(X, time, event, epochs=2, verbose=0, validation_split=0.0)
    edges = m.model.time_bins
    assert np.all(np.diff(edges) > 0) and edges[0] == time.min() and edges[-1] == time.max()
    counts = np.histogram(time[event == 1], bins=edges)[0]
    assert counts.min() > 0.5 * counts.mean()  # roughly equal numbers of events per bin

    # expected-time risk = minus the area under the predicted survival curve
    S = m.predict_survival(X[:4], times=np.linspace(edges[0], edges[-1], 2001))
    rmst = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(S, np.linspace(edges[0], edges[-1], 2001), axis=1)
    assert np.allclose(m.predict_risk(X[:4]), -rmst, rtol=1e-3)

    path = tmp_path / "q.pkl"
    m.save(str(path))
    m2 = load_model(str(path), verbose=0)
    assert m2.model.time_binning == "quantile" and m2.model.risk_score == "expected_time"
    assert np.allclose(m.predict_risk(X[:4]), m2.predict_risk(X[:4]), atol=1e-6)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        TexGISa(activation="nope", verbose=0).fit(
            np.random.randn(20, 2), np.random.rand(20) + 1, np.ones(20, dtype=int), epochs=1
        )


def test_unknown_model_option_raises():
    with pytest.raises(TypeError):
        TexGISa(lambda_rank=1.0, verbose=0)


def test_feature_importance_with_default_ig_steps():
    # ig_steps=50 asks for 16 * 10 reference samples, more than the 100 kept as
    # background; this used to raise StopIteration
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    model = TexGISa(random_state=0, verbose=0)
    model.fit(X, rng.exponential(5, 300), rng.binomial(1, 0.6, 300), epochs=2, verbose=0)
    imp = model.get_feature_importance()
    assert len(imp) == 4 and np.isclose(imp["importance"].sum(), 1.0)


def test_explain_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    model = TexGISa(hidden_layers=[16], num_time_bins=5, ig_steps=5, random_state=0, verbose=0)
    model.fit(X, rng.exponential(5, 120), rng.binomial(1, 0.6, 120), epochs=2, verbose=0)
    phi = model.explain(X[:7])
    assert phi.shape == (5, 7, 4)
    assert np.isfinite(phi).all()


def test_cox_loss_prefers_correct_ranking_and_matches_lifelines():
    from lifelines import CoxPHFitter
    import pandas as pd
    from texgisa_survival.models.texgisa import TEXGISAModel

    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    e = torch.tensor([1.0, 1.0, 0.0, 1.0])
    good = torch.tensor([3.0, 2.0, 1.0, 0.0])  # earlier events have higher risk
    assert TEXGISAModel._cox_loss(good, t, e) < TEXGISAModel._cox_loss(-good, t, e)
    # with a linear risk beta*x, the minimizer matches lifelines' Cox estimate
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    tt = rng.exponential(np.exp(-0.8 * x))
    ee = rng.binomial(1, 0.8, 200).astype(float)
    beta = CoxPHFitter().fit(pd.DataFrame({"x": x, "T": tt, "E": ee}), "T", "E").params_["x"]
    grid = np.linspace(beta - 0.3, beta + 0.3, 61)
    losses = [
        TEXGISAModel._cox_loss(
            torch.tensor(b * x, dtype=torch.float64), torch.tensor(tt), torch.tensor(ee)
        ).item()
        for b in grid
    ]
    assert abs(grid[int(np.argmin(losses))] - beta) < 0.02


def test_cox_loss_is_zero_without_events():
    from texgisa_survival.models.texgisa import TEXGISAModel

    r = torch.tensor([0.5, -0.2], requires_grad=True)
    loss = TEXGISAModel._cox_loss(r, torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    assert loss.item() == 0.0 and torch.isfinite(loss)


def test_pch_loss_rewards_correct_event_bin():
    from texgisa_survival.models.texgisa import TEXGISAModel

    m = TEXGISAModel()
    bins, events, frac = torch.tensor([2]), torch.tensor([1.0]), torch.tensor([0.5])
    right = torch.tensor([[0.02, 0.02, 0.6, 0.02]])  # high hazard in the event bin
    wrong = torch.tensor([[0.6, 0.02, 0.02, 0.02]])
    assert m._pch_loss(right, bins, events, frac) < m._pch_loss(wrong, bins, events, frac)


@pytest.mark.parametrize(
    "opts",
    [
        dict(),  # defaults: ResNet, Cox head, PCH likelihood, calibration, EMA, input noise
        dict(architecture="mlp"),
        dict(architecture="mlp", norm="batch", activation="relu", dropout=0.1),
        dict(lambda_cox=0.0, risk_score="hazard_sum"),
        dict(lambda_cox=0.0, likelihood="bce", time_binning="uniform"),
        dict(calibrate=False, ema_decay=0.0, input_noise=0.0),
        dict(input_gate=True, es_metric="cindex"),
    ],
)
def test_model_options_train_explain_and_reload(tmp_path, opts):
    from texgisa_survival import load_model

    rng = np.random.default_rng(13)
    X = rng.normal(size=(180, 4)).astype(np.float32)
    t, e = rng.exponential(np.exp(-X[:, 0])), rng.binomial(1, 0.7, 180)
    m = TexGISa(
        hidden_layers=[16, 16], num_time_bins=6, ig_steps=4, random_state=0, verbose=0, **opts
    )
    m.fit(
        X[:140],
        t[:140],
        e[:140],
        validation_data=(X[140:], t[140:], e[140:]),
        epochs=5,
        batch_size=64,
        verbose=0,
    )
    r, S = m.predict_risk(X[:20]), m.predict_survival(X[:20])
    assert np.isfinite(r).all() and S.shape == (20, 6)
    assert np.all(np.diff(S, axis=1) <= 1e-7) and np.all((S >= 0) & (S <= 1))
    assert m.explain(X[:3]).shape == (6, 3, 4)
    path = tmp_path / "v.pkl"
    m.save(str(path))
    m2 = load_model(str(path), verbose=0)
    for k, v in opts.items():
        assert getattr(m2.model, k) == v
    assert np.allclose(m2.predict_survival(X[:20]), S, atol=1e-6)
    assert np.allclose(m2.predict_risk(X[:20]), r, atol=1e-5)


def test_cox_risk_shifts_every_hazard_logit():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    m = TexGISa(hidden_layers=[8], num_time_bins=5, random_state=0, verbose=0)
    m.fit(X, rng.exponential(5, 100), rng.binomial(1, 0.7, 100), epochs=2, verbose=0)
    net = m.model.model
    net.eval()
    with torch.no_grad():
        f = net._features(torch.as_tensor(X[:6]))
        assert torch.allclose(net._logits(f), net.base.logits(f) + net.cox(f))
        # the risk score is the Cox head output
        x = torch.as_tensor(m.model.scaler.transform(X[:6]))
        assert np.allclose(m.predict_risk(X[:6]), net.risk(x).numpy(), atol=1e-6)


def test_fit_without_validation_data_holds_out_a_split():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    t, e = rng.exponential(5, 200), rng.binomial(1, 0.7, 200)
    m = TexGISa(hidden_layers=[8], random_state=0, verbose=0)
    m.fit(X, t, e, epochs=3, verbose=0)
    assert m.model.scaler.n_samples_seen_ == 160  # 20% held out for validation
    m0 = TexGISa(hidden_layers=[8], random_state=0, verbose=0)
    m0.fit(X, t, e, epochs=3, verbose=0, validation_split=0.0)
    assert m0.model.scaler.n_samples_seen_ == 200
    assert np.isfinite(m.predict_risk(X)).all() and np.isfinite(m0.predict_risk(X)).all()


def test_calibration_does_not_increase_validation_bce():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    t, e = rng.exponential(np.exp(-X[:, 0])), rng.binomial(1, 0.7, 300)
    val = (X[240:], t[240:], e[240:])
    losses = []
    for cal in (False, True):
        m = TexGISa(
            hidden_layers=[16],
            num_time_bins=6,
            lambda_cox=0.0,
            calibrate=cal,
            random_state=0,
            verbose=0,
        )
        m.fit(X[:240], t[:240], e[:240], validation_data=val, epochs=5, verbose=0)
        losses.append(m.model._validate(val))
    assert losses[1] <= losses[0] + 1e-6


def test_large_learning_rate_stays_finite():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    t, e = rng.exponential(np.exp(-X[:, 0])), rng.binomial(1, 0.7, 200)
    for arch in ("resnet", "mlp"):
        m = TexGISa(
            hidden_layers=[16], num_time_bins=8, architecture=arch, random_state=0, verbose=0
        )
        m.fit(
            X[:160],
            t[:160],
            e[:160],
            validation_data=(X[160:], t[160:], e[160:]),
            epochs=5,
            learning_rate=0.1,
            verbose=0,
        )
        assert np.isfinite(m.predict_survival(X)).all()
