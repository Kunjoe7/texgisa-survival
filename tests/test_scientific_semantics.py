# Copyright (c) 2026 DHAI Lab
# Licensed under the MIT License; see LICENSE.

"""Regression checks for mathematical semantics used in the MLOSS article."""
import numpy as np
import pytest
import torch
from torch import nn
from texgisa_survival import TexGISa
from texgisa_survival.models.texgisa import TEXGISAModel
from texgisa_survival.texgi_utils import integrated_gradients_time


class IncreasingHazard(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return torch.sigmoid(self.weight * x[:, :1]).repeat(1, 3)


def test_high_reference_attribution_and_feature_direction_have_opposite_signs():
    net = IncreasingHazard()
    x, ref = torch.tensor([[-1.0], [0.0], [1.0]]), torch.full((3, 1), 2.0)
    phi = integrated_gradients_time(net, x, ref, 0, M=200)
    slope = integrated_gradients_time(net, x, ref, 0, M=200, multiply_by_inputs=False)
    assert (phi < 0).all() and (slope > 0).all()
    assert torch.allclose(phi, slope * (x - ref))


def test_completeness_error_decreases_with_integration_steps():
    net = IncreasingHazard()
    x, ref = torch.tensor([[-1.0], [0.0]]), torch.full((2, 1), 2.0)
    delta = (net(x) - net(ref))[:, 0]
    errors = [
        (integrated_gradients_time(net, x, ref, 0, M=n).sum(1) - delta).abs().max()
        for n in (5, 500)
    ]
    assert errors[1] < errors[0] and errors[1] < 5e-4


def test_direction_rule_penalizes_wrong_derivative_and_has_parameter_gradient():
    m = TEXGISAModel(num_time_bins=3, ig_steps=10)
    m.model = IncreasingHazard()
    m.feature_names = ['age']
    m.baseline_reference_ = np.array([2.0])
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    rule = dict(feature='age', relation='>', threshold=-3, sign=1, weight=2,
                min_magnitude=0, direction_mode='path_gradient')
    m.expert_rules = [rule]
    assert m._compute_expert_loss(x, m.model(x)).item() == 0
    rule['sign'] = -1
    wrong = m._compute_expert_loss(x, m.model(x))
    assert wrong.item() > 0
    wrong.backward()
    assert m.model.weight.grad.abs().item() > 0
    rule['sign'], rule['direction_mode'] = 1, 'attribution'
    assert m._compute_expert_loss(x, m.model(x)).item() > 0


def test_breslow_ties_match_explicit_risk_sets_and_are_order_invariant():
    r = torch.tensor([0.6, -0.2, 0.9, 0.1], requires_grad=True)
    t = torch.tensor([1., 1., 2., 3.])
    e = torch.tensor([1., 1., 0., 1.])
    expected = -sum(r[i] - torch.logsumexp(r[t >= t[i]], 0) for i in (0, 1, 3)) / 3
    actual = TEXGISAModel._cox_loss(r, t, e)
    assert torch.allclose(actual, expected)
    perm = torch.tensor([1, 0, 2, 3])
    assert torch.allclose(actual, TEXGISAModel._cox_loss(r[perm], t[perm], e[perm]))
    actual.backward()
    assert torch.isfinite(r.grad).all()


def test_censoring_does_not_count_unobserved_remainder_of_interval():
    m = TEXGISAModel(num_time_bins=3)
    labels, masks = m._create_supervision(np.array([1, 1, 2]), np.array([0, 1, 0]),
                                        np.array([.5, .5, 1.]))
    assert np.array_equal(masks, [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    assert np.array_equal(labels, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_empty_event_cox_loss_keeps_a_zero_gradient():
    r = torch.tensor([1., 2.], requires_grad=True)
    TEXGISAModel._cox_loss(r, torch.tensor([1., 2.]), torch.zeros(2)).backward()
    assert torch.equal(r.grad, torch.zeros(2))


def test_trainer_does_not_reuse_an_unrequested_working_directory_checkpoint(tmp_path, monkeypatch):
    import texgisa_survival.models.texgisa_advanced as advanced

    monkeypatch.chdir(tmp_path)
    (tmp_path / "texgisa_generator.pt").write_bytes(b"checkpoint from another fold")
    accessed = []
    fitted = []

    def load_checkpoint(*args, **kwargs):
        accessed.append(True)
        raise ValueError("foreign checkpoint")

    def fit_generator(model, x, mu, std, **kwargs):
        fitted.append(kwargs.get("checkpoint_path"))
        return nn.Linear(2, 2), {"mu": mu, "std": std}

    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(advanced, "train_adv_generator", fit_generator)
    trainer = advanced.MySATrainer(nn.Linear(2, 2), lambda_expert=1,
                                   X_train_ref=torch.randn(8, 2))
    trainer.fit_generator_if_needed()
    assert accessed == [] and fitted == [None]


@pytest.mark.parametrize('params', [dict(sign=0), dict(relation='!='),
                                  dict(weight=-1), dict(direction_mode='unknown')])
def test_invalid_rules_raise_instead_of_being_silently_skipped(params):
    values = dict(feature='age', relation='>', threshold='median', sign=1)
    values.update(params)
    with pytest.raises(ValueError):
        TexGISa(verbose=0).add_expert_rule(**values)


def test_training_baseline_and_thresholds_survive_save_load(tmp_path):
    from texgisa_survival import load_model
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 2)).astype('float32')
    m = TexGISa(hidden_layers=[8], num_time_bins=3, ig_steps=4, random_state=0, verbose=0)
    m.add_expert_rule(0, '>', 'median', sign=1)
    m.fit(x, rng.exponential(5, 100), rng.binomial(1, .6, 100), epochs=2, verbose=0)
    ref = m.model.baseline_reference_.copy()
    phi = m.explain(x[:3])
    m.save(str(tmp_path/'model.pkl'))
    restored = load_model(str(tmp_path/'model.pkl'), verbose=0)
    assert np.array_equal(ref, restored.model.baseline_reference_)
    assert restored.model.expert_rules[0]['direction_mode'] == 'path_gradient'
    assert np.allclose(phi, restored.explain(x[:3]))
    batch = torch.as_tensor(m.model.scaler.transform(x[:3]))
    threshold = m.model._compute_threshold('median', batch, 0)
    assert threshold == m.model._compute_threshold('median', batch[:1], 0)
