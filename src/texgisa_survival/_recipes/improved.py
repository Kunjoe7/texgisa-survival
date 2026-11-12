"""New PCH-head development hypotheses; no edits to the public/frozen package.

The native head is log *interval cumulative hazard*, not a sigmoid/softplus
logit. Prediction, RMST ranking, calibration and TEXGI all share that head.
TEXGI below explains log cumulative rates, explicitly not hazard probabilities.
"""
from pathlib import Path
import copy
import json
import math

import numpy as np
import torch
from torch import nn

from .candidate import CandidateModel
from .metrics import CensoringKM, evaluation_grid


METHODS = {"ImproveAlign", "ImproveBrier", "ImproveResidual", "ImprovePH",
           "ImproveResidualNoPrior"}


def checkpoint_eligible(epoch, warmup):
    """No best/patience update before full prior weight, including NoPrior."""
    return epoch+1 >= max(1, int(warmup))


def _rates(log_rates):
    value = log_rates.to(torch.float64).exp()
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite interval cumulative rate")
    return value


def pch_log_rate_nll(log_rates, times, events, edges):
    """PCH NLL, dropping parameter-independent log interval-width constants."""
    rates = _rates(log_rates)
    edges = torch.as_tensor(edges, dtype=rates.dtype, device=rates.device)
    times = torch.as_tensor(times, dtype=rates.dtype, device=rates.device)
    events = torch.as_tensor(events, dtype=rates.dtype, device=rates.device)
    clipped = times.clamp(min=float(edges[0]), max=float(edges[-1]))
    observed = events * (times <= edges[-1])
    index = torch.bucketize(clipped.contiguous(), edges[1:], right=True).clamp(max=len(edges)-2)
    fraction = ((clipped-edges[index])/(edges[index+1]-edges[index])).clamp(0., 1.)
    before = torch.cat([torch.zeros_like(rates[:, :1]), rates.cumsum(1)[:, :-1]], 1)
    selected = index[:, None]
    return (before.gather(1, selected).squeeze(1)
            + fraction*rates.gather(1, selected).squeeze(1)
            - observed*log_rates.to(rates.dtype).gather(1, selected).squeeze(1)).mean()


def survival_from_log_rates(log_rates, edges, grid):
    rates = _rates(log_rates)
    edges = torch.as_tensor(edges, dtype=rates.dtype, device=rates.device)
    grid = torch.as_tensor(grid, dtype=rates.dtype, device=rates.device)
    if grid.ndim != 1 or not torch.isfinite(grid).all():
        raise ValueError("Prediction times must be a finite vector")
    fraction = ((grid[None, :]-edges[:-1, None]) / torch.diff(edges)[:, None]).clamp(0., 1.)
    return torch.exp(-(rates @ fraction))


def truncated_rmst(log_rates, edges, horizon):
    """Exact integral of the same PCH curve from zero to fitting-only tau."""
    rates = _rates(log_rates)
    edges = torch.as_tensor(edges, dtype=rates.dtype, device=rates.device)
    horizon = torch.as_tensor(horizon, dtype=rates.dtype, device=rates.device)
    if horizon <= 0 or horizon > edges[-1] or edges[0] != 0:
        raise ValueError("RMST requires zero origin and 0 < tau <= fitted support")
    width = torch.diff(edges)
    included = (horizon-edges[:-1]).clamp(min=0.)
    included = torch.minimum(included, width)
    amount = rates*(included/width)
    denominator = torch.where(amount > 0., amount, torch.ones_like(amount))
    ratio = -torch.expm1(-amount)/denominator
    ratio = torch.where(amount < 1e-5, 1.-amount/2.+amount.square()/6., ratio)
    before = torch.cat([torch.zeros_like(rates[:, :1]), rates.cumsum(1)[:, :-1]], 1)
    return (torch.exp(-before)*included*ratio).sum(1)


def ipcw_brier(survival, times, events, grid, event_g_left, grid_g):
    """Training-KM IPCW IBS: event/G(T-), alive/G(u), censored-before-u zero."""
    dtype, device = survival.dtype, survival.device
    times = torch.as_tensor(times, dtype=dtype, device=device)
    events = torch.as_tensor(events, dtype=dtype, device=device)
    grid = torch.as_tensor(grid, dtype=dtype, device=device)
    event_g_left = torch.as_tensor(event_g_left, dtype=dtype, device=device)
    grid_g = torch.as_tensor(grid_g, dtype=dtype, device=device)
    if survival.shape != (len(times), len(grid)) or len(grid) < 2:
        raise ValueError("IPCW curve/grid shape mismatch")
    if (torch.diff(grid) <= 0).any() or (grid_g < .05-1e-12).any():
        raise ValueError("IPCW grid must increase and stay within fitting G>=0.05")
    observed = (events[:, None] == 1) & (times[:, None] <= grid)
    alive = times[:, None] > grid
    if ((event_g_left <= 0) & observed.any(1)).any():
        raise ValueError("An evaluated event has no fitting censoring support")
    safe_event = torch.where(event_g_left > 0, event_g_left, torch.ones_like(event_g_left))
    errors = (observed*survival.square()/safe_event[:, None]
              + alive*(1.-survival).square()/grid_g[None, :])
    return torch.trapezoid(errors.mean(0), grid)/(grid[-1]-grid[0])


def fitting_baseline(times, events, edges, pseudo_events=.5, pseudo_exposure=1.):
    """Positive-smoothed event count / fraction-at-risk exposure per interval."""
    times, events, edges = np.asarray(times), np.asarray(events), np.asarray(edges)
    if pseudo_events <= 0 or pseudo_exposure <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Positive smoothing and strictly increasing edges required")
    fractions = np.clip((times[:, None]-edges[:-1])/np.diff(edges), 0., 1.)
    exposure = fractions.sum(0)
    index = np.clip(np.searchsorted(edges[1:], times, side="right"), 0, len(edges)-2)
    counts = np.bincount(index, weights=events*(times <= edges[-1]), minlength=len(edges)-1)
    rates = (counts+pseudo_events)/(exposure+pseudo_exposure)
    return np.log(rates), dict(event_counts=counts.tolist(), fraction_exposure=exposure.tolist(),
        pseudo_events=float(pseudo_events), pseudo_exposure=float(pseudo_exposure),
        initial_cumulative_rates=rates.tolist(), source="optimization fitting rows only")


class RateNetwork(nn.Module):
    """Bounded covariate contrasts, trainable time offsets and optional skip/PH."""
    def __init__(self, features, config, method, initial_log_rates, edges, horizon):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown improved recipe")
        self.method = method
        self.is_ph = method == "ImprovePH"
        self.has_skip = method in {"ImproveResidual", "ImproveResidualNoPrior", "ImprovePH"}
        self.contrast_bound = float(config.get("contrast_bound", 8.))
        self.baseline_shift_bound = float(config.get("baseline_shift_bound", 4.))
        if self.contrast_bound <= 0 or self.baseline_shift_bound <= 0:
            raise ValueError("Architectural contrast bounds must be positive")
        self.risk_scale = float(config.get("risk_scale", 5.))
        layers, previous = [], features
        for width in config["hidden"]:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.SELU(),
                           nn.Dropout(config["dropout"])])
            previous = width
        self.features = nn.Sequential(*layers)
        outputs = 1 if self.is_ph else len(initial_log_rates)
        self.nonlinear_head = nn.Linear(previous, outputs)
        nn.init.normal_(self.nonlinear_head.weight, std=.01)
        nn.init.zeros_(self.nonlinear_head.bias)
        self.skip = nn.Linear(features, outputs, bias=False) if self.has_skip else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
        self.baseline_delta = nn.Parameter(torch.zeros(len(initial_log_rates)))
        self.register_buffer("initial_log_rates", torch.as_tensor(initial_log_rates, dtype=torch.float32))
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float64))
        self.register_buffer("horizon", torch.tensor(float(horizon), dtype=torch.float64))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))

    def components(self, x):
        baseline = (self.initial_log_rates + self.baseline_shift_bound
                    * torch.tanh(self.baseline_delta/self.baseline_shift_bound))
        contrast = self.nonlinear_head(self.features(x))
        if self.skip is not None:
            contrast = contrast+self.skip(x)
        contrast = self.contrast_bound*torch.tanh(contrast/self.contrast_bound)
        return baseline, contrast

    def log_rates(self, x, calibrated=True):
        baseline, contrast = self.components(x)
        if calibrated:
            return baseline[None, :]+contrast/self.temperature+self.logit_bias
        return baseline[None, :]+contrast

    def raw_logits(self, x):
        # Native log cumulative rates, deliberately not legacy softplus logits.
        return self.log_rates(x, calibrated=False)

    def risk_from_log_rates(self, log_rates):
        return -self.risk_scale*truncated_rmst(log_rates, self.edges, self.horizon)/self.horizon

    def risk(self, x):
        return self.risk_from_log_rates(self.log_rates(x))

    def hazards_and_risk(self, x):
        log_rates = self.log_rates(x)
        return -torch.expm1(-_rates(log_rates)), self.risk_from_log_rates(log_rates)

    def forward(self, x):
        return self.hazards_and_risk(x)[0]


class _LogRateView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.log_rates(x)


class ImprovedModel(CandidateModel):
    """Curve-aligned, explicitly new log-rate-head PCH estimator."""
    def __init__(self, method, config, rules, feature_names, random_state=0):
        super().__init__(ranking_target="rmst", hidden_layers=config["hidden"],
            num_time_bins=config["bins"], likelihood="pch", device="cpu", random_state=random_state)
        if method not in METHODS:
            raise ValueError("Unknown improved method")
        self.method, self.config = method, dict(config)
        self.rules, self.feature_names = copy.deepcopy(rules), list(feature_names)
        for rule in rules:
            if rule["feature"] not in self.feature_names or rule["sign"] not in {-1, 1}:
                raise ValueError("Unresolved directional hypothesis")
            if rule["condition"] not in {"all", "above_training_median"}:
                raise ValueError("Unknown rule condition")
        self.trace, self.prior_audit = [], {}
        self.calibration_record = {"status": "not_requested"}
        self.has_prior = method != "ImproveResidualNoPrior"
        self.brier_weight = 0. if method == "ImproveAlign" else float(config["lambda_brier"])

    def _g(self, times):
        return self.censoring_.predict(np.asarray(times), left=True)

    def _brier(self, log_rates, times, events):
        curves = survival_from_log_rates(log_rates, self.time_bins, self.brier_grid_)
        event_g = self._g(torch.as_tensor(times).detach().cpu().numpy())
        return ipcw_brier(curves, times, events, self.brier_grid_, event_g, self.grid_g_)

    def predictive_loss(self, x, times, events):
        log_rates = self.model.log_rates(x)
        nll = pch_log_rate_nll(log_rates, times, events, self.time_bins)
        risk = self.model.risk_from_log_rates(log_rates)
        cox = self._cox_loss(risk, times, events)
        brier = self._brier(log_rates, times, events) if self.brier_weight else nll*0.
        total = nll+float(self.config["lambda_cox"])*cox+self.brier_weight*brier
        return total, dict(pch=nll, curve_cox=cox, ipcw_brier=brier)

    def _indices(self, samples):
        if samples is not None and 0 < samples < self.num_time_bins:
            step = max(1, self.num_time_bins//samples)
            return list(range(0, self.num_time_bins, step))[:samples]
        return list(range(self.num_time_bins))

    def prior_loss(self, x):
        if not self.has_prior:
            return x.sum()*0., dict(attr_smooth=0., attr_size=0., direction=0., active_rule_rows=0)
        from texgisa_survival.texgi_utils import texgi_time_series
        subset = x[:int(self.config["prior_rows"])]
        baseline = torch.as_tensor(self.baseline_reference_, dtype=x.dtype).expand_as(subset)
        training = self.model.training
        self.model.eval()
        try:
            indices = self._indices(int(self.config["prior_time_samples"]))
            path = texgi_time_series(_LogRateView(self.model), subset, baseline,
                M=int(self.config["ig_steps"]), t_sample=int(self.config["prior_time_samples"]),
                create_graph=True, multiply_by_inputs=False)
            phi = path*(subset-baseline).unsqueeze(0)
            centers = (self.model.edges[1:]+self.model.edges[:-1])/2.
            gaps = torch.diff(centers[indices]); gaps = gaps/gaps.mean()
            smooth = (torch.diff(phi, dim=0)/gaps[:, None, None]).abs().mean()
            size = phi.abs().mean()
            z = subset.detach().clone().requires_grad_(True)
            risk_gradient = torch.autograd.grad(self.model.risk(z).sum(), z, create_graph=True)[0]
            direction, active = phi.sum()*0., 0
            margin = float(self.config.get("direction_margin", .001))
            for rule in self.rules:
                j = self.feature_names.index(rule["feature"])
                mask = (subset[:, j] > self.rule_reference_["median"][j]
                        if rule["condition"] == "above_training_median"
                        else torch.ones(len(subset), dtype=torch.bool))
                if mask.any():
                    active += int(mask.sum())
                    direction = direction+float(rule.get("weight", 1.))*(
                        torch.relu(margin-rule["sign"]*path[:, mask, j]).mean()
                        + torch.relu(margin-rule["sign"]*risk_gradient[mask, j]).mean())
            total = (float(self.config["lambda_attr_smooth"])*smooth
                     + float(self.config.get("lambda_attr_size", .001))*size
                     + float(self.config["lambda_expert"])*direction)
            return total, dict(attr_smooth=float(smooth.detach()), attr_size=float(size.detach()),
                               direction=float(direction.detach()), active_rule_rows=active)
        finally:
            self.model.train(training)

    def _gradient_norm(self, loss, retain_graph=False):
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()),
                                       allow_unused=True, retain_graph=retain_graph)
        if any(not torch.isfinite(g).all() for g in gradients if g is not None):
            raise FloatingPointError("Nonfinite objective parameter gradient")
        return math.sqrt(sum(float(g.detach().double().square().sum())
                             for g in gradients if g is not None))

    def _audit_best(self, x, times, events, best_epoch):
        self.model.eval()
        subset = slice(0, min(len(x), int(self.config["batch"])))
        predictive, _ = self.predictive_loss(x[subset], times[subset], events[subset])
        predictive_norm = self._gradient_norm(predictive)
        prior, diagnostics = self.prior_loss(x[subset])
        prior_norm = self._gradient_norm(prior) if self.has_prior else 0.
        warmup = min(1., (best_epoch+1)/max(1, int(self.config.get("prior_warmup", 10))))
        return dict(predictive_parameter_gradient_norm=predictive_norm,
                    prior_parameter_gradient_norm=prior_norm,
                    effective_prior_gradient_norm=warmup*prior_norm,
                    prior_warmup_factor=warmup, **diagnostics,
                    stage="best predictive checkpoint, before calibration")

    def fit_improved(self, x, times, events, stop):
        if int(self.config["epochs"]) < max(1, int(self.config.get("prior_warmup", 10))):
            raise ValueError("Epoch budget cannot finish the fixed prior warmup")
        self.scaler.fit(x)
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        xs = torch.as_tensor(self.scaler.transform(stop[0]), dtype=torch.float32)
        tt, ee = torch.as_tensor(times, dtype=torch.float64), torch.as_tensor(events, dtype=torch.float64)
        st, se = torch.as_tensor(stop[1], dtype=torch.float64), torch.as_tensor(stop[2], dtype=torch.float64)
        self.time_bins = self._make_time_bins(np.asarray(times), np.asarray(events))
        self.num_time_bins = len(self.time_bins)-1
        self.brier_grid_ = evaluation_grid(times, events, points=int(self.config.get("brier_points", 24)))
        self.risk_horizon_ = float(self.brier_grid_[-1])
        self.censoring_ = CensoringKM(times, events)
        self.grid_g_ = self.censoring_.predict(self.brier_grid_)
        initial, self.baseline_initialization_ = fitting_baseline(times, events, self.time_bins,
            float(self.config.get("baseline_pseudo_events", .5)),
            float(self.config.get("baseline_pseudo_exposure", 1.)))
        self.model = RateNetwork(xt.shape[1], self.config, self.method, initial,
                                 self.time_bins, self.risk_horizon_)
        self.baseline_reference_ = np.quantile(xt.numpy(), .98, axis=0).astype(np.float32)
        self.rule_reference_ = {"median": np.median(xt.numpy(), axis=0)}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["wd"])
        best, best_state, best_epoch, stale = float("inf"), None, -1, 0
        for epoch in range(int(self.config["epochs"])):
            self.model.train(); records = []
            warmup = min(1., (epoch+1)/max(1, int(self.config.get("prior_warmup", 10))))
            for ids in torch.randperm(len(xt)).split(int(self.config["batch"])):
                optimizer.zero_grad()
                predictive, parts = self.predictive_loss(xt[ids], tt[ids], ee[ids])
                prior, diagnostics = self.prior_loss(xt[ids])
                total = predictive+warmup*prior
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite improved fitting objective")
                if not self.prior_audit:
                    predictive_norm = self._gradient_norm(predictive, retain_graph=True)
                    prior_norm = self._gradient_norm(prior, retain_graph=True) if self.has_prior else 0.
                    self.prior_audit["initial"] = dict(predictive_parameter_gradient_norm=predictive_norm,
                        prior_parameter_gradient_norm=prior_norm, effective_prior_gradient_norm=warmup*prior_norm,
                        prior_warmup_factor=warmup, **diagnostics, stage="first fitting batch before optimizer update")
                total.backward()
                if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                    raise FloatingPointError("Nonfinite improved total gradient")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                records.append(dict(predictive=float(predictive.detach()), total=float(total.detach()),
                    raw_weighted_prior=float(prior.detach()), effective_prior=float((warmup*prior).detach()),
                    gradient_norm_before_clip=float(norm), **{k: float(v.detach()) for k, v in parts.items()}, **diagnostics))
            self.model.eval()
            with torch.no_grad():
                validation = float(self.predictive_loss(xs, st, se)[0])
            if not np.isfinite(validation):
                raise FloatingPointError("Nonfinite improved checkpoint objective")
            self.trace.append(dict(epoch=epoch, prior_warmup_factor=warmup,
                checkpoint_eligible=checkpoint_eligible(epoch, self.config.get("prior_warmup", 10)),
                validation_predictive_objective=validation,
                **{key: float(np.mean([r[key] for r in records])) for key in records[0]}))
            if not checkpoint_eligible(epoch, self.config.get("prior_warmup", 10)):
                continue
            if validation < best:
                best, best_state, best_epoch, stale = validation, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= int(self.config["patience"]):
                break
        if best_state is None:
            raise FloatingPointError("No finite improved checkpoint")
        self.model.load_state_dict(best_state); self.model.eval(); self.is_fitted = True
        self.best_epoch_ = best_epoch
        self.prior_audit["best_checkpoint"] = self._audit_best(xt, tt, ee, best_epoch)
        self.training_data_sample = xt[:min(200, len(xt))].detach()

    def _fit_calibration(self, validation_data, steps=None):
        x, times, events = validation_data
        self.model.eval()
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            baseline, contrast = [v.detach() for v in self.model.components(xt)]
        log_t = self.model.temperature.detach().log().clone().requires_grad_(True)
        bias = self.model.logit_bias.detach().clone().requires_grad_(True)
        weight = float(self.config.get("calibration_brier_weight", self.brier_weight))
        def objective():
            log_rates = baseline[None, :]+contrast/log_t.exp()+bias
            nll = pch_log_rate_nll(log_rates, times, events, self.time_bins)
            brier = self._brier(log_rates, times, events) if weight else nll*0.
            return nll+weight*brier, nll, brier
        with torch.no_grad():
            initial = tuple(float(v) for v in objective())
        if not np.isfinite(initial).all():
            raise FloatingPointError("Nonfinite initial calibration objective")
        best, best_values = initial[0], initial
        best_t, best_b, accepted = float(log_t.detach()), float(bias.detach()), 0
        optimizer = torch.optim.Adam([log_t, bias], lr=.03)
        reason = "completed projected optimization"
        for _ in range(int(steps if steps is not None else self.config.get("calibration_steps", 60))):
            optimizer.zero_grad()
            value = objective()[0]
            if not torch.isfinite(value):
                reason = "nonfinite iterate rejected"; break
            value.backward()
            if not torch.isfinite(log_t.grad) or not torch.isfinite(bias.grad):
                reason = "nonfinite calibration gradient rejected"; break
            optimizer.step()
            with torch.no_grad():
                log_t.clamp_(math.log(.2), math.log(5.)); bias.clamp_(-3., 3.)
                values = tuple(float(v) for v in objective())
                if np.isfinite(values).all() and values[0] < best:
                    best, best_values, best_t, best_b = values[0], values, float(log_t), float(bias)
                    accepted += 1
        with torch.no_grad():
            self.model.temperature.fill_(math.exp(best_t)); self.model.logit_bias.fill_(best_b)
        self.calibration_record = dict(status="fitted" if accepted else "initial_retained",
            objective="calibration-row PCH NLL + fitting-KM IPCW IBS", brier_weight=weight,
            initial_objective=initial[0], final_objective=best_values[0],
            initial_nll=initial[1], final_nll=best_values[1], initial_brier=initial[2], final_brier=best_values[2],
            temperature=math.exp(best_t), bias=best_b, accepted_improvements=accepted,
            stop_reason=reason, n_calibration=len(times), censoring_source="core fitting rows only",
            parameterization="time baseline unchanged; positive covariate-contrast temperature and global log-rate bias")

    def _x_tensor(self, x):
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return torch.as_tensor(self.scaler.transform(np.asarray(x, dtype=np.float32)), dtype=torch.float32)

    def predict_survival(self, x, times=None):
        self.model.eval()
        with torch.no_grad():
            log_rates = self.model.log_rates(self._x_tensor(x))
            grid = self.time_bins[1:] if times is None else times
            return survival_from_log_rates(log_rates, self.time_bins, grid).numpy()

    def predict_risk(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model.risk(self._x_tensor(x)).numpy()

    def explain(self, x, n_steps=None, time_bins=None):
        from texgisa_survival.texgi_utils import texgi_time_series
        self.model.eval()
        z = self._x_tensor(x)
        baseline = torch.as_tensor(self.baseline_reference_).expand_as(z)
        return texgi_time_series(_LogRateView(self.model), z, baseline,
            M=int(n_steps or self.config["ig_steps"]), t_sample=time_bins).detach().numpy()

    def final_prior_diagnostics(self, x):
        from texgisa_survival.texgi_utils import texgi_time_series
        self.model.eval()
        z = self._x_tensor(x[:int(self.config["prior_rows"]) ]).requires_grad_(True)
        baseline = torch.as_tensor(self.baseline_reference_).expand_as(z)
        path = texgi_time_series(_LogRateView(self.model), z, baseline,
            M=int(self.config["ig_steps"]), t_sample=int(self.config["prior_time_samples"]),
            multiply_by_inputs=False)
        gradient = torch.autograd.grad(self.model.risk(z).sum(), z)[0]
        if not torch.isfinite(path).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite final log-rate TEXGI or curve-risk gradient")
        margin = float(self.config.get("direction_margin", .001)); records = []
        for rule in self.rules:
            j = self.feature_names.index(rule["feature"])
            mask = (z[:, j] > self.rule_reference_["median"][j]
                    if rule["condition"] == "above_training_median" else torch.ones(len(z), dtype=torch.bool))
            records.append(dict(feature=rule["feature"], active_rows=int(mask.sum()),
                path_sign_violation=float((rule["sign"]*path[:, mask, j] < 0).double().mean()) if mask.any() else None,
                risk_sign_violation=float((rule["sign"]*gradient[mask, j] < 0).double().mean()) if mask.any() else None,
                path_margin_shortfall=float(torch.relu(margin-rule["sign"]*path[:, mask, j]).mean()) if mask.any() else None,
                risk_margin_shortfall=float(torch.relu(margin-rule["sign"]*gradient[mask, j]).mean()) if mask.any() else None))
        return dict(n_rows=len(z), finite=True, risk_gradient_finite=True,
            time_gradient_shape=list(path.shape), time_bin_indices=self._indices(int(self.config["prior_time_samples"])),
            attribution_target="calibrated log interval cumulative hazard; not hazard probability",
            risk_target="risk_scale * negative RMST(tau)/tau before positive fitting-reference z-score",
            risk_horizon=self.risk_horizon_, risk_scale=self.model.risk_scale,
            ph_time_smoothness="theoretically zero: shared covariate log-rate gradient at every time" if self.model.is_ph else None,
            rules=records, stage="after calibration; diagnostic only, not used to select settings")


def fit_improved(method, c, x, t, e, query, grid, seed, split_seed):
    from .adapters import Prediction
    from .data import duplicate_groups, neural_partitions, preprocess
    core, stop, cal = neural_partitions(e, split_seed, True, groups=duplicate_groups(x, t, e))
    (xf, xs, xc, xq), transformer = preprocess(x.iloc[core], x.iloc[stop], x.iloc[cal], query, return_transformer=True)
    names = list(x.columns)
    if xf.shape[1] != len(names):
        raise ValueError("Improved named-feature mapping changed")
    priors = json.loads((Path(__file__).resolve().parents[1]/"priors.json").read_text(encoding="utf-8"))
    source = priors[x.attrs["dataset"]]
    model = ImprovedModel(method, c, source["rules"], names, random_state=seed)
    model.fit_improved(xf, np.asarray(t)[core], np.asarray(e)[core],
                       (xs, np.asarray(t)[stop], np.asarray(e)[stop]))
    model._fit_calibration((xc, np.asarray(t)[cal], np.asarray(e)[cal]))
    diagnostics = dict(fitting=model.final_prior_diagnostics(xf), query=model.final_prior_diagnostics(xq))
    reference, raw_risk = model.predict_risk(xf), model.predict_risk(xq)
    center, scale = float(reference.mean()), max(float(reference.std()), 1e-8)
    risk = (raw_risk-center)/scale
    survival = model.predict_survival(xq, times=grid)
    architecture = dict(native_head="log interval cumulative hazard", contrast_bound=model.model.contrast_bound,
        baseline_shift_bound=model.model.baseline_shift_bound, calibration_temperature_bounds=[.2, 5.],
        calibration_bias_bounds=[-3., 3.], nonlinear_head_initialization_std=.01,
        raw_input_linear_skip=model.model.has_skip, proportional_hazards=model.model.is_ph,
        bounded_contrasts="explicit tanh architecture, not post-hoc prediction repair")
    if c.get("_checkpoint"):
        torch.save(dict(state=model.model.state_dict(), config=dict(c), method=method, scaler=model.scaler,
            preprocessor=transformer, feature_names=names, time_bins=model.time_bins,
            risk_horizon=model.risk_horizon_, brier_grid=model.brier_grid_,
            baseline_reference=model.baseline_reference_, rule_reference=model.rule_reference_,
            baseline_initialization=model.baseline_initialization_, prior_source=source,
            risk_center=center, risk_scale=scale, architecture=architecture,
            fitting_censor_times=model.censoring_.times, fitting_censor_survival=model.censoring_.values), c["_checkpoint"])
    return Prediction(risk, survival, dict(seed=seed, n_fit=len(core), n_stop=len(stop), n_calibration=len(cal),
        fit_rows=core.tolist(), stop_rows=stop.tolist(), calibration_rows=cal.tolist(), recipe=method,
        prior_status=source, attribution_prior_training_enabled=model.has_prior,
        directional_prior_training_enabled=model.has_prior and bool(model.rules),
        prior_audit=model.prior_audit, training_trace=model.trace, best_epoch=model.best_epoch_,
        final_prior_diagnostics=diagnostics, calibration=model.calibration_record,
        attribution_target="log interval cumulative hazard; changed development hypothesis, not original hazard IG",
        structural_prior="normalized time differences of log-rate Phi plus separately weighted L1 Phi magnitude",
        ph_time_smoothing="identically zero in exact PH; magnitude/available directional priors can remain active" if model.model.is_ph else None,
        prior_warmup=int(c.get("prior_warmup", 10)), architecture=architecture,
        baseline_reference=model.baseline_reference_.tolist(), rule_reference_median=model.rule_reference_["median"].tolist(),
        baseline_initialization=model.baseline_initialization_, risk_horizon=model.risk_horizon_,
        risk_target="same calibrated curve: risk_scale * negative RMST(fitting-supported q80)/horizon",
        fitting_brier_grid=model.brier_grid_.tolist(), fitting_censor_grid_survival=model.grid_g_.tolist(),
        risk_normalization=dict(center=center, scale=scale, source="fitting-reference predictions only"),
        ipcw="core-only censoring KM; event G(T-), alive G(u); grid restricted to G>=0.05",
        pair_sampling="Breslow Cox risk sets within shuffled minibatches; same curve-derived risk",
        implementation="new native log-rate PCH head; full curve/prior variant, not strict original-method reproduction"))
