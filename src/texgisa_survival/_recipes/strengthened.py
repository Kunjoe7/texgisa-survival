"""Development variants with a positive, fitting-scaled point-time head.

Point estimates are auxiliary conditional estimates trained with event absolute
error / censoring lower bounds, not extrapolated PCH means. Domain directions
are sourced hypotheses, not clinically confirmed knowledge. The frozen package
and earlier campaign implementations are deliberately unchanged.
"""
from pathlib import Path
import copy
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .improved import (ImprovedModel, _LogRateView, checkpoint_eligible,
    fitting_baseline, pch_log_rate_nll, survival_from_log_rates, truncated_rmst)
from .losses import smooth_concordance
from .metrics import CensoringKM, evaluation_grid


METHODS = {"StrengthBrier", "StrengthPoint", "StrengthCox", "StrengthPointNoPrior"}


def strengthened_partitions(x, times, events, split_seed, config):
    """Keep calibration fixed while moving stage-one stop rows into fresh refit."""
    from .data import duplicate_groups, neural_partitions
    original_core, original_stop, cal = neural_partitions(events, split_seed, True,
        groups=duplicate_groups(x, times, events))
    refit = "_stage_epochs" in config
    core = np.sort(np.r_[original_core, original_stop]) if refit else original_core
    stop = np.array([], dtype=int) if refit else original_stop
    return core, stop, cal, original_core, original_stop


def fitting_time_scale(times):
    """Original-unit scale and a strictly positive median initialization."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Point-time fitting values must be finite and nonnegative")
    positive = times[times > 0]
    if not len(positive):
        raise ValueError("A positive fitting time is required")
    median = float(np.median(times))
    iqr = float(np.quantile(times, .75)-np.quantile(times, .25))
    scale = max(median, iqr, float(np.median(positive))*.01)
    initial = max(median, scale*.01)
    ratio = initial/scale
    inverse_softplus = ratio+math.log(-math.expm1(-ratio))
    return dict(scale=scale, initial_time=initial, initial_logit=inverse_softplus,
                observed_median=median, observed_iqr=iqr,
                source="optimization fitting observations only; initialization is not a survival median estimate")


def stable_point_loss(point_time, times, events, scale, sigma):
    """Dimensionless smooth MAE; censored observations impose lower bounds only."""
    if float(scale) <= 0 or float(sigma) <= 0:
        raise ValueError("Positive point scale and smoothing required")
    point_time = point_time.double()
    times = torch.as_tensor(times, dtype=point_time.dtype, device=point_time.device)
    events = torch.as_tensor(events, dtype=point_time.dtype, device=point_time.device)
    residual = (point_time-times)/float(scale)
    residual = torch.where(events == 1, residual, torch.relu(-residual))
    # hypot form avoids catastrophic cancellation in the loss derivative.
    return (torch.hypot(residual, torch.full_like(residual, float(sigma)))-float(sigma)).mean()


class StrengthNetwork(nn.Module):
    """Shared time contrast with optional interval residuals; exact PH for Cox."""
    def __init__(self, features, config, method, initial_rates, edges, horizon, point_scale):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown strengthened method")
        self.method, self.is_ph = method, method == "StrengthCox"
        self.has_skip = True
        self.risk_scale = float(config.get("risk_scale", 5.))
        self.contrast_bound = float(config.get("point_contrast_bound", 20.))
        self.residual_bound = float(config.get("residual_bound", 4.))
        self.baseline_shift_bound = float(config.get("baseline_shift_bound", 4.))
        if min(self.risk_scale, self.contrast_bound, self.residual_bound, self.baseline_shift_bound) <= 0:
            raise ValueError("Architectural bounds and risk scale must be positive")
        layers, previous = [], features
        for width in config["hidden"]:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.SELU(),
                           nn.Dropout(config["dropout"])])
            previous = width
        self.features = nn.Sequential(*layers)
        self.time_head = nn.Linear(previous, 1)
        self.time_skip = nn.Linear(features, 1, bias=False)
        nn.init.normal_(self.time_head.weight, std=.01)
        nn.init.zeros_(self.time_head.bias); nn.init.zeros_(self.time_skip.weight)
        self.residual_head = None if self.is_ph else nn.Linear(previous, len(initial_rates))
        if self.residual_head is not None:
            nn.init.normal_(self.residual_head.weight, std=.01)
            nn.init.zeros_(self.residual_head.bias)
        self.baseline_delta = nn.Parameter(torch.zeros(len(initial_rates)))
        self.register_buffer("initial_log_rates", torch.as_tensor(initial_rates, dtype=torch.float32))
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float64))
        self.register_buffer("horizon", torch.tensor(float(horizon), dtype=torch.float64))
        self.register_buffer("point_scale", torch.tensor(point_scale["scale"], dtype=torch.float64))
        self.register_buffer("point_origin", torch.tensor(point_scale["initial_logit"], dtype=torch.float64))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))

    def latent_components(self, x):
        hidden = self.features(x)
        raw = (self.time_head(hidden)+self.time_skip(x)).squeeze(1)
        point = self.contrast_bound*torch.tanh(raw/self.contrast_bound)
        residual = (torch.zeros((len(x), len(self.initial_log_rates)), dtype=x.dtype, device=x.device)
                    if self.residual_head is None else
                    self.residual_bound*torch.tanh(self.residual_head(hidden)/self.residual_bound))
        baseline = self.initial_log_rates+self.baseline_shift_bound*torch.tanh(
            self.baseline_delta/self.baseline_shift_bound)
        return baseline, point, residual

    def from_components(self, components, temperature=None, bias=None):
        baseline, point, residual = components
        temperature = self.temperature if temperature is None else temperature
        bias = self.logit_bias if bias is None else bias
        time_contrast = point/temperature-bias
        log_rates = baseline[None, :]-time_contrast[:, None]+residual/temperature
        point_time = self.point_scale*F.softplus(self.point_origin+time_contrast.double())
        return log_rates, point_time, time_contrast

    def components(self, x):
        baseline, point, residual = self.latent_components(x)
        return baseline, -point[:, None]+residual

    def log_rates(self, x, calibrated=True):
        components = self.latent_components(x)
        return self.from_components(components, None if calibrated else 1.,
                                    None if calibrated else 0.)[0]

    def raw_logits(self, x):
        return self.log_rates(x, calibrated=False)

    def point_time(self, x):
        return self.from_components(self.latent_components(x))[1]

    def point_risk(self, x):
        # Larger risk means shorter predicted time; no query-derived scaling.
        return -self.risk_scale*self.point_time(x)/self.point_scale

    def risk_from_log_rates(self, log_rates):
        return -self.risk_scale*truncated_rmst(log_rates, self.edges, self.horizon)/self.horizon

    def risk(self, x):
        components = self.latent_components(x)
        rates, point_time, contrast = self.from_components(components)
        if self.method in {"StrengthPoint", "StrengthPointNoPrior"}:
            return -self.risk_scale*point_time/self.point_scale
        if self.is_ph:
            # PH scalar is exactly the same ordering as every curve and point;
            # unlike RMST, its Cox gradient does not saturate at the horizon.
            return -self.risk_scale*contrast
        return self.risk_from_log_rates(rates)

    def hazards_and_risk(self, x):
        return -torch.expm1(-self.log_rates(x).double().exp()), self.risk(x)

    def forward(self, x):
        return self.hazards_and_risk(x)[0]


class StrengthenedModel(ImprovedModel):
    def __init__(self, method, config, rules, feature_names, random_state=0):
        if method not in METHODS:
            raise ValueError("Unknown strengthened method")
        super().__init__("ImproveResidual", config, rules, feature_names, random_state)
        self.method = method
        self.has_prior = method != "StrengthPointNoPrior"
        self.brier_weight = float(config.get("lambda_brier", 2.))
        self.point_weight = float(config.get("lambda_point", 1.))
        self.pair_weight = float(config.get("lambda_pair", 1.)) if method in {
            "StrengthPoint", "StrengthPointNoPrior"} else 0.

    def _create_network(self, features, initial, edges, horizon):
        """Factory hook for isolated exploratory heads; default stays unchanged."""
        return StrengthNetwork(features, self.config, self.method, initial,
                               edges, horizon, self.point_scaling_)

    def _begin_epoch(self, epoch):
        self._prior_multiplier = 1.

    def _prepare_prior_multiplier(self, predictive, prior):
        return self._prior_multiplier

    def architecture_record(self, default):
        return default

    def extra_metadata(self):
        return {}

    def _prepare_fitting_features(self, xt, times, events):
        """Optional fit-only state preparation before constructing a network."""
        pass

    def _checkpoint_objective(self, x, times, events):
        """Default criterion; isolated variants may protect a ranking head."""
        return self.predictive_loss(x, times, events)[0]

    def _clip_gradients(self):
        """Default global clipping, unchanged for inherited methods."""
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

    def _point_loss(self, point, times, events):
        return stable_point_loss(point, times, events, self.point_scaling_["scale"],
                                 self.config.get("point_sigma", .03))

    def _nll(self, representation, times, events):
        return pch_log_rate_nll(representation, times, events, self.time_bins)

    def predictive_loss(self, x, times, events):
        components = self.model.latent_components(x)
        log_rates, point_time, contrast = self.model.from_components(components)
        nll = pch_log_rate_nll(log_rates, times, events, self.time_bins)
        point = stable_point_loss(point_time, times, events,
            self.point_scaling_["scale"], self.config.get("point_sigma", .03))
        if self.method in {"StrengthPoint", "StrengthPointNoPrior"}:
            risk = -self.model.risk_scale*point_time/self.model.point_scale
        elif self.model.is_ph:
            risk = -self.model.risk_scale*contrast
        else:
            risk = self.model.risk_from_log_rates(log_rates)
        cox = self._cox_loss(risk, times, events)
        pair = smooth_concordance(times, events, point_time/self.model.point_scale,
            sigma=float(self.config.get("point_sigma", .03))) if self.pair_weight else nll*0.
        brier = self._brier(log_rates, times, events) if self.brier_weight else nll*0.
        total = (nll+self.point_weight*point+self.pair_weight*pair+
                 float(self.config.get("lambda_cox", .3))*cox+self.brier_weight*brier)
        return total, dict(pch=nll, point_smooth_mae=point, point_pair=pair,
                           ranking_cox=cox, ipcw_brier=brier)

    def prior_loss(self, x):
        # Base implementation constrains real log-rate paths and the actual
        # reported ranking head. Brier/Cox additionally audit the time head.
        total, record = super().prior_loss(x)
        record["point_direction"] = 0.
        if not self.has_prior or self.method in {"StrengthPoint", "StrengthPointNoPrior"}:
            return total, record
        subset = x[:int(self.config["prior_rows"])].detach().clone().requires_grad_(True)
        training = self.model.training; self.model.eval()
        try:
            gradient = torch.autograd.grad(self.model.point_risk(subset).sum(), subset, create_graph=True)[0]
            penalty = gradient.sum()*0.
            margin = float(self.config.get("direction_margin", .001))
            for rule in self.rules:
                j = self.feature_names.index(rule["feature"])
                mask = (subset[:, j] > self.rule_reference_["median"][j]
                        if rule["condition"] == "above_training_median"
                        else torch.ones(len(subset), dtype=torch.bool))
                if mask.any():
                    penalty = penalty+float(rule.get("weight", 1.))*torch.relu(
                        margin-rule["sign"]*gradient[mask, j]).mean()
            record["point_direction"] = float(penalty.detach())
            return total+float(self.config["lambda_expert"])*penalty, record
        finally:
            self.model.train(training)

    def fit_strengthened(self, x, times, events, stop=None):
        fixed = self.config.get("_stage_epochs")
        epochs = int(self.config["epochs"] if fixed is None else fixed)
        warmup_epochs = max(1, int(self.config.get("prior_warmup", 10)))
        if epochs < warmup_epochs or (fixed is not None and float(fixed) != epochs):
            raise ValueError("Fixed epoch count must be integral and reach full prior warmup")
        if fixed is None and (stop is None or len(stop[0]) == 0):
            raise ValueError("Stage one requires reserved stopping observations")
        self.scaler.fit(x)
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        tt, ee = torch.as_tensor(times, dtype=torch.float64), torch.as_tensor(events, dtype=torch.float64)
        if fixed is None:
            xs = torch.as_tensor(self.scaler.transform(stop[0]), dtype=torch.float32)
            st, se = torch.as_tensor(stop[1], dtype=torch.float64), torch.as_tensor(stop[2], dtype=torch.float64)
        self.time_bins = self._make_time_bins(np.asarray(times), np.asarray(events))
        self.num_time_bins = len(self.time_bins)-1
        self.brier_grid_ = evaluation_grid(times, events, points=int(self.config.get("brier_points", 24)))
        quantile = float(self.config.get("horizon_quantile", .8))
        if not 0 < quantile <= 1:
            raise ValueError("Risk horizon quantile must lie in (0, 1]")
        self.risk_horizon_ = min(float(np.quantile(times, quantile)), float(self.time_bins[-1]))
        if self.risk_horizon_ <= 0:
            raise ValueError("Risk horizon must be positive")
        self.censoring_ = CensoringKM(times, events)
        self.grid_g_ = self.censoring_.predict(self.brier_grid_)
        self.point_scaling_ = fitting_time_scale(times)
        initial, self.baseline_initialization_ = fitting_baseline(times, events, self.time_bins)
        self._prepare_fitting_features(xt, tt, ee)
        self.model = self._create_network(xt.shape[1], initial, self.time_bins, self.risk_horizon_)
        self.baseline_reference_ = np.quantile(xt.numpy(), .98, axis=0).astype(np.float32)
        self.rule_reference_ = {"median": np.median(xt.numpy(), axis=0)}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["wd"])
        best, state, best_epoch, stale = float("inf"), None, -1, 0
        for epoch in range(epochs):
            self._begin_epoch(epoch)
            self.model.train(); records = []
            warmup = min(1., (epoch+1)/warmup_epochs)
            for ids in torch.randperm(len(xt)).split(int(self.config["batch"])):
                optimizer.zero_grad()
                predictive, parts = self.predictive_loss(xt[ids], tt[ids], ee[ids])
                prior, diagnostics = self.prior_loss(xt[ids])
                multiplier = self._prepare_prior_multiplier(predictive, prior)
                total = predictive+warmup*multiplier*prior
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite strengthened fitting objective")
                if not self.prior_audit:
                    prior_norm = self._gradient_norm(prior, retain_graph=True) if self.has_prior else 0.
                    self.prior_audit["initial"] = dict(
                        predictive_parameter_gradient_norm=self._gradient_norm(predictive, retain_graph=True),
                        prior_parameter_gradient_norm=prior_norm,
                        effective_prior_gradient_norm=warmup*multiplier*prior_norm,
                        prior_multiplier=multiplier,
                        prior_warmup_factor=warmup, **diagnostics)
                total.backward()
                if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                    raise FloatingPointError("Nonfinite strengthened objective gradient")
                norm = self._clip_gradients()
                optimizer.step()
                records.append(dict(total=float(total.detach()), predictive=float(predictive.detach()),
                    raw_weighted_prior=float(prior.detach()), effective_prior=float((warmup*multiplier*prior).detach()),
                    prior_multiplier=multiplier,
                    gradient_norm_before_clip=float(norm), **{k: float(v.detach()) for k, v in parts.items()}, **diagnostics))
            self.model.eval()
            validation = None
            if fixed is None:
                with torch.no_grad():
                    validation = float(self._checkpoint_objective(xs, st, se))
                if not np.isfinite(validation):
                    raise FloatingPointError("Nonfinite strengthened stopping objective")
            eligible = checkpoint_eligible(epoch, warmup_epochs)
            self.trace.append(dict(epoch=epoch, prior_warmup_factor=warmup, checkpoint_eligible=eligible,
                validation_predictive_objective=validation,
                **{key: float(np.mean([r[key] for r in records])) for key in records[0]}))
            if fixed is not None:
                best_epoch = epoch
                continue
            if not eligible:
                continue
            if validation < best:
                best, state, best_epoch, stale = validation, copy.deepcopy(self.model.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= int(self.config["patience"]):
                break
        if fixed is None:
            if state is None:
                raise FloatingPointError("No eligible strengthened checkpoint")
            self.model.load_state_dict(state)
        self.model.eval(); self.is_fitted = True
        self.best_epoch_, self.actual_epoch_count_ = best_epoch, len(self.trace)
        self.best_epochs_ = best_epoch+1
        self.prior_audit["best_checkpoint"] = self._audit_best(xt, tt, ee, best_epoch)
        self.training_data_sample = xt[:min(200, len(xt))].detach()

    def _fit_calibration(self, validation_data, steps=None):
        x, times, events = validation_data
        self.model.eval()
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            components = tuple(v.detach() for v in self.model.latent_components(xt))
        log_t = self.model.temperature.detach().log().clone().requires_grad_(True)
        bias = self.model.logit_bias.detach().clone().requires_grad_(True)
        brier_weight = float(self.config.get("calibration_brier_weight", self.brier_weight))
        def objective():
            rates, point, _ = self.model.from_components(components, log_t.exp(), bias)
            nll = self._nll(rates, times, events)
            brier = self._brier(rates, times, events) if brier_weight else nll*0.
            point_loss = self._point_loss(point, times, events)
            return nll+brier_weight*brier+self.point_weight*point_loss, nll, brier, point_loss
        with torch.no_grad():
            initial = tuple(float(v) for v in objective())
        if not np.isfinite(initial).all():
            raise FloatingPointError("Nonfinite initial strengthened calibration")
        best, values, best_t, best_b, accepted = initial[0], initial, float(log_t.detach()), float(bias.detach()), 0
        optimizer = torch.optim.Adam([log_t, bias], lr=.03)
        reason = "completed projected optimization"
        for _ in range(int(self.config.get("calibration_steps", 60) if steps is None else steps)):
            optimizer.zero_grad(); loss = objective()[0]
            if not torch.isfinite(loss):
                reason = "nonfinite calibration iterate rejected"; break
            loss.backward()
            if not torch.isfinite(log_t.grad) or not torch.isfinite(bias.grad):
                reason = "nonfinite calibration gradient rejected"; break
            optimizer.step()
            with torch.no_grad():
                log_t.clamp_(math.log(.2), math.log(5.)); bias.clamp_(-3., 3.)
                candidate = tuple(float(v) for v in objective())
                if np.isfinite(candidate).all() and candidate[0] < best:
                    best, values, best_t, best_b = candidate[0], candidate, float(log_t), float(bias)
                    accepted += 1
        with torch.no_grad():
            self.model.temperature.fill_(math.exp(best_t)); self.model.logit_bias.fill_(best_b)
        self.calibration_record = dict(status="fitted" if accepted else "initial_retained",
            objective="reserved-calibration PCH NLL + fitting-KM IPCW IBS + censor-aware scaled point loss",
            initial_objective=initial[0], final_objective=values[0],
            initial_nll=initial[1], final_nll=values[1], initial_brier=initial[2], final_brier=values[2],
            initial_point_loss=initial[3], final_point_loss=values[3],
            brier_weight=brier_weight, point_weight=self.point_weight,
            temperature=math.exp(best_t), bias=best_b, accepted_improvements=accepted,
            stop_reason=reason, n_calibration=len(times),
            parameterization="same positive temperature and scalar risk shift for curve/time heads; PH order preserved")

    def predict_point_time(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model.point_time(self._x_tensor(x)).numpy()

    def final_prior_diagnostics(self, x):
        record = super().final_prior_diagnostics(x)
        self.model.eval()
        z = self._x_tensor(x[:int(self.config["prior_rows"])]).requires_grad_(True)
        gradient = torch.autograd.grad(self.model.point_risk(z).sum(), z)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite final point-time risk gradient")
        for row, rule in zip(record["rules"], self.rules):
            j = self.feature_names.index(rule["feature"])
            mask = (z[:, j] > self.rule_reference_["median"][j]
                    if rule["condition"] == "above_training_median" else torch.ones(len(z), dtype=torch.bool))
            row["point_risk_sign_violation"] = float((rule["sign"]*gradient[mask, j] < 0).double().mean()) if mask.any() else None
        record.update(point_gradient_finite=True, risk_target=self.risk_target(),
                      point_target="negative original-unit point time divided by fitting time scale")
        return record

    def risk_target(self):
        if self.method in {"StrengthPoint", "StrengthPointNoPrior"}:
            return "negative fitting-scaled calibrated auxiliary point time"
        if self.method == "StrengthCox":
            return "shared PH log-hazard contrast, inverse monotone to the same point-time head"
        return "same calibrated PCH curve: negative RMST at fixed fitting-derived horizon quantile"


def fit_strengthened(method, c, x, t, e, query, grid, seed, split_seed, *, _model_class=None, _return_fitted=False):
    """One charged fit only: stage selection OR a fresh fixed-epoch refit."""
    from .adapters import Prediction
    from .data import preprocess
    refit = "_stage_epochs" in c
    core, stop, cal, original_core, original_stop = strengthened_partitions(x, t, e, split_seed, c)
    # Never ask sklearn to transform an empty stopping partition.
    frames = [x.iloc[cal], query] if refit else [x.iloc[stop], x.iloc[cal], query]
    arrays, transformer = preprocess(x.iloc[core], *frames, return_transformer=True)
    if refit:
        xf, xc, xq = arrays; stopping = None
    else:
        xf, xs, xc, xq = arrays
        stopping = xs, np.asarray(t)[stop], np.asarray(e)[stop]
    names = list(x.columns)
    if xf.shape[1] != len(names):
        raise ValueError("Strengthened named feature mapping changed")
    prior_source = json.loads((Path(__file__).resolve().parents[1]/"priors.json").read_text(encoding="utf-8"))[x.attrs["dataset"]]
    model_class = StrengthenedModel if _model_class is None else _model_class
    model = model_class(method, c, prior_source["rules"], names, random_state=seed)
    model.fit_strengthened(xf, np.asarray(t)[core], np.asarray(e)[core], stopping)
    model._fit_calibration((xc, np.asarray(t)[cal], np.asarray(e)[cal]))
    diagnostics = dict(fitting=model.final_prior_diagnostics(xf), query=model.final_prior_diagnostics(xq))
    reference, raw_risk = model.predict_risk(xf), model.predict_risk(xq)
    center, scale = float(reference.mean()), max(float(reference.std()), 1e-8)
    risk, survival = (raw_risk-center)/scale, model.predict_survival(xq, times=grid)
    point = model.predict_point_time(xq)
    if not np.isfinite(point).all() or (point <= 0).any():
        raise FloatingPointError("Invalid positive original-unit point prediction")
    architecture = dict(native_head="log interval cumulative hazard with shared negative time contrast",
        point_head="fitting_scale * softplus(fitting_median_origin + bounded_time_contrast/temperature - bias)",
        point_contrast_bound=model.model.contrast_bound, residual_bound=model.model.residual_bound,
        baseline_shift_bound=model.model.baseline_shift_bound, proportional_hazards=model.model.is_ph,
        interval_residual_head=not model.model.is_ph, raw_input_time_skip=True,
        calibration_temperature_bounds=[.2, 5.], calibration_bias_bounds=[-3., 3.])
    architecture = model.architecture_record(architecture)
    metadata = dict(seed=seed, recipe=method, n_fit=len(core), n_stop=len(stop), n_calibration=len(cal),
        fit_rows=core.tolist(), stop_rows=stop.tolist(), calibration_rows=cal.tolist(),
        original_core_rows=original_core.tolist(), original_stop_rows=original_stop.tolist(),
        training_stage="fresh_fixed_epoch_core_plus_stop_refit" if refit else "core_stop_epoch_selection",
        best_epochs=model.best_epochs_, actual_epoch_count=model.actual_epoch_count_, best_epoch=model.best_epoch_,
        stage_epoch_budget=int(c["_stage_epochs"]) if refit else None,
        point_time_prediction=point.tolist(), point_time_definition="auxiliary positive conditional point estimate; not PCH full mean/median",
        point_time_units="original target units", point_scaling=model.point_scaling_,
        point_loss="scaled smooth event absolute error; censored lower-bound shortfall; not IPCW MAE",
        prior_status=prior_source, attribution_prior_training_enabled=model.has_prior,
        directional_prior_training_enabled=model.has_prior and bool(model.rules),
        prior_audit=model.prior_audit, training_trace=model.trace, calibration=model.calibration_record,
        final_prior_diagnostics=diagnostics, architecture=architecture,
        attribution_target="calibrated log interval cumulative hazard TEXGI, distinct from point-time/ranking gradients",
        structural_prior="time differences of log-rate Phi and separately weighted L1 Phi; domain directions are sourced hypotheses",
        ph_time_smoothing="identically zero in exact PH; magnitude and available directional priors remain active" if model.model.is_ph else None,
        baseline_reference=model.baseline_reference_.tolist(), rule_reference_median=model.rule_reference_["median"].tolist(),
        baseline_initialization=model.baseline_initialization_, risk_horizon=model.risk_horizon_,
        horizon_quantile=float(c.get("horizon_quantile", .8)), risk_target=model.risk_target(),
        fitting_brier_grid=model.brier_grid_.tolist(), fitting_censor_grid_survival=model.grid_g_.tolist(),
        risk_normalization=dict(center=center, scale=scale, source="optimization fitting predictions only"),
        implementation="new strengthened development hypotheses; not strict original-method reproduction")
    metadata.update(model.extra_metadata())
    if c.get("_checkpoint"):
        torch.save(dict(state=model.model.state_dict(), config=dict(c), method=method, scaler=model.scaler,
            preprocessor=transformer, feature_names=names, time_bins=model.time_bins,
            risk_horizon=model.risk_horizon_, brier_grid=model.brier_grid_, point_scaling=model.point_scaling_,
            baseline_reference=model.baseline_reference_, rule_reference=model.rule_reference_,
            baseline_initialization=model.baseline_initialization_, prior_source=prior_source,
            risk_center=center, risk_scale=scale, architecture=architecture,
            fitting_censor_times=model.censoring_.times, fitting_censor_survival=model.censoring_.values,
            metadata=metadata), c["_checkpoint"])
    if _return_fitted:
        return Prediction(risk, survival, metadata), model, transformer, center, scale
    return Prediction(risk, survival, metadata)
