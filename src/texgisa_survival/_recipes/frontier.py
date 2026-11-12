"""Gradient-isolated ranking and curve/time towers, a development hypothesis.

Curve fitting stops gradients at the ranking offset. Prediction and TEXGI use
the identical numerical function with its full real input/parameter gradient.
Ranking, native auxiliary point C and curve IBS are distinct reported outputs;
this is not a reproduction of a named prior multitask method.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .exploratory import ExploratoryModel, balanced_point_loss
from .improved import pch_log_rate_nll
from .losses import smooth_concordance
from .strengthened import fit_strengthened


METHODS = {"FrontierSplit", "FrontierUno", "FrontierEmbed", "FrontierSmooth", "FrontierSplitNoPrior"}


class NumericEmbedding(nn.Module):
    """Fixed smooth Fourier features of standardized raw inputs; raw retained."""
    def __init__(self, features, frequencies=3):
        super().__init__()
        if not 1 <= int(frequencies) <= 6:
            raise ValueError("Embedding frequencies must be in [1,6]")
        self.register_buffer("frequencies", torch.arange(1, int(frequencies)+1, dtype=torch.float32)/2.)
        self.output_features = features*(1+2*int(frequencies))

    def forward(self, x):
        phase = x[:, :, None]*self.frequencies
        return torch.cat([x, phase.sin().flatten(1), phase.cos().flatten(1)], 1)


def smooth_time_basis(edges, dimensions):
    """Fit-only centered cosine functions, not an interval-index penalty."""
    edges = np.asarray(edges, float)
    dimensions = int(dimensions)
    if dimensions not in {3, 5} or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Positive fitting intervals and 3/5 smooth components required")
    centers = (edges[1:]+edges[:-1])/2./edges[-1]
    basis = np.cos(np.pi*centers[:, None]*np.arange(1, dimensions+1)[None, :])
    weights = np.diff(edges)/edges[-1]
    basis -= weights@basis
    basis /= np.maximum(np.max(np.abs(basis), axis=0, keepdims=True), 1e-12)
    return basis.astype(np.float32)


def _tower(features, config):
    layers, previous = [], features
    for width in config["hidden"]:
        layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.SELU(), nn.Dropout(config["dropout"])])
        previous = width
    return nn.Sequential(*layers)


class FrontierNetwork(nn.Module):
    def __init__(self, features, config, method, initial_rates, edges, horizon, point_scaling):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown frontier variant")
        self.method, self.is_ph, self.has_skip = method, False, True
        self.risk_scale = float(config.get("risk_scale", 5.))
        self.rank_bound = float(config.get("rank_contrast_bound", 8.))
        self.contrast_bound = float(config.get("point_contrast_bound", 20.))
        self.residual_bound = float(config.get("residual_bound", 4.))
        self.baseline_shift_bound = float(config.get("baseline_shift_bound", 4.))
        if min(self.risk_scale, self.rank_bound, self.contrast_bound, self.residual_bound, self.baseline_shift_bound) <= 0:
            raise ValueError("Positive architectural bounds required")
        self.embedding = NumericEmbedding(features, config.get("embedding_frequencies", 3)) if method == "FrontierEmbed" else nn.Identity()
        input_features = self.embedding.output_features if method == "FrontierEmbed" else features
        # Separate instances are essential: detaching only the offset while
        # sharing an encoder would still let curve/point gradients alter rank.
        self.rank_features = _tower(input_features, config)
        self.curve_features = _tower(input_features, config)
        self.rank_head, self.point_head = nn.Linear(config["hidden"][-1], 1), nn.Linear(config["hidden"][-1], 1)
        self.rank_skip, self.point_skip = nn.Linear(features, 1, bias=False), nn.Linear(features, 1, bias=False)
        self.smooth_dimensions = int(config.get("temporal_rank", config.get("smooth_basis_dim", 3))) if method == "FrontierSmooth" else None
        outputs = self.smooth_dimensions or len(initial_rates)
        self.residual_head = nn.Linear(config["hidden"][-1], outputs)
        for head in [self.rank_head, self.point_head, self.residual_head]:
            nn.init.normal_(head.weight, std=.01); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.rank_skip.weight); nn.init.zeros_(self.point_skip.weight)
        self.baseline_delta = nn.Parameter(torch.zeros(len(initial_rates)))
        self.offset_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("initial_log_rates", torch.as_tensor(initial_rates, dtype=torch.float32))
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float64))
        self.register_buffer("horizon", torch.tensor(float(horizon), dtype=torch.float64))
        self.register_buffer("point_scale", torch.tensor(point_scaling["scale"], dtype=torch.float64))
        self.register_buffer("point_origin", torch.tensor(point_scaling["initial_logit"], dtype=torch.float64))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))
        self.register_buffer("point_temperature", torch.ones(()))
        self.register_buffer("point_bias", torch.zeros(()))
        self.register_buffer("time_basis", torch.as_tensor(smooth_time_basis(edges, self.smooth_dimensions))
                             if self.smooth_dimensions else torch.empty((len(initial_rates), 0)))

    def rank_parameters(self):
        return tuple(p for module in [self.rank_features, self.rank_head, self.rank_skip] for p in module.parameters())

    def curve_parameters(self):
        rank_ids = {id(p) for p in self.rank_parameters()}
        return tuple(p for p in self.parameters() if id(p) not in rank_ids)

    def _rank(self, x, embedded):
        raw = (self.rank_head(self.rank_features(embedded))+self.rank_skip(x)).squeeze(1)
        return self.risk_scale*self.rank_bound*torch.tanh(raw/self.rank_bound)

    def risk(self, x):
        return self._rank(x, self.embedding(x))

    def latent_components(self, x):
        embedded = self.embedding(x)
        rank = self._rank(x, embedded)
        hidden = self.curve_features(embedded)
        raw_point = (self.point_head(hidden)+self.point_skip(x)).squeeze(1)
        point = self.contrast_bound*torch.tanh(raw_point/self.contrast_bound)
        raw_residual = self.residual_head(hidden)
        if self.smooth_dimensions:
            coefficients = self.residual_bound*torch.tanh(raw_residual/self.residual_bound)/self.smooth_dimensions
            residual = coefficients@self.time_basis.T
        else:
            residual = self.residual_bound*torch.tanh(raw_residual/self.residual_bound)
        baseline = self.initial_log_rates+self.baseline_shift_bound*torch.tanh(self.baseline_delta/self.baseline_shift_bound)
        return baseline, rank, point, residual

    def from_components(self, components, temperature=None, bias=None, *, point_temperature=None, point_bias=None, detach_rank=False):
        baseline, rank, point, residual = components
        temperature = self.temperature if temperature is None else temperature
        bias = self.logit_bias if bias is None else bias
        point_temperature = self.point_temperature if point_temperature is None else point_temperature
        point_bias = self.point_bias if point_bias is None else point_bias
        rank = rank.detach() if detach_rank else rank
        # Positive offset coefficient in (0,2); independent curve parameter.
        offset = 2.*torch.sigmoid(self.offset_logit)*rank/self.risk_scale
        rates = baseline[None, :]+offset[:, None]+(-point[:, None]+residual)/temperature+bias
        point_time = self.point_scale*F.softplus(self.point_origin+point.double()/point_temperature-point_bias)
        return rates, point_time, point

    def loss_representation(self, x):
        return self.from_components(self.latent_components(x), detach_rank=True)

    def log_rates(self, x, calibrated=True):
        # Real evaluation gradient, including the actual ranking-offset path.
        return self.from_components(self.latent_components(x), None if calibrated else 1., None if calibrated else 0.)[0]

    def raw_logits(self, x):
        return self.log_rates(x, calibrated=False)

    def point_time(self, x):
        return self.from_components(self.latent_components(x))[1]

    def point_risk(self, x):
        return -self.risk_scale*self.point_time(x)/self.point_scale

    def hazards_and_risk(self, x):
        return -torch.expm1(-self.log_rates(x).double().exp()), self.risk(x)

    def forward(self, x):
        return self.hazards_and_risk(x)[0]


class FrontierModel(ExploratoryModel):
    def __init__(self, method, config, rules, feature_names, random_state=0):
        if method not in METHODS:
            raise ValueError("Unknown frontier method")
        super().__init__("ExploreBalance", config, rules, feature_names, random_state)
        self.method, self.has_prior = method, method != "FrontierSplitNoPrior"
        self.routing_audit = {}

    def _create_network(self, features, initial, edges, horizon):
        return FrontierNetwork(features, self.config, self.method, initial, edges, horizon, self.point_scaling_)

    def _point_loss(self, point, times, events):
        return balanced_point_loss(point, times, events, self.point_scaling_["scale"],
            float(self.config.get("point_sigma", .05)), self.censor_weight)

    def _ranking_parts(self, x, times, events):
        risk = self.model.risk(x)
        weights = None
        if self.method == "FrontierUno":
            # Keep original float64 durations up to the exact KM left lookup.
            weights = torch.as_tensor(1./np.maximum(self._g(torch.as_tensor(times).detach().cpu().numpy()), .05)**2,
                                      dtype=risk.dtype, device=risk.device)
        pair = smooth_concordance(times, events, -risk/self.model.risk_scale,
            sigma=float(self.config.get("rank_sigma", .1)), ipcw=weights)
        cox = self._cox_loss(risk, times, events)
        return float(self.config.get("lambda_cox", .3))*cox+self.pair_weight*pair, cox, pair

    def _curve_parts(self, x, times, events):
        rates, point, _ = self.model.loss_representation(x)
        nll = pch_log_rate_nll(rates, times, events, self.time_bins)
        brier = self._brier(rates, times, events) if self.brier_weight else nll*0.
        point_loss = self._point_loss(point, times, events)
        return nll+self.brier_weight*brier+self.point_weight*point_loss, nll, brier, point_loss

    def predictive_loss(self, x, times, events):
        ranking, cox, pair = self._ranking_parts(x, times, events)
        curve, nll, brier, point = self._curve_parts(x, times, events)
        return ranking+curve, dict(ranking_cox=cox, ranking_pair=pair, pch=nll,
                                  ipcw_brier=brier, point_balanced=point,
                                  ranking_objective=ranking, curve_point_objective=curve)

    def _checkpoint_objective(self, x, times, events):
        # Fixed rank-only stopping rule. Curve/point losses cannot select rank.
        return self._cox_loss(self.model.risk(x), times, events)

    def _clip_gradients(self):
        # A global norm would let arbitrarily large curve gradients shrink the
        # ranking step despite detached objectives and disjoint parameters.
        rank = torch.nn.utils.clip_grad_norm_(self.model.rank_parameters(), 1.)
        curve = torch.nn.utils.clip_grad_norm_(self.model.curve_parameters(), 1.)
        return math.hypot(float(rank), float(curve))

    def prior_loss(self, x):
        total, diagnostics = super().prior_loss(x)
        diagnostics["point_direction"] = 0.
        if not self.has_prior:
            return total, diagnostics
        training = self.model.training; self.model.eval()
        try:
            z = x[:int(self.config["prior_rows"])].detach().clone().requires_grad_(True)
            gradient = torch.autograd.grad(self.model.point_risk(z).sum(), z, create_graph=True)[0]
            penalty = gradient.sum()*0.
            for rule in self.rules:
                j = self.feature_names.index(rule["feature"])
                mask = (z[:, j] > self.rule_reference_["median"][j]
                        if rule["condition"] == "above_training_median" else torch.ones(len(z), dtype=torch.bool))
                if mask.any():
                    penalty = penalty+float(rule.get("weight", 1.))*torch.relu(
                        float(self.config.get("direction_margin", .001))-rule["sign"]*gradient[mask, j]).mean()
            diagnostics["point_direction"] = float(penalty.detach())
            return total+float(self.config["lambda_expert"])*penalty, diagnostics
        finally:
            self.model.train(training)

    @staticmethod
    def _norm_on(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        if any(not torch.isfinite(g).all() for g in gradients if g is not None):
            raise FloatingPointError("Nonfinite frontier routing gradient")
        return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))

    def _audit_best(self, x, times, events, best_epoch):
        record = super()._audit_best(x, times, events, best_epoch)
        self.model.eval(); rows = slice(0, min(len(x), int(self.config["batch"])))
        a, t, e = x[rows], times[rows], events[rows]
        curve = self._curve_parts(a, t, e)[0]; ranking = self._ranking_parts(a, t, e)[0]
        rank_params, curve_params = self.model.rank_parameters(), self.model.curve_parameters()
        self.routing_audit = dict(curve_point_gradient_on_rank=self._norm_on(curve, rank_params),
            rank_objective_gradient_on_curve=self._norm_on(ranking, curve_params),
            rank_objective_gradient_on_rank=self._norm_on(ranking, rank_params),
            curve_point_gradient_on_curve=self._norm_on(curve, curve_params),
            actual_lograte_gradient_on_rank=self._norm_on(self.model.log_rates(a).sum(), rank_params),
            stage="selected/final checkpoint before independent calibration")
        if self.routing_audit["curve_point_gradient_on_rank"] != 0. or self.routing_audit["rank_objective_gradient_on_curve"] != 0.:
            raise RuntimeError("Frontier objective routing leaked between towers")
        record["routing_audit"] = self.routing_audit
        return record

    def _fit_calibration(self, validation_data, steps=None):
        x, times, events = validation_data
        self.model.eval()
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            components = tuple(v.detach() for v in self.model.latent_components(xt))
            rank_before = self.model.risk(xt).clone()
        count = int(self.config.get("calibration_steps", 60) if steps is None else steps)
        brier_weight = float(self.config.get("calibration_brier_weight", self.brier_weight))
        def optimize(which):
            temperature_buffer = self.model.temperature if which == "curve" else self.model.point_temperature
            bias_buffer = self.model.logit_bias if which == "curve" else self.model.point_bias
            log_t = temperature_buffer.detach().log().clone().requires_grad_(True)
            bias = bias_buffer.detach().clone().requires_grad_(True)
            def objective():
                if which == "curve":
                    rates = self.model.from_components(components, log_t.exp(), bias)[0]
                    nll = pch_log_rate_nll(rates, times, events, self.time_bins)
                    brier = self._brier(rates, times, events) if brier_weight else nll*0.
                    return nll+brier_weight*brier, nll, brier
                point = self.model.from_components(components, point_temperature=log_t.exp(), point_bias=bias)[1]
                loss = self._point_loss(point, times, events)
                return loss, loss, loss*0.
            with torch.no_grad(): initial = tuple(float(v) for v in objective())
            if not np.isfinite(initial).all():
                raise FloatingPointError("Nonfinite initial frontier "+which+" calibration")
            best, values, best_t, best_b, accepted = initial[0], initial, float(log_t.detach()), float(bias.detach()), 0
            optimizer = torch.optim.Adam([log_t, bias], lr=.03)
            reason = "completed projected optimization"
            for _ in range(count):
                optimizer.zero_grad(); loss = objective()[0]
                if not torch.isfinite(loss): reason = "nonfinite iterate rejected"; break
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
                temperature_buffer.fill_(math.exp(best_t)); bias_buffer.fill_(best_b)
            return dict(status="fitted" if accepted else "initial_retained", initial_objective=initial[0],
                final_objective=values[0], initial_nll=initial[1] if which == "curve" else None,
                final_nll=values[1] if which == "curve" else None,
                initial_brier=initial[2] if which == "curve" else None,
                final_brier=values[2] if which == "curve" else None,
                temperature=math.exp(best_t), bias=best_b, accepted_improvements=accepted, stop_reason=reason)
        curve, point = optimize("curve"), optimize("point")
        with torch.no_grad(): rank_after = self.model.risk(xt)
        if not torch.equal(rank_before, rank_after):
            raise RuntimeError("Independent calibration changed the ranking head")
        self.calibration_record = dict(status="independently_fitted_or_initial_retained", curve=curve, point=point,
            n_calibration=len(times), curve_objective="PCH NLL + fitting-KM IPCW IBS",
            point_objective="group-balanced censor-aware point loss only", rank_unchanged=True,
            brier_weight=brier_weight, parameterization="independent curve T/b and point T/b; ranking parameters and output unchanged")

    def risk_target(self):
        return "independent ranking-tower score; distinct from auxiliary point-time score and curve RMST"

    def final_prior_diagnostics(self, x):
        record = super().final_prior_diagnostics(x)
        record.update(attribution_target="actual calibrated log PCH interval cumulative hazard, including real ranking-offset input gradient",
                      point_target="negative independent curve-tower auxiliary point estimate / fitting scale",
                      explanation_detach_rank=False)
        return record

    def architecture_record(self, default):
        return dict(native_head="PCH log interval cumulative hazard with independent rank and curve/point towers",
            rank_point_parameter_tying=False, rank_to_curve_offset="positive coefficient in (0,2), divided by risk_scale",
            gradient_clipping="independent rank and curve parameter groups, each norm1; preclip trace is combined Euclidean norm",
            training_curve_rank_offset_detached=True, actual_prediction_and_texgi_offset_detached=False,
            point_contrast_bound=self.model.contrast_bound, rank_contrast_bound=self.model.rank_bound,
            residual_bound=self.model.residual_bound, baseline_shift_bound=self.model.baseline_shift_bound,
            smooth_basis_dimensions=self.model.smooth_dimensions,
            smooth_basis="fit-time-centered cosine, width-weighted zero mean" if self.model.smooth_dimensions else None,
            numerical_embedding="fixed sin/cos standardized raw input; raw skip retained" if self.method == "FrontierEmbed" else None,
            embedding_frequencies=self.config.get("embedding_frequencies", 3) if self.method == "FrontierEmbed" else None,
            calibration="four scalar curve T/b and point T/b, separately nonincreasing objectives; rank unchanged")

    def extra_metadata(self):
        return dict(recipe=self.method, selection_criterion="stop_ranking_cox_loss_after_full_prior_warmup",
            point_time_definition="auxiliary positive curve-tower estimate; not full survival mean or median",
            point_loss="group-balanced event smooth MAE + predeclared censor-weighted lower-bound smooth shortfall",
            point_censor_weight=self.censor_weight, routing_audit=self.routing_audit,
            rank_pair_weighting="fitting-only KM G(T-)^-2, fixed floor0.05" if self.method == "FrontierUno" else "strict comparable pairs, unweighted",
            rank_pair_sigma=float(self.config.get("rank_sigma", .1)),
            attribution_target="actual calibrated PCH log interval cumulative hazard; real rank-offset gradient retained",
            structural_prior="true q98-reference log-rate TEXGI time differences and magnitude; expert directions constrain log-rate, actual rank and point risk",
            risk_target=self.risk_target(), horizon_role="inherited fitting metadata only; reported risk is independent rank",
            smooth_time_basis=self.model.time_basis.tolist() if self.model.smooth_dimensions else None,
            implementation="new gradient-isolated development hypotheses; not a claimed reproduction or clinically confirmed prior")


def fit_frontier(method, c, x, t, e, query, grid, seed, split_seed):
    return fit_strengthened(method, c, x, t, e, query, grid, seed, split_seed, _model_class=FrontierModel)
