"""Isolated, predeclared exploratory mechanisms, never an original-paper claim.

PCH variants use an auxiliary positive point estimate and a true PCH curve.
ExploreLogTime instead uses an exact continuous shifted lognormal distribution
conditional on T>=0, with its native conditional median. All priors are actual
differentiable fitting objectives and sourced domain hypotheses, not confirmed
clinical knowledge. The public package and earlier experiments are untouched.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .improved import ImprovedModel, ipcw_brier
from .losses import smooth_concordance
from .strengthened import StrengthNetwork, StrengthenedModel, fit_strengthened


METHODS = {"ExploreBalance", "ExploreDense", "ExploreLogTime", "ExplorePrior", "ExploreBalanceNoPrior"}


def balanced_point_loss(point, times, events, scale, sigma, censor_weight=.1, *, log_time=False):
    """Separate event/censor means; prevalence cannot silently change weights.

    This deliberately changes the old all-observation objective. It is not an
    unbiased estimator of complete-data MAE under arbitrary censoring.
    """
    if scale <= 0 or sigma <= 0 or not 0 <= censor_weight <= 1:
        raise ValueError("Invalid point scaling, smoothing or censor coefficient")
    point = point.double()
    times = torch.as_tensor(times, dtype=point.dtype, device=point.device)
    events = torch.as_tensor(events, dtype=point.dtype, device=point.device)
    if point.ndim != 1 or point.shape != times.shape or times.shape != events.shape:
        raise ValueError("Point/time/event shape mismatch")
    if (times < 0).any() or ((events != 0) & (events != 1)).any():
        raise ValueError("Nonnegative times and binary events required")
    a, b = point/scale, times/scale
    if log_time:
        if (point < 0).any():
            raise ValueError("Log point error requires nonnegative predictions")
        a, b = torch.log1p(a), torch.log1p(b)
    residual = a-b
    smooth = lambda r: torch.hypot(r, torch.full_like(r, float(sigma)))-float(sigma)
    event_loss = (events*smooth(residual)).sum()/events.sum().clamp_min(1.)
    censored = 1.-events
    censor_loss = (censored*smooth(torch.relu(-residual))).sum()/censored.sum().clamp_min(1.)
    return event_loss+float(censor_weight)*censor_loss


def detached_prior_multiplier(predictive_norm, prior_norm, target=.05, lower=.2, upper=5.):
    """Scalar from detached full-parameter norms, bounded and epoch-frozen."""
    values = [predictive_norm, prior_norm, target, lower, upper]
    if not np.isfinite(values).all() or predictive_norm < 0 or prior_norm < 0 or not 0 < lower <= upper or target <= 0:
        raise ValueError("Invalid prior gradient ratio inputs")
    return float(np.clip(target*predictive_norm/max(prior_norm, 1e-12), lower, upper))


class DenseGatedFeatures(nn.Module):
    """Raw normalized inputs feed every block; learned gated cross features.

    Multiplicative input interactions and dense concatenation address feature
    interactions without relying on tree/boosting models. Output width equals
    the old final hidden width so the time/curve head itself is unchanged.
    """
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.input_norm = nn.LayerNorm(features)
        self.cross_projection = nn.Linear(features, features, bias=False)
        self.cross_gate = nn.Linear(features, features)
        self.blocks, self.gates = nn.ModuleList(), nn.ModuleList()
        previous = 2*features
        for width in hidden:
            self.blocks.append(nn.Sequential(nn.Linear(previous, width), nn.LayerNorm(width),
                                             nn.SELU(), nn.Dropout(dropout)))
            self.gates.append(nn.Linear(previous, width))
            previous = 2*features+width

    def forward(self, x):
        normalized = self.input_norm(x)
        cross = normalized*self.cross_projection(normalized)*torch.sigmoid(self.cross_gate(normalized))
        raw = torch.cat([x, cross], 1)
        value = raw
        for block, gate in zip(self.blocks, self.gates):
            value = block(value)*torch.sigmoid(gate(value))
            if block is not self.blocks[-1]:
                value = torch.cat([raw, value], 1)
        return value


def _distribution_parameters(parameters):
    parameters = parameters.double()
    if parameters.ndim != 2 or parameters.shape[1] != 2 or not torch.isfinite(parameters).all():
        raise ValueError("Lognormal parameters must be a finite (n,2) matrix")
    mu, sigma = parameters[:, 0], parameters[:, 1]
    if (sigma <= 0).any():
        raise ValueError("Lognormal sigma must be positive")
    return mu, sigma


def conditional_lognormal_log_survival(parameters, grid, delta):
    """Exact log survival, retaining tail information below exp underflow."""
    if float(delta) <= 0:
        raise ValueError("Fitting-only positive time translation required")
    mu, sigma = _distribution_parameters(parameters)
    grid = torch.as_tensor(grid, dtype=mu.dtype, device=mu.device)
    if grid.ndim != 1 or not torch.isfinite(grid).all():
        raise ValueError("Finite vector of prediction times required")
    log_s0 = torch.special.log_ndtr(-(math.log(float(delta))-mu)/sigma)
    shifted = grid.clamp_min(0.)+float(delta)
    log_s = torch.special.log_ndtr(-(shifted.log()[None, :]-mu[:, None])/sigma[:, None])
    # For u=0 the same arithmetic is used in numerator/denominator; S(0)=1.
    result = log_s-log_s0[:, None]
    return torch.where(grid[None, :] < 0, torch.zeros_like(result), result)


def conditional_lognormal_survival(parameters, grid, delta):
    """S(T>u | T>=0)=S_LN(u+delta)/S_LN(delta), with full right tail."""
    return conditional_lognormal_log_survival(parameters, grid, delta).exp()


def conditional_lognormal_nll(parameters, times, events, delta):
    """Exact density/censor survival, including the conditioning normalizer."""
    if float(delta) <= 0:
        raise ValueError("Fitting-only positive time translation required")
    mu, sigma = _distribution_parameters(parameters)
    times = torch.as_tensor(times, dtype=mu.dtype, device=mu.device)
    events = torch.as_tensor(events, dtype=mu.dtype, device=mu.device)
    if times.shape != mu.shape or events.shape != mu.shape or (times < 0).any():
        raise ValueError("Nonnegative aligned duration/event vectors required")
    if ((events != 0) & (events != 1)).any():
        raise ValueError("Binary events required")
    shifted = times+float(delta)
    z = (shifted.log()-mu)/sigma
    log_s = torch.special.log_ndtr(-z)
    log_density = -.5*z.square()-.5*math.log(2.*math.pi)-sigma.log()-shifted.log()
    log_s0 = torch.special.log_ndtr(-(math.log(float(delta))-mu)/sigma)
    return -(events*log_density+(1.-events)*log_s-log_s0).mean()


def conditional_lognormal_median(parameters, delta):
    mu, sigma = _distribution_parameters(parameters)
    log_s0 = torch.special.log_ndtr(-(math.log(float(delta))-mu)/sigma)
    probability = 1.-.5*log_s0.exp()
    if (probability >= 1.).any():
        raise FloatingPointError("Conditional quantile exceeds floating-point CDF support")
    quantile = torch.special.ndtri(probability)
    return torch.exp(mu+sigma*quantile)-float(delta)


def lognormal_log_hazard(parameters, knots, delta):
    """Exact instantaneous log hazard at real fitting knots, not PCH rates.

    Conditioning divides density and survival by the same S(delta), hence
    cancels from their ratio. Translation still appears in density Jacobian.
    """
    mu, sigma = _distribution_parameters(parameters)
    knots = torch.as_tensor(knots, dtype=mu.dtype, device=mu.device)
    shifted = knots+float(delta)
    if (shifted <= 0).any():
        raise ValueError("Log-hazard knots must be in translated positive support")
    z = (shifted.log()[None, :]-mu[:, None])/sigma[:, None]
    log_density = -.5*z.square()-.5*math.log(2.*math.pi)-sigma.log()[:, None]-shifted.log()[None, :]
    return log_density-torch.special.log_ndtr(-z)


class LogTimeNetwork(nn.Module):
    """Heteroscedastic conditional lognormal with explicit stable support.

    mu>log(delta)+.1 ensures S_LN(delta)>.5. This is an architectural lower
    location bound, not a post-hoc repair. sigma is in [.15,3] by sigmoid.
    """
    def __init__(self, features, config, edges, horizon, point_scale, initialization):
        super().__init__()
        self.is_ph, self.has_skip = False, True
        self.risk_scale = float(config.get("risk_scale", 5.))
        self.contrast_bound = float(config.get("logtime_contrast_bound", 20.))
        self.residual_bound, self.baseline_shift_bound = 0., 0.
        self.delta = float(initialization["delta"])
        if self.contrast_bound <= 0 or self.risk_scale <= 0:
            raise ValueError("Positive logtime architecture scales required")
        layers, previous = [], features
        for width in config["hidden"]:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.SELU(), nn.Dropout(config["dropout"])])
            previous = width
        self.features = nn.Sequential(*layers)
        self.location_head = nn.Linear(previous, 1)
        self.location_skip = nn.Linear(features, 1, bias=False)
        self.scale_head = nn.Linear(previous, 1)
        nn.init.normal_(self.location_head.weight, std=.01); nn.init.zeros_(self.location_head.bias)
        nn.init.zeros_(self.location_skip.weight)
        nn.init.normal_(self.scale_head.weight, std=.01); nn.init.zeros_(self.scale_head.bias)
        self.scale_offset = nn.Parameter(torch.zeros(()))
        self.register_buffer("location_origin", torch.tensor(initialization["location_origin"], dtype=torch.float64))
        self.register_buffer("scale_origin", torch.tensor(initialization["scale_origin"], dtype=torch.float64))
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float64))
        self.register_buffer("horizon", torch.tensor(float(horizon), dtype=torch.float64))
        self.register_buffer("point_scale", torch.tensor(point_scale["scale"], dtype=torch.float64))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))

    def latent_components(self, x):
        hidden = self.features(x)
        raw = (self.location_head(hidden)+self.location_skip(x)).squeeze(1)
        location = self.contrast_bound*torch.tanh(raw/self.contrast_bound)
        scale = self.scale_origin+self.scale_offset+2.*torch.tanh(self.scale_head(hidden).squeeze(1)/2.)
        return self.location_origin.expand(len(x)), location, scale

    def from_components(self, components, temperature=None, bias=None):
        origin, contrast, scale_raw = components
        temperature = self.temperature if temperature is None else temperature
        bias = self.logit_bias if bias is None else bias
        mu = math.log(self.delta)+.1+F.softplus(origin+contrast.double()/temperature-bias)
        sigma = .15+2.85*torch.sigmoid(scale_raw.double())
        parameters = torch.stack([mu, sigma], 1)
        point = conditional_lognormal_median(parameters, self.delta)
        return parameters, point, torch.log1p(point/self.point_scale)

    def distribution(self, x):
        return self.from_components(self.latent_components(x))[0]

    def point_time(self, x):
        return self.from_components(self.latent_components(x))[1]

    def point_risk(self, x):
        return -self.risk_scale*torch.log1p(self.point_time(x)/self.point_scale)

    def risk(self, x):
        return self.point_risk(x)

    def log_rates(self, x, calibrated=True):
        # Name kept only for the shared TEXGI interface, not a PCH likelihood.
        components = self.latent_components(x)
        parameters = self.from_components(components, None if calibrated else 1., None if calibrated else 0.)[0]
        knots = (self.edges[1:]+self.edges[:-1])/2.
        return lognormal_log_hazard(parameters, knots, self.delta)

    def raw_logits(self, x):
        return self.log_rates(x, calibrated=False)

    def hazards_and_risk(self, x):
        # Descriptive interval failure probabilities from the actual continuous
        # curve, used for display only; fitting never treats them as PCH rates.
        log_survival = conditional_lognormal_log_survival(self.distribution(x), self.edges, self.delta)
        # exp(log hazard) is not substituted into the continuous likelihood.
        return -torch.expm1(log_survival[:, 1:]-log_survival[:, :-1]), self.risk(x)

    def forward(self, x):
        return self.hazards_and_risk(x)[0]


class ExploratoryModel(StrengthenedModel):
    def __init__(self, method, config, rules, feature_names, random_state=0):
        if method not in METHODS:
            raise ValueError("Unknown exploratory method")
        if config.get("event_weighting", "plain") != "plain":
            raise ValueError("Only the predeclared plain event-group mean is implemented")
        super().__init__("StrengthPoint", config, rules, feature_names, random_state)
        self.method = method
        self.has_prior = method != "ExploreBalanceNoPrior"
        self.pair_weight = float(config.get("lambda_pair", .5))
        self.censor_weight = float(config.get("point_censor_weight", .1))
        self._prior_multiplier = 1.
        self.multiplier_audit = []

    def _create_network(self, features, initial, edges, horizon):
        if self.method != "ExploreLogTime":
            network = StrengthNetwork(features, self.config, "StrengthPoint", initial, edges, horizon, self.point_scaling_)
            if self.method == "ExploreDense":
                network.features = DenseGatedFeatures(features, self.config["hidden"], self.config["dropout"])
            return network
        # These values are computed from the current optimization durations by
        # _make_time_bins; neither stop, calibration nor query labels are used.
        self.baseline_initialization_ = dict(self.logtime_initialization_)
        return LogTimeNetwork(features, self.config, edges, horizon, self.point_scaling_, self.logtime_initialization_)

    def _make_time_bins(self, times, events):
        edges = super()._make_time_bins(times, events)
        if self.method == "ExploreLogTime":
            positive = np.asarray(times)[np.asarray(times) > 0]
            if not len(positive):
                raise ValueError("Logtime requires a positive fitting duration")
            delta = max(float(np.median(positive))*1e-6, 1e-8)
            initial_time = max(float(np.median(times)), float(np.median(positive))*.01)
            distance = math.log(initial_time+delta)-math.log(delta)-.1
            location_origin = distance+math.log(-math.expm1(-distance))
            initial_sigma = float(np.clip(np.std(np.log(np.asarray(times)+delta)), .3, 2.))
            probability = (initial_sigma-.15)/2.85
            self.logtime_initialization_ = dict(delta=delta, initial_time=initial_time,
                location_origin=location_origin, scale_origin=math.log(probability/(1.-probability)),
                initial_sigma=initial_sigma, source="optimization fitting observations only",
                support="T>=0; shifted lognormal conditioned on T+delta>=delta",
                location_constraint="mu=log(delta)+.1+softplus(raw), so S_LN(delta)>.5")
        return edges

    def _point_loss(self, point, times, events):
        return balanced_point_loss(point, times, events, self.point_scaling_["scale"],
            float(self.config.get("point_sigma", .05)), self.censor_weight,
            log_time=self.method == "ExploreLogTime")

    def _nll(self, representation, times, events):
        if self.method == "ExploreLogTime":
            return conditional_lognormal_nll(representation, times, events, self.model.delta)
        return super()._nll(representation, times, events)

    def _brier(self, representation, times, events):
        if self.method != "ExploreLogTime":
            return super()._brier(representation, times, events)
        survival = conditional_lognormal_survival(representation, self.brier_grid_, self.model.delta)
        event_g = self._g(torch.as_tensor(times).detach().cpu().numpy())
        return ipcw_brier(survival, times, events, self.brier_grid_, event_g, self.grid_g_)

    def predictive_loss(self, x, times, events):
        representation, point, _ = self.model.from_components(self.model.latent_components(x))
        nll = self._nll(representation, times, events)
        point_loss = self._point_loss(point, times, events)
        normalized = point/self.model.point_scale
        if self.method == "ExploreLogTime":
            normalized = torch.log1p(normalized)
        risk = -self.model.risk_scale*normalized
        cox = self._cox_loss(risk, times, events)
        pair = smooth_concordance(times, events, normalized, sigma=float(self.config.get("point_sigma", .05)))
        brier = self._brier(representation, times, events) if self.brier_weight else nll*0.
        total = nll+self.point_weight*point_loss+self.pair_weight*pair+float(self.config.get("lambda_cox", .3))*cox+self.brier_weight*brier
        return total, dict(continuous_nll=nll if self.method == "ExploreLogTime" else nll*0.,
            pch=nll if self.method != "ExploreLogTime" else nll*0., point_balanced=point_loss,
            point_pair=pair, ranking_cox=cox, ipcw_brier=brier)

    def prior_loss(self, x):
        # Actual log-rate/log-hazard TEXGI and actual reported scalar risk.
        # Point risk is already the reported risk, so do not count it twice.
        return ImprovedModel.prior_loss(self, x)

    def _begin_epoch(self, epoch):
        self._prior_multiplier = None if self.method == "ExplorePrior" else 1.
        self._multiplier_epoch = epoch

    def _prepare_prior_multiplier(self, predictive, prior):
        if self._prior_multiplier is None:
            predictive_norm = self._gradient_norm(predictive, retain_graph=True)
            prior_norm = self._gradient_norm(prior, retain_graph=True)
            self._prior_multiplier = detached_prior_multiplier(predictive_norm, prior_norm,
                float(self.config.get("prior_ratio_target", .05)),
                float(self.config.get("prior_multiplier_min", .2)),
                float(self.config.get("prior_multiplier_max", 5.)))
            self.multiplier_audit.append(dict(epoch=self._multiplier_epoch,
                predictive_parameter_gradient_norm=predictive_norm,
                raw_prior_parameter_gradient_norm=prior_norm, multiplier=self._prior_multiplier,
                raw_ratio=prior_norm/max(predictive_norm, 1e-12),
                effective_ratio_before_warmup=self._prior_multiplier*prior_norm/max(predictive_norm, 1e-12),
                source="detached full-parameter norms on first optimization batch; fixed for this epoch"))
        return self._prior_multiplier

    def _audit_best(self, x, times, events, best_epoch):
        record = super()._audit_best(x, times, events, best_epoch)
        multiplier = float(self.trace[best_epoch].get("prior_multiplier", 1.))
        record.update(prior_multiplier=multiplier,
            effective_prior_gradient_norm=record["prior_warmup_factor"]*multiplier*record["prior_parameter_gradient_norm"])
        return record

    def _fit_calibration(self, validation_data, steps=None):
        super()._fit_calibration(validation_data, steps)
        self.calibration_record["objective"] = ("reserved-calibration exact conditional lognormal NLL" if self.method == "ExploreLogTime" else
            "reserved-calibration PCH NLL")+" + fitting-KM IPCW IBS + group-balanced point loss"
        self.calibration_record["point_censor_weight"] = self.censor_weight
        self.calibration_record["parameterization"] = ("positive location-contrast temperature and location shift; sigma unchanged" if self.method == "ExploreLogTime" else
            "positive shared contrast temperature and risk shift for curve/time heads")

    def predict_survival(self, x, times=None):
        if self.method != "ExploreLogTime":
            return super().predict_survival(x, times)
        self.model.eval()
        with torch.no_grad():
            parameters = self.model.distribution(self._x_tensor(x))
            grid = self.time_bins[1:] if times is None else times
            return conditional_lognormal_survival(parameters, grid, self.model.delta).numpy()

    def risk_target(self):
        return ("negative log1p(native conditional lognormal median / fitting scale)" if self.method == "ExploreLogTime" else
                "negative fitting-scaled calibrated auxiliary positive point estimate")

    def final_prior_diagnostics(self, x):
        record = super().final_prior_diagnostics(x)
        if self.method == "ExploreLogTime":
            record.update(attribution_target="exact continuous lognormal instantaneous log hazard at fitting-bin centers",
                point_target="negative log1p(native conditional median / fitting time scale)")
        return record

    def architecture_record(self, default):
        if self.method == "ExploreLogTime":
            return dict(native_head="continuous conditional shifted lognormal location-scale",
                likelihood="exact density/right-survival with log S(delta) normalization; no PCH approximation",
                location_constraint="mu>log(delta)+.1", sigma_bounds=[.15, 3.],
                location_contrast_bound=self.model.contrast_bound, calibration_temperature_bounds=[.2, 5.],
                calibration_location_shift_bounds=[-3., 3.], heteroscedastic=True, delta=self.model.delta,
                raw_input_location_skip=True, proportional_hazards=False)
        default["features"] = ("dense concatenation with gated multiplicative normalized input crosses" if self.method == "ExploreDense" else
                               "original layer-normalized MLP with raw-input linear time skip")
        return default

    def extra_metadata(self):
        logtime = self.method == "ExploreLogTime"
        return dict(recipe=self.method, point_censor_weight=self.censor_weight,
            event_weighting="plain: event-group mean plus coefficient times censor-group lower-bound mean",
            point_loss="separate event/censor smooth absolute errors on "+("log1p(time/fitting_scale)" if logtime else "time/fitting_scale"),
            point_time_definition=("native median of exact shifted lognormal conditional on T>=0" if logtime else
                "auxiliary positive conditional point estimate; not the PCH full mean or median"),
            attribution_target=("exact continuous lognormal instantaneous log hazard at fitting-bin centers" if logtime else
                "calibrated log PCH interval cumulative hazard"),
            attribution_units=("log(1/original time unit)" if logtime else "log cumulative interval hazard"),
            structural_prior="true fitting-knot log-hazard/log-rate TEXGI Phi time differences and separately weighted magnitude",
            adaptive_prior_multiplier=(dict(target=float(self.config.get("prior_ratio_target", .05)),
                bounds=[float(self.config.get("prior_multiplier_min", .2)),float(self.config.get("prior_multiplier_max", 5.))],
                observations=self.multiplier_audit, rule="detached epoch-first-batch norm ratio; not an extra prediction fit") if self.method == "ExplorePrior" else None),
            logtime_initialization=self.logtime_initialization_ if logtime else None,
            implementation="new exploratory mechanism; no claim of original-paper reproduction or clinically confirmed priors")


def fit_exploratory(method, c, x, t, e, query, grid, seed, split_seed):
    return fit_strengthened(method, c, x, t, e, query, grid, seed, split_seed,
                            _model_class=ExploratoryModel)
