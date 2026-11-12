"""Single-metric development variants built on the frozen Fourier frontier.

The local module retrieves fitting COVARIATES only, with no outcome values or
hard neighbor switching. Ranking, auxiliary time, true log-rate TEXGI and fit
partitions remain explicit. These are adaptations, not source-paper replicas.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .frontier import FrontierModel, FrontierNetwork
from .strengthened import fit_strengthened


METHODS = {"MetricRank", "MetricLocal", "MetricOutcome", "MetricCensor", "MetricMono", "MetricLocalNoPrior"}


def comparable_logistic_loss(risk, times, events, temperature=.25):
    """Non-saturating wrong-pair logistic surrogate; larger risk means earlier."""
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("A finite positive rank temperature is required")
    risk = risk.double()
    times = torch.as_tensor(times, dtype=torch.float64, device=risk.device)
    events = torch.as_tensor(events, dtype=torch.float64, device=risk.device)
    if risk.ndim != 1 or times.shape != risk.shape or events.shape != risk.shape:
        raise ValueError("Aligned rank/time/event vectors required")
    if not torch.isfinite(risk).all() or not torch.isfinite(times).all() or ((events != 0) & (events != 1)).any():
        raise ValueError("Finite risks/times and binary events required")
    comparable = (events[:, None] == 1) & (times[:, None] < times[None, :])
    if not comparable.any():
        return risk.sum()*0.
    margin = (risk[:, None]-risk[None, :])/float(temperature)
    return F.softplus(-margin[comparable]).mean()


class SoftCovariateMemory(nn.Module):
    """Smooth fitting-X attention with independent metric, gate and temperature."""
    def __init__(self, landmarks, initial_temperature=.5):
        super().__init__()
        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)
        if landmarks.ndim != 2 or not 1 <= len(landmarks) <= 256 or not torch.isfinite(landmarks).all():
            raise ValueError("A finite fitting-X landmark matrix of size1..256 is required")
        self.temperature_bounds = (.05, 5.)
        if not self.temperature_bounds[0] < float(initial_temperature) < self.temperature_bounds[1]:
            raise ValueError("Local initial temperature must be inside (.05,5)")
        probability = (float(initial_temperature)-.05)/4.95
        self.register_buffer("landmarks", landmarks.detach().clone())
        self.diagonal_raw = nn.Parameter(torch.zeros(landmarks.shape[1]))
        self.temperature_raw = nn.Parameter(torch.tensor(math.log(probability/(1.-probability))))
        self.gate_raw = nn.Parameter(torch.full((landmarks.shape[1],), math.log(.1/.9)))

    def temperature(self):
        return .05+4.95*torch.sigmoid(self.temperature_raw)

    def attention(self, x):
        metric = F.softplus(self.diagonal_raw)+1e-4
        metric = metric/metric.mean()
        difference = x[:, None, :]-self.landmarks[None, :, :]
        distance = (difference.square()*metric).mean(2)
        return torch.softmax(-distance/self.temperature(), dim=1)

    def forward(self, x):
        context = self.attention(x)@self.landmarks
        return x+torch.sigmoid(self.gate_raw)*(context-x)


def outcome_contrastive_loss(projection, times, events, fitting_scale, temperature=.1, time_width=.5):
    """Observed-event soft positives and only known-at-risk censor negatives.

    A censored j can enter event anchor i's denominator only when t_j>t_i.
    It never receives a positive event-time kernel target. Missing positive
    anchors are skipped; no censor duration is imputed as an event duration.
    """
    if min(float(fitting_scale), float(temperature), float(time_width)) <= 0:
        raise ValueError("Positive contrastive scales required")
    if projection.ndim != 2 or not torch.isfinite(projection).all():
        raise ValueError("Finite projected event representations required")
    times = torch.as_tensor(times, dtype=torch.float64, device=projection.device)
    events = torch.as_tensor(events, dtype=torch.float64, device=projection.device)
    if times.shape != (len(projection),) or events.shape != times.shape or (times < 0).any() or ((events != 0) & (events != 1)).any():
        raise ValueError("Aligned nonnegative times and binary events required")
    normalized = F.normalize(projection, dim=1, eps=1e-8)
    logits = (normalized@normalized.T).double()/float(temperature)
    diagonal = torch.eye(len(times), dtype=torch.bool, device=projection.device)
    event = events == 1
    admissible = (~diagonal) & (event[None, :] | (times[None, :] > times[:, None]))
    log_time = torch.log1p(times/float(fitting_scale))
    target = torch.exp(-.5*((log_time[:, None]-log_time[None, :])/float(time_width)).square())
    target = target*event[:, None]*event[None, :]*(~diagonal)
    positive_mass = target.sum(1)
    eligible = event & (positive_mass > 0) & admissible.any(1)
    audit = dict(event_anchors=int(event.sum()), eligible_anchors=int(eligible.sum()),
        positive_event_pairs=int((target > 0).sum()),
        known_later_censor_denominator_pairs=int((admissible & (~event)[None, :] & event[:, None]).sum()),
        censored_positive_pairs=0, source="current fitting minibatch only; event-time kernel targets")
    if not eligible.any():
        return projection.sum()*0., audit
    selected = logits[eligible]
    log_denominator = torch.logsumexp(selected.masked_fill(~admissible[eligible], -torch.inf), dim=1)
    probability = target[eligible]/positive_mass[eligible, None]
    return -(probability*(selected-log_denominator[:, None])).sum(1).mean(), audit


class SingleMetricNetwork(FrontierNetwork):
    def __init__(self, features, config, method, initial_rates, edges, horizon, point_scaling,
                 landmarks=None, censor_initial=None):
        super().__init__(features, config, "FrontierEmbed", initial_rates, edges, horizon, point_scaling)
        self.method = method
        self.rank_localizer, self.curve_localizer = nn.Identity(), nn.Identity()
        self.outcome_projector = None
        if method in {"MetricLocal", "MetricLocalNoPrior"}:
            if landmarks is None:
                raise ValueError("Fit-only covariate landmarks were not prepared")
            local_temperature = config.get("local_temperature", config.get("local_temp", .5))
            self.rank_localizer = SoftCovariateMemory(landmarks, local_temperature)
            self.curve_localizer = SoftCovariateMemory(landmarks, local_temperature)
        if method == "MetricOutcome":
            width = config["hidden"][-1]
            self.outcome_projector = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, min(width, 32)))
        if method == "MetricCensor":
            from .censor_representation import CensorDecomposedRepresentation
            if censor_initial is None:
                raise ValueError("Censor process baseline must be independently fitted")
            self.rank_features = CensorDecomposedRepresentation(
                self.embedding.output_features, config["hidden"], config["dropout"], censor_initial)
        if method == "MetricMono":
            self.point_shift = nn.Parameter(torch.zeros(()))
            self.point_slope_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))

    def rank_parameters(self):
        parameters = list(super().rank_parameters())
        parameters.extend(self.rank_localizer.parameters())
        if self.outcome_projector is not None:
            parameters.extend(self.outcome_projector.parameters())
        return tuple(parameters)

    def rank_encoded(self, x):
        return self.embedding(self.rank_localizer(x))

    def rank_hidden(self, x):
        return self.rank_features(self.rank_encoded(x))

    def risk(self, x):
        return self._rank(x, self.rank_encoded(x))

    def latent_components(self, x):
        # The Fourier inputs can be localized, while the skip remains raw x.
        rank = self._rank(x, self.rank_encoded(x))
        hidden = self.curve_features(self.embedding(self.curve_localizer(x)))
        point_raw = (self.point_head(hidden)+self.point_skip(x)).squeeze(1)
        point = self.contrast_bound*torch.tanh(point_raw/self.contrast_bound)
        residual = self.residual_bound*torch.tanh(self.residual_head(hidden)/self.residual_bound)
        baseline = self.initial_log_rates+self.baseline_shift_bound*torch.tanh(self.baseline_delta/self.baseline_shift_bound)
        return baseline, rank, point, residual

    def from_components(self, components, temperature=None, bias=None, *, point_temperature=None, point_bias=None, detach_rank=False):
        rates, point_time, contrast = super().from_components(components, temperature, bias,
            point_temperature=point_temperature, point_bias=point_bias, detach_rank=detach_rank)
        if self.method == "MetricMono":
            rank = components[1].detach() if detach_rank else components[1]
            point_temperature = self.point_temperature if point_temperature is None else point_temperature
            point_bias = self.point_bias if point_bias is None else point_bias
            slope = F.softplus(self.point_slope_raw.double())+1e-6
            point_time = self.point_scale*F.softplus(self.point_origin+self.point_shift.double()
                -slope*rank.double()/self.risk_scale/point_temperature-point_bias)
        return rates, point_time, contrast


class SingleMetricModel(FrontierModel):
    def __init__(self, method, config, rules, feature_names, random_state=0):
        if method not in METHODS:
            raise ValueError("Unknown single-metric variant")
        super().__init__("FrontierEmbed", config, rules, feature_names, random_state)
        self.method, self.has_prior = method, method != "MetricLocalNoPrior"
        self.preparation_seed = int(random_state)
        self.memory_audit, self.auxiliary_audit = None, {}

    def _prepare_fitting_features(self, xt, times, events):
        if self.method in {"MetricLocal", "MetricLocalNoPrior"}:
            count = int(self.config.get("local_memory", 256))
            if not 1 <= count <= 256:
                raise ValueError("Local memory budget must be in [1,256]")
            selected = np.random.default_rng(self.preparation_seed).choice(len(xt), min(count, len(xt)), replace=False)
            self.local_landmarks_ = xt.detach()[torch.as_tensor(selected)].clone()
            self.memory_audit = dict(source="current optimization standardized X only; no t/e, stop/cal/query values",
                fitting_rows=len(xt), landmark_count=len(selected), fitting_row_indices=selected.tolist(),
                seed=self.preparation_seed, uniform_without_replacement=True,
                standardized_covariate_sha256=hashlib.sha256(self.local_landmarks_.cpu().numpy().tobytes()).hexdigest())
        if self.method == "MetricCensor":
            self.censor_fit_times_ = torch.as_tensor(times, dtype=torch.float64).detach().cpu().numpy().copy()
            self.censor_fit_events_ = torch.as_tensor(events, dtype=torch.float64).detach().cpu().numpy().copy()

    def _create_network(self, features, initial, edges, horizon):
        censor_initial = None
        if self.method == "MetricCensor":
            from .censor_representation import fit_censor_baseline
            censor_initial, self.censor_baseline_audit_ = fit_censor_baseline(
                self.censor_fit_times_, self.censor_fit_events_, edges)
        return SingleMetricNetwork(features, self.config, self.method, initial, edges, horizon, self.point_scaling_,
            landmarks=getattr(self, "local_landmarks_", None), censor_initial=censor_initial)

    def _ranking_parts(self, x, times, events):
        risk = self.model.risk(x)
        pair = comparable_logistic_loss(risk/self.model.risk_scale, times, events,
                                        self.config.get("rank_sigma", .25))
        cox = self._cox_loss(risk, times, events)
        return float(self.config.get("lambda_cox", 1.))*cox+self.pair_weight*pair, cox, pair

    def _checkpoint_objective(self, x, times, events):
        from lifelines.utils import concordance_index
        risk = self.model.risk(x).detach().cpu().numpy()
        value = concordance_index(torch.as_tensor(times).detach().cpu().numpy(), -risk,
                                  torch.as_tensor(events).detach().cpu().numpy())
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite stop Harrell concordance")
        return torch.tensor(-float(value), dtype=torch.float64)

    def predictive_loss(self, x, times, events):
        total, parts = super().predictive_loss(x, times, events)
        if self.method == "MetricOutcome":
            projection = self.model.outcome_projector(self.model.rank_hidden(x))
            auxiliary, self.auxiliary_audit = outcome_contrastive_loss(projection, times, events,
                self.point_scaling_["scale"], self.config.get("contrastive_temperature", self.config.get("contrastive_temp", .1)),
                self.config.get("contrastive_time_width", self.config.get("time_kernel_width", self.config.get("timekernelwidth", .5))))
            total = total+float(self.config.get("contrastive_weight", .05))*auxiliary
            parts["outcome_contrastive"] = auxiliary
        if self.method == "MetricCensor":
            censor, orthogonal, diagnostics = self.model.rank_features.auxiliary(
                self.model.rank_encoded(x), times, events, self.time_bins)
            self.auxiliary_audit = diagnostics
            total = (total+float(self.config.get("censor_aux_weight", self.config.get("censoraux", .1)))*censor
                     +float(self.config.get("censor_orthogonal_weight", self.config.get("orthogonal", .01)))*orthogonal)
            parts.update(censor_auxiliary_nll=censor, censor_latent_cross_cov=orthogonal)
        return total, parts

    def architecture_record(self, default):
        record = super().architecture_record(default)
        record.update(numerical_embedding="inherited fixed smooth Fourier representation with raw skip",
            rank_pair_objective="strict comparable-pair logistic softplus(-margin/temperature), mean over valid pairs",
            checkpoint="reserved-stop negative Harrell C, eligible after full prior warmup",
            covariate_local_memory=self.memory_audit,
            local_attention="independent rank/curve full soft RBF; positive normalized diagonal metric; temperature(.05,5); featurewise sigmoid gate" if self.memory_audit else None,
            point_link="strictly decreasing positive-slope rank link; rank detached only for curve/point predictive losses" if self.method == "MetricMono" else None)
        if self.method == "MetricCensor":
            record["censor_representation"] = self.model.rank_features.metadata()
        return record

    def extra_metadata(self):
        record = super().extra_metadata()
        record.update(recipe=self.method, selection_criterion="reserved_stop_negative_harrell_c_after_full_prior_warmup",
            rank_pair_sigma=float(self.config.get("rank_sigma", .25)),
            rank_pair_weighting="strict comparable pairs, unweighted; non-saturating wrong-pair logistic",
            fit_covariate_memory=self.memory_audit, auxiliary_audit=self.auxiliary_audit,
            local_memory_contains_outcomes=False,
            outcome_auxiliary_definition=("event-event Gaussian log1p-time soft targets; observed events or known later censor values in denominator only; heuristic transfer, not ConSurv reproduction" if self.method == "MetricOutcome" else None),
            checkpoint_objective="negative Harrell C; trace field validation_predictive_objective retains compatibility name",
            inner_selection="unchanged paired .001 C-tolerance then IBS rule; not new C-only retuning of cached methods",
            point_time_definition=("auxiliary strictly decreasing function of actual rank; not native survival median" if self.method == "MetricMono" else
                                   "auxiliary positive curve-tower estimate; not full survival mean or median"),
            implementation="single-metric development adaptations; no claim of source-paper or clinically elicited-prior reproduction")
        if self.method == "MetricMono":
            record["point_link_parameters"] = dict(positive_slope=float(F.softplus(self.model.point_slope_raw.double()).detach())+1e-6,
                learned_shift=float(self.model.point_shift.detach()), actual_forward_rank_detached=False,
                predictive_point_loss_rank_detached=True)
        if self.method == "MetricCensor":
            record["censor_baseline_initialization"] = self.censor_baseline_audit_
        return record


def fit_single_metric(method, c, x, t, e, query, grid, seed, split_seed):
    return fit_strengthened(method, c, x, t, e, query, grid, seed, split_seed, _model_class=SingleMetricModel)
