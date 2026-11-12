"""Small SALaD-inspired event/censor/shared representation, not a reproduction.

Primary sources inspected: https://proceedings.mlr.press/v306/qi26o.html and
https://github.com/shi-ang/SALaD (README, run.py, model/loss.py). We transfer
the three-representation/explicit-censor supervision idea. Centered normalized
cross-covariance below is our simpler local regularizer, not SALaD's complete
OrthoNets/IPM objective. No identification claim under dependent censoring.

All parameters belong to the containing ranking tower. The independent event
curve tower is deliberately absent from this module. Inputs are fitting-only
preprocessed (possibly Fourier-encoded) covariates, never outcome columns.
"""
import math

import numpy as np
import torch
from torch import nn

from .improved import fitting_baseline, pch_log_rate_nll


def _validated_edges(edges):
    value = np.asarray(edges, dtype=np.float64)
    if (value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
            or value[0] != 0 or (np.diff(value) <= 0).any()):
        raise ValueError("Censor head requires finite increasing fitting edges from zero")
    return value


def fit_censor_baseline(fit_times, fit_events, fit_edges):
    """Initialize PCH *censor* rates using only the supplied fitting outcomes.

    As for the event head, rates are interval cumulative hazards; duration-width
    constants are omitted from NLL. Event times right-censor the censor process,
    so its observation indicator is exactly 1-fit_events, not fit_events.
    """
    times = np.asarray(fit_times, dtype=np.float64)
    events = np.asarray(fit_events, dtype=np.float64)
    edges = _validated_edges(fit_edges)
    if (times.ndim != 1 or events.shape != times.shape or len(times) == 0
            or not np.isfinite(times).all() or (times < 0).any()
            or not np.isfinite(events).all() or not np.isin(events, [0., 1.]).all()):
        raise ValueError("Censor initialization needs nonempty finite fitting outcomes and binary event")
    observed_censoring = 1. - events
    log_rates, audit = fitting_baseline(times, observed_censoring, edges)
    if not np.isfinite(log_rates).all():
        raise FloatingPointError("Nonfinite fitting censor baseline")
    audit = dict(audit, source="current optimization fitting rows only; never stop/calibration/query",
        observed_indicator="1 - event", event_indicator_flipped=True,
        n_fit=len(times), observed_censoring=int(observed_censoring.sum()),
        observed_censoring_within_head=int((observed_censoring*(times <= edges[-1])).sum()),
        time_nodes="provided current-fitting PCH grid; shared with event curve",
        role="auxiliary censor-time head; not evaluation IPCW replacement")
    return log_rates, audit


def normalized_cross_covariance(representations, epsilon=1e-6):
    """Mean squared cross-correlations between centered latent coordinates.

    Fixed epsilon keeps constant-coordinate and second input derivatives finite.
    Each pair is divided by coordinate RMS before cross covariance, so a large
    private activation scale cannot dominate the penalty. This encourages
    separation; it does not establish statistical independence or causal roles.
    """
    views = tuple(representations)
    if not math.isfinite(epsilon) or epsilon <= 0 or len(views) < 2:
        raise ValueError("At least two representations and positive epsilon required")
    first = views[0]
    if first.ndim != 2 or first.shape[0] == 0:
        raise ValueError("Representations must be nonempty row-by-latent matrices")
    for view in views:
        if (view.ndim != 2 or len(view) != len(first) or view.shape[1] == 0
                or view.device != first.device or not view.is_floating_point()
                or not torch.isfinite(view).all()):
            raise ValueError("Representation shapes/devices or finite floating values differ")
    if len(first) < 2:
        return sum(view.sum()*0. for view in views)
    standardized = []
    for view in views:
        centered = view.double()-view.double().mean(0, keepdim=True)
        standardized.append(centered/torch.sqrt(centered.square().mean(0, keepdim=True)+epsilon))
    penalties = []
    for i, a in enumerate(standardized):
        for b in standardized[i+1:]:
            covariance = a.T@b/len(a)
            penalties.append(covariance.square().mean())
    return torch.stack(penalties).mean()


def _tower(features, hidden, dropout):
    layers, width = [], features
    for next_width in hidden:
        layers.extend([nn.Linear(width, next_width), nn.LayerNorm(next_width),
                       nn.SELU(), nn.Dropout(dropout)])
        width = next_width
    return nn.Sequential(*layers)


class CensorDecomposedRepresentation(nn.Module):
    """Three private/shared towers and event projection compatible with rank_features.

    ``forward(encoded)`` returns hidden[-1] event/shared features. The auxiliary
    censor head uses censor/shared features only. The caller adds its unweighted
    NLL and covariance outputs with predeclared weights; no hidden extra fit.
    """
    def __init__(self, input_features, hidden, dropout, initial_censor_log_rates):
        super().__init__()
        hidden = tuple(hidden)
        initial = np.asarray(initial_censor_log_rates, dtype=np.float64)
        if (not isinstance(input_features, int) or input_features <= 0 or not hidden
                or any(not isinstance(v, int) or v < 2 for v in hidden)
                or not math.isfinite(dropout) or not 0 <= dropout < 1
                or initial.ndim != 1 or len(initial) == 0 or not np.isfinite(initial).all()):
            raise ValueError("Invalid censor representation dimensions, dropout or initial rates")
        self.input_features, self.hidden, self.dropout = input_features, hidden, float(dropout)
        self.output_features = hidden[-1]
        self.event_private = _tower(input_features, hidden, dropout)
        self.censor_private = _tower(input_features, hidden, dropout)
        self.shared = _tower(input_features, hidden, dropout)
        self.event_projection = nn.Sequential(nn.Linear(2*hidden[-1], hidden[-1]),
                                              nn.LayerNorm(hidden[-1]), nn.SELU())
        self.censor_head = nn.Linear(2*hidden[-1], len(initial))
        nn.init.normal_(self.censor_head.weight, std=.01)
        nn.init.zeros_(self.censor_head.bias)
        self.censor_baseline_delta = nn.Parameter(torch.zeros(len(initial)))
        self.register_buffer("initial_censor_log_rates", torch.as_tensor(initial, dtype=torch.float64))
        self.censor_contrast_bound = 4.
        self.censor_baseline_shift_bound = 4.

    def _validate_input(self, encoded):
        if (encoded.ndim != 2 or encoded.shape[1] != self.input_features or len(encoded) == 0
                or not encoded.is_floating_point() or not torch.isfinite(encoded).all()):
            raise ValueError("Encoded covariates must be a finite nonempty matrix")

    def latent_components(self, encoded):
        self._validate_input(encoded)
        return self.event_private(encoded), self.censor_private(encoded), self.shared(encoded)

    def forward(self, encoded):
        self._validate_input(encoded)
        # Censor-private values/parameters cannot enter the reported risk path.
        event, shared = self.event_private(encoded), self.shared(encoded)
        return self.event_projection(torch.cat([event, shared], dim=1))

    def _censor_rates(self, censor_private, shared):
        raw = self.censor_head(torch.cat([censor_private, shared], dim=1))
        baseline = (self.initial_censor_log_rates
                    + self.censor_baseline_shift_bound*torch.tanh(
                        self.censor_baseline_delta.double()/self.censor_baseline_shift_bound))
        contrast = self.censor_contrast_bound*torch.tanh(raw.double()/self.censor_contrast_bound)
        return baseline[None, :]+contrast

    def censor_log_rates(self, encoded):
        self._validate_input(encoded)
        return self._censor_rates(self.censor_private(encoded), self.shared(encoded))

    def auxiliary(self, encoded, times, events, edges):
        event_private, censor_private, shared = self.latent_components(encoded)
        times = torch.as_tensor(times, dtype=torch.float64, device=encoded.device)
        events = torch.as_tensor(events, dtype=torch.float64, device=encoded.device)
        edges = torch.as_tensor(edges, dtype=torch.float64, device=encoded.device)
        if (times.shape != (len(encoded),) or events.shape != times.shape
                or not torch.isfinite(times).all() or (times < 0).any()
                or not torch.isfinite(events).all() or not ((events == 0) | (events == 1)).all()
                or edges.ndim != 1 or len(edges) != len(self.initial_censor_log_rates)+1
                or not torch.isfinite(edges).all() or edges[0] != 0 or (torch.diff(edges) <= 0).any()):
            raise ValueError("Auxiliary censor outcomes or PCH grid invalid")
        rates = self._censor_rates(censor_private, shared)
        nll = pch_log_rate_nll(rates, times, 1.-events, edges)
        covariance = normalized_cross_covariance([event_private, censor_private, shared])
        if not torch.isfinite(nll) or not torch.isfinite(covariance):
            raise FloatingPointError("Nonfinite censor auxiliary loss")
        diagnostics = dict(censor_nll=float(nll.detach()), latent_cross_cov=float(covariance.detach()),
            observed_censoring_in_batch=int((1.-events).sum()), auxiliary_batch_rows=len(encoded),
            observation_indicator="1 - event", losses_are_unweighted=True)
        return nll, covariance, diagnostics

    def auxiliary_losses(self, encoded, times, events, edges):
        nll, covariance, _ = self.auxiliary(encoded, times, events, edges)
        return dict(censor_nll=nll, latent_cross_cov=covariance)

    def metadata(self):
        return dict(implementation="minimal SALaD-inspired transfer; not full published reproduction",
            source_paper="Explicitly Modeling Censoring Produces Superior Survival Predictors, ICML2026",
            source_url="https://proceedings.mlr.press/v306/qi26o.html",
            representations=["event_private", "censor_private", "shared"],
            input_features=self.input_features, hidden=list(self.hidden), output_features=self.output_features,
            event_path="event_projection(event_private, shared)",
            censor_path="PCH head(censor_private, shared); observed indicator 1-event",
            parameter_group="all module parameters in ranking group; independent curve tower absent",
            censor_initial_log_cumulative_rates=self.initial_censor_log_rates.detach().cpu().tolist(),
            censor_contrast_bound=self.censor_contrast_bound,
            orthogonality="mean squared centered RMS-normalized cross covariance across all three latent pairs",
            published_components_not_reproduced=["OrthoNets weight-product penalty", "KM-weighted IPM/MMD"],
            assumption_boundary="auxiliary supervision does not identify outcomes under conditionally dependent censoring",
            censor_role="training auxiliary only; does not replace external evaluation censoring KM")
