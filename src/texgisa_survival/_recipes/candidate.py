# Licensed under the MIT license in ../LICENSE.
"""Experimental TexGISa changes; the reference package is never edited.

Changes: PCH-based checkpoint selection, bounded PCH calibration on a separate
calibration set, exponential PCH interpolation, and optional analytic-RMST ranking.
Architecture, fitting loss, attribution operations and expert rules are inherited.
"""
import numpy as np
import torch
from torch.nn import functional as F
from texgisa_survival.models.texgisa import TEXGISAModel
from .pch import survival_on_grid, negative_rmst


class CandidateModel(TEXGISAModel):
    def __init__(self, *, ranking_target="cox", **kwargs):
        if ranking_target not in {"cox", "rmst"}:
            raise ValueError("ranking_target must be cox or rmst")
        if kwargs.get("likelihood", "pch") != "pch":
            raise ValueError("This candidate is defined for likelihood='pch'")
        super().__init__(**kwargs)
        self.ranking_target = ranking_target
        self.calibration_record = {"status": "not_requested"}

    def _make_time_bins(self, time, event):
        edges = super()._make_time_bins(time, event)
        # Survival starts at time zero, including when the earliest training
        # observation is later. No validation/test time determines these edges.
        edges[0] = 0.
        if np.any(np.diff(edges) <= 0):
            raise ValueError("PCH intervals must have positive width")
        return edges

    def _pch_targets(self, times, events):
        # Events beyond the fitted horizon are administratively censored there;
        # counting them as failures in the last interval biases calibration.
        clipped = np.minimum(times, self.time_bins[-1])
        observed = events * (times <= self.time_bins[-1])
        bins = np.clip(np.digitize(clipped, self.time_bins[1:]), 0, self.num_time_bins - 1)
        return bins, observed, self._fraction_in_bin(clipped, bins)

    def _validate(self, validation_data):
        """Checkpoint by the same PCH+mini-batch-Cox family used for fitting."""
        x, times, events = validation_data
        times, events = np.asarray(times), np.asarray(events)
        x = np.asarray(x, dtype=np.float32)
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
        bins, observed, fractions = self._pch_targets(times, events)
        self.model.eval()
        with torch.no_grad():
            hazards, risk = self.model.hazards_and_risk(xt)
            loss = self._pch_loss(
                hazards, torch.as_tensor(bins, device=self.device),
                torch.as_tensor(observed, dtype=torch.float32, device=self.device),
                torch.as_tensor(fractions, dtype=torch.float32,
                                device=self.device))
            if risk is not None:
                loss = loss + self.lambda_cox * self._cox_loss(
                    risk, torch.as_tensor(times, dtype=torch.float32, device=self.device),
                    torch.as_tensor(events, dtype=torch.float32, device=self.device))
        return float(loss)

    def _validation_cindex(self, validation_data):
        from lifelines.utils import concordance_index
        x, times, events = validation_data
        # During fit(), BaseSurvivalModel.is_fitted is still false. Use the
        # network directly and match the selected ranking target.
        if self.ranking_target == "cox":
            risk = self._cox_risk(np.asarray(x, dtype=np.float32))
        else:
            risk = self._analytic_rmst(np.asarray(x, dtype=np.float32))
        return float(concordance_index(times, -risk, events))

    @staticmethod
    def _raw_pch_nll(logits, bins, events, fractions, log_temperature, bias):
        """Stable logit-space likelihood; width constants are parameter independent."""
        cumulative = F.softplus(logits / log_temperature.exp() + bias)
        selected = cumulative.gather(1, bins[:, None]).squeeze(1)
        previous = (cumulative.cumsum(1) - cumulative).gather(1, bins[:, None]).squeeze(1)
        return (previous + fractions * selected - events * selected.clamp_min(1e-12).log()).mean()

    def _fit_calibration(self, validation_data, steps=100):
        """Fit on dedicated calibration rows; reject nonfinite or worse solutions."""
        x, times, events = validation_data
        times, events = np.asarray(times), np.asarray(events)
        xt = torch.as_tensor(self.scaler.transform(np.asarray(x, dtype=np.float32)),
                             dtype=torch.float32, device=self.device)
        indices, observed, fractions = self._pch_targets(times, events)
        bins = torch.as_tensor(indices, device=self.device)
        evt = torch.as_tensor(observed, dtype=torch.float32, device=self.device)
        frac = torch.as_tensor(fractions, dtype=torch.float32,
                               device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.raw_logits(xt).detach()
        # Adam with projection keeps every evaluated iterate inside the bounds.
        log_t = torch.zeros((), device=self.device, requires_grad=True)
        bias = torch.zeros((), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([log_t, bias], lr=.03)
        initial = float(self._raw_pch_nll(logits, bins, evt, frac, log_t, bias).detach())
        if not np.isfinite(initial):
            raise FloatingPointError("Nonfinite uncalibrated PCH objective")
        best, best_t, best_b = initial, 0., 0.
        for _ in range(steps):
            optimizer.zero_grad()
            loss = self._raw_pch_nll(logits, bins, evt, frac, log_t, bias)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_t.clamp_(np.log(.2), np.log(5.))
                bias.clamp_(-3., 3.)
                value = float(self._raw_pch_nll(logits, bins, evt, frac, log_t, bias))
                if np.isfinite(value) and value < best:
                    best, best_t, best_b = value, float(log_t), float(bias)
        with torch.no_grad():
            self.model.temperature.fill_(np.exp(best_t))
            self.model.logit_bias.fill_(best_b)
        self.calibration_record = dict(status="fitted", initial_nll=initial, final_nll=best,
                                       temperature=float(np.exp(best_t)), bias=best_b,
                                       n_calibration=len(times))

    def _cumulative_intervals(self, x):
        x = np.asarray(x, dtype=np.float32)
        xt = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model.raw_logits(xt)
            value = F.softplus(raw / self.model.temperature + self.model.logit_bias)
        return value.double().cpu().numpy()

    def predict_survival(self, X, times=None):
        """Exact PCH survival on fitted support; hold endpoints outside support."""
        cumulative = self._cumulative_intervals(X)
        if times is None:
            return np.exp(-cumulative.cumsum(axis=1))
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("times must be a finite vector")
        return survival_on_grid(cumulative, self.time_bins, times)

    def predict_risk(self, X):
        if self.ranking_target == "cox":
            return super().predict_risk(X)
        return self._analytic_rmst(X)

    def _analytic_rmst(self, X):
        cumulative = self._cumulative_intervals(X)
        return negative_rmst(cumulative, self.time_bins)
