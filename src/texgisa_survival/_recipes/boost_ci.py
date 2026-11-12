"""Independent bounded-pair, component-wise linear smooth Uno-C boosting.

References: Mayr and Schmid (2014), doi:10.1371/journal.pone.0084483;
https://search.r-project.org/CRAN/refmans/mboost/html/Family.html and the
``Cindex`` family in https://github.com/boost-R/mboost/blob/master/R/family.R.
No R code is copied. This is an approximate formula reimplementation, not an
R/mboost reproduction. Larger reported scores mean earlier events. Differences
are explicit: strict-time pairs, fitting-only G(T-) with a positivity floor,
bounded uniform pair sampling, n-scaled normalized-objective pseudo-responses,
and a training-objective backtracking safeguard. Rank scores are not native
survival probabilities; IBS/point outputs use a labelled fitting-only PH
Breslow extension, with a restricted median instead of an invented native time.
"""
from dataclasses import dataclass
import hashlib
import pickle

import numpy as np
from scipy.special import expit


MAX_PAIRS = 65536


@dataclass(frozen=True)
class PairSet:
    left: np.ndarray
    right: np.ndarray
    weights: np.ndarray
    n_samples: int
    total_pairs: int
    sampled: bool


def comparable_pairs(times, events, observation_weights=None, max_pairs=MAX_PAIRS, seed=0):
    """Uniformly sample strict comparable pairs without constructing n-by-n data.

    Sampling is with replacement when the bound is active. Each eligible pair
    has identical proposal probability, so its Uno weight needs no additional
    proposal correction. The self-normalized finite sample approximates the
    full weighted objective; it is not asserted to be an unbiased ratio.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events)
    if (times.ndim != 1 or events.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or (times < 0).any()
            or not np.isin(events, [0, 1]).all()):
        raise ValueError("Finite nonnegative times and binary events required")
    if int(max_pairs) != max_pairs or not 1 <= max_pairs <= MAX_PAIRS:
        raise ValueError("Pair bound must be an integer in [1, 65536]")
    weights = (np.ones(len(times)) if observation_weights is None
               else np.asarray(observation_weights, dtype=float))
    if weights.shape != times.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid fitting-only observation weights")
    order = np.argsort(times, kind="stable")
    left = np.flatnonzero((events == 1) & (weights > 0))
    starts = np.searchsorted(times[order], times[left], side="right")
    counts = len(times)-starts
    valid = counts > 0
    left, starts, counts = left[valid], starts[valid], counts[valid]
    cumulative = np.cumsum(counts, dtype=np.int64)
    if not len(cumulative) or cumulative[-1] == 0:
        raise ValueError("No strict comparable event pairs")
    total = int(cumulative[-1])
    sampled = total > max_pairs
    offsets = (np.random.default_rng(seed).integers(total, size=int(max_pairs))
               if sampled else np.arange(total, dtype=np.int64))
    slot = np.searchsorted(cumulative, offsets, side="right")
    preceding = np.r_[0, cumulative[:-1]]
    selected_left = left[slot]
    selected_right = order[starts[slot]+offsets-preceding[slot]]
    selected_weights = weights[selected_left]
    selected_weights = selected_weights/selected_weights.sum()
    return PairSet(selected_left, selected_right, selected_weights,
                   len(times), total, sampled)


def loss_and_negative_gradient(risk, pairs, sigma):
    """L=-sum(w*sigmoid((r_early-r_late)/sigma)); return L and -dL/dr."""
    risk = np.asarray(risk, dtype=float)
    if risk.shape != (pairs.n_samples,) or not np.isfinite(risk).all():
        raise ValueError("One finite risk score per fitting observation required")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Smoothing must be positive and finite")
    probability = expit((risk[pairs.left]-risk[pairs.right])/float(sigma))
    value = -float(np.dot(pairs.weights, probability))
    derivative = pairs.weights*probability*(1.-probability)/float(sigma)
    negative_gradient = (np.bincount(pairs.left, derivative, minlength=len(risk))
                         - np.bincount(pairs.right, derivative, minlength=len(risk)))
    return value, negative_gradient


def component_update(design, risk, pairs, sigma, learning_rate, gradient_multiplier):
    """One centered univariate least-squares base-learner update, no model fit.

    n * (-dL/dr) is the configured per-observation gradient convention. Backtrack
    only against this fixed fitting pair objective; never inspect query labels.
    """
    design = np.asarray(design, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if design.ndim != 2 or len(design) != len(risk) or not np.isfinite(design).all():
        raise ValueError("Finite aligned component design required")
    if (not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(gradient_multiplier) or gradient_multiplier <= 0):
        raise ValueError("Positive finite update scales required")
    loss, negative_gradient = loss_and_negative_gradient(risk, pairs, sigma)
    pseudo = negative_gradient*gradient_multiplier
    norm = np.einsum("ij,ij->j", design, design)
    correlation = design.T@pseudo
    gain = np.divide(correlation**2, norm, out=np.zeros_like(norm), where=norm > 1e-12)
    if not len(gain) or not np.isfinite(gain).all():
        raise ValueError("Invalid component least-squares gain")
    feature = int(np.argmax(gain))
    coefficient = float(correlation[feature]/norm[feature]) if gain[feature] > 0 else 0.
    step = float(learning_rate)*coefficient
    record = dict(loss_before=loss, loss_after=loss, selected_feature=feature,
                  negative_gradient_l2=float(np.linalg.norm(negative_gradient)),
                  pseudo_response_l2=float(np.linalg.norm(pseudo)), backtracks=0,
                  coefficient_step=0., accepted=False)
    if coefficient == 0.:
        return risk.copy(), record
    for backtracks in range(21):
        proposal = risk+step*design[:, feature]
        candidate, _ = loss_and_negative_gradient(proposal, pairs, sigma)
        if np.isfinite(candidate) and candidate < loss-1e-14:
            record.update(loss_after=candidate, backtracks=backtracks,
                          coefficient_step=step, accepted=True)
            return proposal, record
        step *= .5
    record["backtracks"] = 21
    return risk.copy(), record


def breslow_baseline(times, events, risk):
    """Training-only PH extension in log cumulative-hazard arithmetic."""
    times, events, risk = np.asarray(times, float), np.asarray(events), np.asarray(risk, float)
    if (times.ndim != 1 or events.shape != times.shape or risk.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(risk).all()
            or (times < 0).any() or not np.isin(events, [0, 1]).all() or not events.any()):
        raise ValueError("Valid fitting events and risk required for Breslow extension")
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    log_risk_set = np.logaddexp.accumulate(risk[order][::-1])[::-1]
    event_times, counts = np.unique(times[events == 1], return_counts=True)
    first = np.searchsorted(sorted_times, event_times, side="left")
    log_increment = np.log(counts)-log_risk_set[first]
    return dict(event_times=event_times,
                log_cumulative_hazard=np.logaddexp.accumulate(log_increment),
                support_max=float(times.max()))


def breslow_survival(baseline, risk, grid):
    risk, grid = np.asarray(risk, float), np.asarray(grid, float)
    if (risk.ndim != 1 or grid.ndim != 1 or not np.isfinite(risk).all()
            or not np.isfinite(grid).all() or (grid < 0).any()):
        raise ValueError("Finite nonnegative prediction grid and finite risk required")
    index = np.searchsorted(baseline["event_times"], grid, side="right")-1
    log_hazard = np.full(len(grid), -np.inf)
    present = index >= 0
    log_hazard[present] = baseline["log_cumulative_hazard"][index[present]]
    # exp(-exp(z)) has exact floating limits 0/1 in extreme PH risks. There is
    # no clipping or replacement of substantively invalid probabilities here.
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-np.exp(risk[:, None]+log_hazard[None, :]))


def breslow_restricted_median(baseline, risk):
    native = np.unique(np.r_[0., baseline["event_times"], baseline["support_max"]])
    survival = breslow_survival(baseline, risk, native)
    crossed = survival <= .5
    unresolved = ~crossed.any(1)
    point = np.where(unresolved, native[-1], native[crossed.argmax(1)])
    return point, int(unresolved.sum())


def fit_boost_ci(c, x, t, e, query, grid, seed):
    """One charged classical fitting stage; no neural core/stop/cal partition."""
    from .adapters import Prediction, _standardize_risk
    from .data import preprocess
    from .metrics import CensoringKM
    (fit, test), transformer = preprocess(x, query, dtype=np.float64, return_transformer=True)
    center = fit.mean(0)
    fit, test = fit-center, test-center
    times, events = np.asarray(t, float), np.asarray(e, int)
    floor = float(c.get("ipcw_floor", .05))
    if not 0 < floor <= 1:
        raise ValueError("IPCW positivity floor must lie in (0, 1]")
    censoring = CensoringKM(times, events)
    g_left = censoring.predict(times, left=True)
    weights = events/np.maximum(g_left, floor)**2
    pairs = comparable_pairs(times, events, weights, c.get("max_pairs", MAX_PAIRS), seed)
    sigma = float(c.get("sigma", .1))
    iterations = c.get("n_estimators", c.get("mstop", 200))
    if int(iterations) != iterations or iterations <= 0:
        raise ValueError("Positive integral boosting iteration budget required")
    learning_rate = float(c.get("learning_rate", .1))
    # This scalar is explicitly documented; it removes the 1/n gradient-size
    # dependence introduced by the globally normalized pair objective.
    multiplier = float(len(fit))
    risk, coefficient = np.zeros(len(fit)), np.zeros(fit.shape[1])
    trace = []
    initial_loss, initial_gradient = loss_and_negative_gradient(risk, pairs, sigma)
    for _ in range(int(iterations)):
        risk, record = component_update(fit, risk, pairs, sigma, learning_rate, multiplier)
        trace.append(record)
        if not record["accepted"]:
            break
        coefficient[record["selected_feature"]] += record["coefficient_step"]
    raw_query = test@coefficient
    final_loss, final_gradient = loss_and_negative_gradient(risk, pairs, sigma)
    if not np.allclose(risk, fit@coefficient, atol=1e-10, rtol=1e-10):
        raise FloatingPointError("Component coefficients and fitting risk differ")
    baseline = breslow_baseline(times, events, risk)
    survival = breslow_survival(baseline, raw_query, grid)
    point, unresolved = breslow_restricted_median(baseline, raw_query)
    metadata = dict(seed=seed, n_fit=len(fit), n_stop=0, n_calibration=0,
        implementation="Independent component-wise linear smooth Uno-C boosting; bounded-pair approximate reimplementation, not exact R/mboost",
        source="Mayr and Schmid 2014; mboost Cindex family; independent formula implementation",
        risk_orientation="larger risk means earlier event; explicit sign convention",
        objective="negative self-normalized weighted sigmoid concordance on fixed strict-time fitting pairs",
        censoring_source="all and only current fitting rows; G(T-) with fixed positivity floor",
        ipcw_floor=floor, sigma=sigma, learning_rate=learning_rate,
        gradient_multiplier=multiplier, gradient_convention="n * negative derivative of normalized pair objective",
        requested_iterations=int(iterations), accepted_iterations=sum(v["accepted"] for v in trace),
        stopped_reason="iteration_budget" if len(trace) == iterations and trace[-1]["accepted"] else "no_improving_component_update",
        fitting_pair_count=len(pairs.left), full_comparable_pair_count=pairs.total_pairs,
        sampled_pairs=pairs.sampled, pair_bound=int(c.get("max_pairs", MAX_PAIRS)),
        sampling="fixed-seed uniform comparable pairs with replacement when bound active; otherwise enumerate all strict pairs",
        pair_sha256=hashlib.sha256(pairs.left.tobytes()+pairs.right.tobytes()+pairs.weights.tobytes()).hexdigest(),
        initial_objective=initial_loss, final_objective=final_loss,
        objective_improvement=initial_loss-final_loss,
        initial_negative_gradient_l2=float(np.linalg.norm(initial_gradient)),
        final_negative_gradient_l2=float(np.linalg.norm(final_gradient)),
        training_trace=trace, coefficients=coefficient.tolist(), design_center=center.tolist(),
        transformed_features=fit.shape[1], fit_rows=list(range(len(fit))),
        survival_extension="Breslow PH baseline on fitting rows and raw learned ranking scores; not native BoostCI probabilities",
        point_time_prediction=point.tolist(), point_is_native=False,
        point_readout="restricted median of explicit fitting-only Breslow PH extension; not native BoostCI time",
        unresolved_median_count=unresolved, median_support_max=baseline["support_max"])
    if c.get("_checkpoint"):
        with open(c["_checkpoint"], "wb") as handle:
            pickle.dump(dict(preprocessor=transformer, center=center, coefficients=coefficient,
                             baseline=baseline, config=dict(c), metadata=metadata), handle)
    return Prediction(_standardize_risk(risk, raw_query), survival, metadata)
