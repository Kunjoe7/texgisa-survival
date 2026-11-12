"""Training-derived grid/censoring weights. No test-dependent grid selection."""
import numpy as np


def checked_survival(values, tolerance=1e-6):
    """Repair float32-scale roundoff only; reject substantive invalid curves."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Survival curves must be a finite matrix')
    if (values < -tolerance).any() or (values > 1+tolerance).any():
        raise ValueError('Survival probability violation exceeds rounding tolerance')
    repaired=np.minimum.accumulate(np.clip(values,0.,1.),axis=1)
    magnitude=float(np.max(np.abs(repaired-values),initial=0.))
    if magnitude>tolerance:
        raise ValueError('Survival monotonicity violation exceeds rounding tolerance')
    return repaired,dict(tolerance=tolerance,max_adjustment=magnitude,
                          adjusted_values=int(np.count_nonzero(repaired!=values)))


class CensoringKM:
    def __init__(self, times, events):
        times, events = np.asarray(times, float), np.asarray(events, int)
        # Own-model correction: remove tied events before censoring risk sets.
        from sksurv.nonparametric import kaplan_meier_estimator
        self.times, self.values = kaplan_meier_estimator(events.astype(bool), times, reverse=True)

    def predict(self, values, left=False):
        indices = np.searchsorted(self.times, values, side="left" if left else "right") - 1
        return np.where(indices < 0, 1., self.values[np.maximum(indices, 0)])


def evaluation_grid(train_t, train_e, points=100):
    train_t = np.asarray(train_t, float)
    lo, hi = np.quantile(train_t, [.1, .8])
    km = CensoringKM(train_t, train_e)
    grid = np.linspace(lo, hi, points)
    # Bound selected using training data only; positivity is shared by all methods.
    supported = grid[km.predict(grid) >= .05]
    if len(supported) < 2 or supported[-1] <= supported[0]:
        raise ValueError("Insufficient training-derived IPCW evaluation support")
    return np.linspace(supported[0], supported[-1], points)


def evaluate(train_t, train_e, test_t, test_e, risk, survival, grid):
    from lifelines.utils import concordance_index
    test_t, test_e = np.asarray(test_t, float), np.asarray(test_e, int)
    risk, survival, grid = np.asarray(risk), np.asarray(survival), np.asarray(grid)
    if risk.shape != (len(test_t),):
        raise ValueError("Risk predictions must contain one score per subject")
    if survival.shape != (len(test_t), len(grid)):
        raise ValueError("Survival predictions and evaluation grid are misaligned")
    if not np.isfinite(risk).all() or not np.isfinite(survival).all():
        raise FloatingPointError("Nonfinite prediction; no artificial failure score")
    survival, _ = checked_survival(survival)
    km = CensoringKM(train_t, train_e)
    event_weights = km.predict(test_t, left=True)
    bs = []
    for j, value in enumerate(grid):
        observed = (test_t <= value) & (test_e == 1)
        alive = test_t > value
        if (event_weights[observed] <= 0).any():
            raise ValueError("Observed event outside training censoring support")
        score = np.zeros(len(test_t))
        score[observed] = survival[observed, j]**2 / event_weights[observed]
        score[alive] = (1-survival[alive, j])**2 / km.predict(value)
        bs.append(float(score.mean()))
    integration = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return dict(c=float(concordance_index(test_t, -risk, test_e)),
                ibs=float(integration(bs, grid)/(grid[-1]-grid[0])),
                n_test=len(test_t), grid_min=float(grid[0]), grid_max=float(grid[-1]))
