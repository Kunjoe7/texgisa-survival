"""Exact piecewise-constant hazard predictions shared by both PCH models."""
import numpy as np


def survival_on_grid(cumulative, edges, grid):
    cumulative, edges, grid = np.asarray(cumulative), np.asarray(edges), np.asarray(grid)
    if cumulative.ndim != 2 or cumulative.shape[1] != len(edges)-1:
        raise ValueError('Interval hazards and edges are misaligned')
    if np.any(np.diff(edges) <= 0) or (cumulative < 0).any():
        raise ValueError('PCH widths must be positive and cumulative hazards nonnegative')
    indices = np.clip(np.searchsorted(edges,grid,side='right')-1,0,cumulative.shape[1]-1)
    fraction = np.clip((grid-edges[indices])/np.diff(edges)[indices],0.,1.)
    previous = cumulative.cumsum(axis=1)-cumulative
    return np.exp(-previous[:,indices]-cumulative[:,indices]*fraction)


def negative_rmst(cumulative, edges):
    cumulative = np.asarray(cumulative,dtype=float)
    previous_survival = np.exp(-(cumulative.cumsum(axis=1)-cumulative))
    ratio = np.ones_like(cumulative)
    np.divide(-np.expm1(-cumulative),cumulative,out=ratio,where=cumulative > 1e-10)
    # A nonzero starting edge contributes the same constant for every subject.
    return -(previous_survival*np.diff(edges)*ratio).sum(axis=1)
