"""Numerically stable LogisticHazard prediction with original PDF interpolation."""
import numpy as np


def survival_nodes(logits):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or not np.isfinite(logits).all():
        raise ValueError('Logistic hazard logits must be a finite matrix')
    # log(1-sigmoid(z)) = -softplus(z); no epsilon can increase S above one.
    return np.exp(-np.logaddexp(0.,logits).cumsum(axis=1))


def survival_frame(logits,cuts,sub=10):
    """Match PyCox const_pdf subgrid; first node includes its own hazard."""
    import pandas as pd
    nodes=survival_nodes(logits)
    cuts=np.asarray(cuts,dtype=np.float64)
    if cuts.ndim!=1 or len(cuts)!=nodes.shape[1] or len(cuts)<2:
        raise ValueError('Discrete nodes and duration cuts are misaligned')
    if not np.isfinite(cuts).all() or (np.diff(cuts)<=0).any() or not isinstance(sub,int) or sub<1:
        raise ValueError('Cuts must increase and interpolation count must be positive')
    fraction=np.arange(sub,dtype=np.float64)/sub
    grid=(cuts[:-1,None]+np.diff(cuts)[:,None]*fraction).reshape(-1)
    values=(nodes[:,:-1,None]*(1-fraction)+nodes[:,1:,None]*fraction).reshape(len(nodes),-1)
    values=np.column_stack([values,nodes[:,-1]])
    grid=np.r_[grid,cuts[-1]]
    return pd.DataFrame(values.T,index=grid)
