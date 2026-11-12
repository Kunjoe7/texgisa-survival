"""Differentiable curve and predecessor objectives; finite censor-aware targets."""
import torch
from torch.nn import functional as F


def log_softplus(logits):
    return torch.where(logits < -20,logits,
        F.softplus(logits).clamp_min(torch.finfo(logits.dtype).tiny).log())


def pch_nll(logits,t,e,edges):
    clipped=t.clamp(max=float(edges[-1]))
    observed=e*(t<=edges[-1])
    k=torch.bucketize(clipped.contiguous(),edges[1:],right=True).clamp(max=len(edges)-2)
    widths=edges[1:]-edges[:-1]
    frac=((clipped-edges[k])/widths[k]).clamp(0,1)
    cumulative=F.softplus(logits)
    selected=cumulative.gather(1,k[:,None]).squeeze(1)
    exclusive=torch.cat([torch.zeros_like(cumulative[:,:1]),cumulative.cumsum(1)[:,:-1]],dim=1)
    previous=exclusive.gather(1,k[:,None]).squeeze(1)
    log_rate=log_softplus(logits)
    event_log_rate=log_rate.gather(1,k[:,None]).squeeze(1)
    return (previous+frac*selected-observed*event_log_rate).mean()


def curve_rmst(logits,edges):
    cumulative=F.softplus(logits)
    before=torch.cat([torch.zeros_like(cumulative[:,:1]),cumulative.cumsum(1)[:,:-1]],dim=1)
    ratio=-torch.expm1(-cumulative)/cumulative.clamp_min(1e-12)
    integral=torch.where(cumulative<1e-5,1-cumulative/2+cumulative.square()/6,ratio)
    return ((edges[1:]-edges[:-1])*torch.exp(-before)*integral).sum(1)


def smooth_concordance(t,e,predicted_time,sigma=.01,ipcw=None,ties=False):
    pair=(t[:,None]<t[None,:]).to(predicted_time.dtype)
    if ties:
        pair=pair+.5*(t[:,None]==t[None,:]).to(pair.dtype)
        pair.fill_diagonal_(0.)
    pair=pair*e[:,None]
    if ipcw is not None:
        pair=pair*ipcw[:,None]
    denominator=pair.sum()
    if denominator<=0:
        return predicted_time.sum()*0
    # Larger predicted time belongs to the later observed outcome.
    score=torch.sigmoid((predicted_time[None,:]-predicted_time[:,None])/sigma)
    return -(pair*score).sum()/denominator


def point_loss(t,e,prediction,sigma=.01,measure='mse',code_squared_weights=True):
    weights=e+(1-e)*torch.sigmoid((t-prediction)/sigma)
    centered=t-t.mean()
    if measure=='mse':
        weight=weights.square() if code_squared_weights else weights
        return ((prediction-t).square()*weight).sum()/((centered.square()*weight).sum()+1e-5)
    if measure=='mae':
        return ((prediction-t).abs()*weights).sum()/((centered.abs()*weights).sum()+1e-5)
    raise ValueError('Unknown point loss')
