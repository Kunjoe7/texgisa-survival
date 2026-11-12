"""Independent nonlinear conditional shifted-lognormal baseline, not DSM.

For original T>=0: S(T)=S_logN(T+delta)/S_logN(delta); density has
the SAME normalizer. Delta and initialization use fitting observations only.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def distribution(mu,sigma,times,delta):
    mu,sigma=mu.double(),sigma.double()
    times=torch.as_tensor(times,dtype=torch.float64,device=mu.device)
    logt=(times+delta).log()
    z=(logt-mu)/sigma
    log_s0=torch.special.log_ndtr((mu-math.log(delta))/sigma)
    log_survival=torch.special.log_ndtr(-z)-log_s0
    log_density=-.5*z.square()-.5*math.log(2*math.pi)-sigma.log()-logt-log_s0
    return log_survival,log_density

def nll(mu,sigma,times,events,delta):
    ls,lf=distribution(mu,sigma,times,delta)
    ee=torch.as_tensor(events,dtype=ls.dtype,device=ls.device)
    return -(ee*lf+(1-ee)*ls).mean()

def median(mu,sigma,delta):
    mu,sigma=mu.double(),sigma.double()
    s0=torch.special.ndtr((mu-math.log(delta))/sigma)
    return (mu+sigma*torch.special.ndtri(1.-.5*s0)).exp()-delta

class LognormalNetwork(nn.Module):
    def __init__(self,p,c,t):
        super().__init__();positive=np.asarray(t)[np.asarray(t)>0]
        self.delta=max(float(np.median(positive))*1e-6,1e-8)
        initial=float(np.median(np.asarray(t)+self.delta))
        distance=max(math.log(initial/self.delta),.1)
        self.origin=distance+math.log(-math.expm1(-distance))
        layers=[];previous=p
        for width in c['hidden']:
            layers.extend([nn.Linear(previous,width),nn.LayerNorm(width),nn.SELU(),nn.Dropout(c['dropout'])]);previous=width
        self.features=nn.Sequential(*layers);self.head=nn.Linear(previous,2);self.skip=nn.Linear(p,1,bias=False)
        nn.init.normal_(self.head.weight,std=.01);nn.init.zeros_(self.head.bias);nn.init.zeros_(self.skip.weight)
    def forward(self,x):
        raw=self.head(self.features(x))
        contrast=12.*torch.tanh((raw[:,0]+self.skip(x).flatten())/12.)
        mu=math.log(self.delta)+F.softplus(self.origin+contrast.double())
        sigma=.05+2.95*torch.sigmoid(raw[:,1].double())
        return mu,sigma

def fit_neural_aft(c,x,t,e,query,grid,seed,split_seed):
    from .data import training_partitions,preprocess
    from .adapters import Prediction,_standardize_risk
    core,stop,cal=training_partitions(x,t,e,split_seed,c)
    arrays,transformer=preprocess(x.iloc[core],x.iloc[stop] if len(stop) else x.iloc[core],query,return_transformer=True)
    xf,xs,xq=map(torch.as_tensor,arrays)
    net=LognormalNetwork(xf.shape[1],c,t[core]);opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['wd'])
    trace=[];best=float('inf');state=None;stale=0;best_epochs=0
    for epoch in range(c.get('_stage_epochs',c['epochs'])):
        net.train()
        for ids in torch.randperm(len(xf)).split(c['batch']):
            opt.zero_grad();mu,sigma=net(xf[ids]);loss=nll(mu,sigma,t[core][ids],e[core][ids],net.delta)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite nonlinear AFT objective')
            loss.backward()
            if any(not torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None):raise FloatingPointError('Nonfinite nonlinear AFT gradient')
            nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        net.eval()
        with torch.no_grad():
            value=float(nll(*net(xs),t[stop],e[stop],net.delta)) if len(stop) else 0.
        if not np.isfinite(value):raise FloatingPointError('Nonfinite nonlinear AFT stopping loss')
        trace.append(value)
        if value<best or not len(stop):best=value;state=copy.deepcopy(net.state_dict());best_epochs=epoch+1;stale=0
        else:stale+=1
        if len(stop) and stale>=c['patience']:break
    if state is None:raise FloatingPointError('No nonlinear AFT checkpoint')
    net.load_state_dict(state);net.eval()
    with torch.no_grad():
        mu,sigma=net(xq);reference=net(xf)[0].numpy()
        survival=distribution(mu[:,None],sigma[:,None],np.asarray(grid)[None,:],net.delta)[0].exp().numpy()
        point=median(mu,sigma,net.delta).numpy()
    if not np.isfinite(point).all() or (point<0).any():raise FloatingPointError('Invalid conditional AFT median')
    meta=dict(seed=seed,best_epochs=best_epochs,selection_trace=trace,selection_criterion='stop_exact_conditional_lognormal_nll',
        fit_rows=core.tolist(),stop_rows=stop.tolist(),calibration_rows=cal.tolist(),n_fit=len(core),n_stop=len(stop),n_calibration=0,
        point_time_prediction=point.tolist(),point_readout='exact median of T>=0 conditional shifted-lognormal',
        fitting_delta=net.delta,sigma_bounds=[.05,3.],support='original T>=0, normalized in likelihood and survival',
        risk_target='negative conditional-lognormal location mu; heteroscedastic native median can order differently; report C_from_point separately',
        implementation='Independent conditional nonlinear lognormal AFT; not a DSM reproduction',
        reserved_calibration_unused=True)
    if c.get('_checkpoint'):torch.save(dict(state=state,config=c,preprocessor=transformer,delta=net.delta,origin=net.origin),c['_checkpoint'])
    return Prediction(_standardize_risk(-reference,-mu.numpy()),survival,meta)
