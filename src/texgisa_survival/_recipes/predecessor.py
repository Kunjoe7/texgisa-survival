"""Code-formula reimplementation; original scalar method has no native IBS.

IBS uses a separately named calibration-only residual-KM location extension,
conditioned on T>=0, with a prespecified positive exponential residual tail.
"""
import copy
import numpy as np
import torch
from torch import nn
from .losses import smooth_concordance,point_loss
from .metrics import CensoringKM


class ResidualKM:
    def __init__(self,predictions,times,events):
        residual=np.asarray(times)-np.asarray(predictions)
        km=CensoringKM(residual,1-np.asarray(events))
        self.times=km.times
        self.log_values=np.log(np.maximum(km.values,1./(len(residual)+1)))
        self.scale=max(float(np.std(residual)),1e-3)

    def log_survival(self,values):
        values=np.asarray(values,dtype=float)
        index=np.searchsorted(self.times,values,side='right')-1
        result=np.where(index<0,0.,self.log_values[np.maximum(index,0)])
        return np.where(values>self.times[-1],self.log_values[-1]-(values-self.times[-1])/self.scale,result)

    def survival(self,predictions,grid):
        predicted=np.asarray(predictions)
        log_s=self.log_survival(np.asarray(grid)[None,:]-predicted[:,None])-self.log_survival(-predicted)[:,None]
        return np.exp(np.minimum(log_s,0.))


def fit_predecessor(method,c,x,t,e,query,grid,seed,split_seed):
    from .adapters import Prediction,_standardize_risk
    from .data import preprocess,neural_partitions,duplicate_groups
    from .data import training_partitions
    core,stop,cal=training_partitions(x,t,e,split_seed,c)
    (xf,xs,xc,xq),preprocessor=preprocess(x.iloc[core],x.iloc[stop] if len(stop) else x.iloc[core],x.iloc[cal] if len(cal) else x.iloc[core],query,return_transformer=True)
    layers=[];previous=xf.shape[1]
    for width in c['hidden']:
        layers.extend([nn.Linear(previous,width),nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(c['dropout'])]);previous=width
    layers.append(nn.Linear(previous,1,bias=False))
    network=nn.Sequential(*layers)
    optimizer=torch.optim.SGD(network.parameters(),lr=c['lr'],momentum=.9)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.95)
    train_x=torch.as_tensor(xf);stop_x=torch.as_tensor(xs)
    train_t=torch.as_tensor(t[core],dtype=torch.float32);train_e=torch.as_tensor(e[core],dtype=torch.float32)
    stop_t=torch.as_tensor(t[stop],dtype=torch.float32);stop_e=torch.as_tensor(e[stop],dtype=torch.float32)
    censoring=CensoringKM(t[core],e[core]);measure='mse' if method.endswith('NMSE') else 'mae'
    def objective(prediction,tt,ee):
        weights=None
        if method.startswith('Uno'):
            weights=torch.as_tensor(1/np.maximum(censoring.predict(tt.numpy(),left=True),.05)**2,dtype=prediction.dtype)
        ranking=smooth_concordance(tt,ee,prediction,c['sigma'],ipcw=weights,ties=True)
        point=point_loss(tt,ee,prediction,c['sigma'],measure,code_squared_weights=True)
        return c['alpha']*point+(1-c['alpha'])*ranking
    best=float('inf');best_state=None;stale=0;trace=[]
    best_epochs=0
    for epoch in range(c.get('_stage_epochs',c['epochs'])):
        network.train()
        for ids in torch.randperm(len(train_x)).split(c['batch']):
            if len(ids)<2:continue
            optimizer.zero_grad()
            loss=objective(network(train_x[ids]).flatten(),train_t[ids],train_e[ids])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite predecessor loss')
            loss.backward();nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step()
        scheduler.step();network.eval()
        with torch.no_grad(): validation=float(objective(network(stop_x).flatten(),stop_t,stop_e)) if len(stop) else 0.
        if not np.isfinite(validation):raise FloatingPointError('Nonfinite predecessor checkpoint')
        trace.append(validation)
        if validation<best or not len(stop):best=validation;best_state=copy.deepcopy(network.state_dict());stale=0;best_epochs=epoch+1
        else:stale+=1
        if stale>=c['patience']:break
    if best_state is None:raise FloatingPointError('No predecessor checkpoint')
    network.load_state_dict(best_state);network.eval()
    with torch.no_grad():
        reference=network(train_x).flatten().numpy()
        cal_prediction=network(torch.as_tensor(xc)).flatten().numpy()
        prediction=network(torch.as_tensor(xq)).flatten().numpy()
    calibration=ResidualKM(cal_prediction,t[cal],e[cal]) if len(cal) else None
    survival=calibration.survival(prediction,grid) if calibration is not None else np.ones((len(query),len(grid)))
    if c.get('_checkpoint') and calibration is not None:
        torch.save(dict(state=network.state_dict(),config=c,preprocessor=preprocessor,feature_names=list(x.columns),residual_times=calibration.times,
            residual_log_survival=calibration.log_values,residual_tail_scale=calibration.scale),c['_checkpoint'])
    return Prediction(_standardize_risk(-reference,-prediction),survival,
        dict(seed=seed,best_epochs=best_epochs,n_fit=len(core),n_stop=len(stop),n_calibration=len(cal),
             fit_rows=core.tolist(),stop_rows=stop.tolist(),calibration_rows=cal.tolist(),
             training_stop_objective=trace,point_time_prediction=prediction.tolist(),
             implementation='Independent author-code loss-formula reimplementation; fitting-only scaling/KM; no test-per-epoch',
             paper_code_discrepancy='NMSE weights squared matches official code, not single-weight printed Eq14',
             prediction=('Native scalar point; anchor no IBS extension' if c.get('_anchor_full') else 'Original scalar-time ranking; IBS is residual-KM-calibrated extension, not paper-native output'),
             residual_tail='KM floor1/(n_cal+1), positive exponential tail std(residual), conditioned T>=0',
             ipcw_training='Fitting-only censoring KM left limits, fixed floor0.05 for Uno objective'))
