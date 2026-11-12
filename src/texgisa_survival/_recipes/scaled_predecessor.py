"""Explicit scale-stabilized extensions, never replacing the raw references."""
import copy
import numpy as np
import torch
from torch import nn
from .losses import smooth_concordance,point_loss
from .metrics import CensoringKM


from .predecessor import ResidualKM

def fit_scaled_predecessor(method,c,x,t,e,query,grid,seed,split_seed):
    from .adapters import Prediction,_standardize_risk
    from .data import preprocess,neural_partitions,duplicate_groups
    from .data import training_partitions
    core,stop,cal=training_partitions(x,t,e,split_seed,c)
    (xf,xs,xc,xq),preprocessor=preprocess(x.iloc[core],x.iloc[stop] if len(stop) else x.iloc[core],x.iloc[cal] if len(cal) else x.iloc[core],query,return_transformer=True)
    layers=[];previous=xf.shape[1]
    for width in c['hidden']:
        layers.extend([nn.Linear(previous,width),nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(c['dropout'])]);previous=width
    scale=max(float(np.median(t[core])),float(np.subtract(*np.percentile(t[core],[75,25]))),1e-6)
    initial=max(float(np.median(t[core]))/scale,.01)
    head=nn.Linear(previous,1,bias=True)
    nn.init.zeros_(head.weight);nn.init.constant_(head.bias,initial+np.log(-np.expm1(-initial)))
    layers.extend([head,nn.Softplus()])
    network=nn.Sequential(*layers)
    optimizer=torch.optim.SGD(network.parameters(),lr=c['lr'],momentum=.9)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.95)
    train_x=torch.as_tensor(xf);stop_x=torch.as_tensor(xs)
    # Preserve original durations through KM lookup; float32 rounding can cross
    # a censor jump and turn an intended G(T-) lookup into G(T).
    train_t=torch.as_tensor(t[core],dtype=torch.float64);train_e=torch.as_tensor(e[core],dtype=torch.float32)
    stop_t=torch.as_tensor(t[stop],dtype=torch.float64);stop_e=torch.as_tensor(e[stop],dtype=torch.float32)
    censoring=CensoringKM(t[core],e[core]);measure=c['measure']
    def objective(prediction,tt,ee):
        weights=None
        if method.startswith('Uno'):
            weights=torch.as_tensor(1/np.maximum(censoring.predict(tt.numpy(),left=True),.05)**2,dtype=prediction.dtype)
        ranking=smooth_concordance(tt/scale,ee,prediction,c['sigma'],ipcw=weights,ties=True)
        point=point_loss(tt/scale,ee,prediction,c['sigma'],measure,code_squared_weights=True)
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
        reference=network(train_x).flatten().numpy()*scale
        cal_prediction=network(torch.as_tensor(xc)).flatten().numpy()*scale
        prediction=network(torch.as_tensor(xq)).flatten().numpy()*scale
    calibration=ResidualKM(cal_prediction,t[cal],e[cal]) if len(cal) else None
    survival=calibration.survival(prediction,grid) if calibration is not None else np.ones((len(query),len(grid)))
    if c.get('_checkpoint') and calibration is not None:
        torch.save(dict(state=network.state_dict(),config=c,fitting_time_scale=scale,point_readout='positive dimensionless network output multiplied by fitting_time_scale',preprocessor=preprocessor,feature_names=list(x.columns),residual_times=calibration.times,
            residual_log_survival=calibration.log_values,residual_tail_scale=calibration.scale),c['_checkpoint'])
    return Prediction(_standardize_risk(-reference,-prediction),survival,
        dict(seed=seed,best_epochs=best_epochs,fitting_time_scale=scale,normalized_sigma=c['sigma'],raw_time_precision='float64 through censoring KM lookup',n_fit=len(core),n_stop=len(stop),n_calibration=len(cal),
             fit_rows=core.tolist(),stop_rows=stop.tolist(),calibration_rows=cal.tolist(),
             training_stop_objective=trace,point_time_prediction=prediction.tolist(),
             implementation='Normalized-loss positive-time extension, not an original paper reproduction; original four methods preserved separately',
             paper_code_discrepancy='Time and sigma are dimensionless; changed objective and positive initialized head explicitly disclosed. NMSE still uses code w-squared',
             prediction=('Native scalar point; anchor no IBS extension' if c.get('_anchor_full') else 'Original scalar-time ranking; IBS is residual-KM-calibrated extension, not paper-native output'),
             residual_tail='KM floor1/(n_cal+1), positive exponential tail std(residual), conditioned T>=0',
             ipcw_training='Fitting-only censoring KM left limits, fixed floor0.05 for Uno objective'))
