"""Real model adapters. Every call is downstream of run.py's approval check."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    risk: np.ndarray
    survival: np.ndarray
    metadata: dict


NEURAL = {"TCGAGeneFeedback", "TCGANoGeneFeedback", "Current", "Candidate", "FullNoPrior", "FullBase", "FullRank", "FullPoint", "FullDecoupled", "HarrellNMSE", "HarrellNMAE", "UnoNMSE", "UnoNMAE", "DeepSurv", "CoxTime", "DeepHit", "LogisticHazard", "PCHazard"}


NEURAL.update({"CoxCC","PMF","NMTLR","NeuralLognormalAFT","HarrellScaled","UnoScaled","ExploreBalance","ExploreDense","ExploreLogTime","ExplorePrior","ExploreBalanceNoPrior","MTLR","StrengthBrier","StrengthPoint","StrengthCox","StrengthPointNoPrior"})


NEURAL.update({'FrontierSplitNoPrior', 'FrontierEmbed', 'FrontierUno', 'FrontierSmooth', 'FrontierSplit'})


NEURAL.update({'MetricOutcome', 'MetricLocalNoPrior', 'MetricCensor', 'MetricLocal', 'MetricRank', 'MetricMono'})
NEURAL.update({'PriorIBSAlign','PriorCurveSeparate','PriorGeneLowRank'})
NEURAL.update({'RefineSmooth','RefineSparse','RefineCDF','RefineSmoothCDF'})
NEURAL.update({'RefineRiskCal','RefineTimeBasis','RefineEventBins','RefineHybridScore'})


def _frame_on_grid(frame, grid):
    times = np.asarray(frame.index, float)
    if np.any(np.diff(times) < 0):
        raise ValueError("Model survival grid must be sorted")
    values = np.asarray(frame.values, float).T
    indices = np.searchsorted(times, grid, side="right")-1
    out = values[:, np.maximum(indices, 0)]
    out[:, indices < 0] = 1.
    return out


def _rmst_risk(frame, grid):
    # Integrate every native step, including all jumps before the IBS grid's
    # lower bound. No sparse-grid trapezoid that could distort baseline ranking.
    horizon = float(np.max(grid))
    native = np.asarray(frame.index, dtype=float)
    boundaries = np.unique(np.r_[0.,native[(native>0) & (native<horizon)],horizon])
    return -(_frame_on_grid(frame,boundaries[:-1])*np.diff(boundaries)).sum(axis=1)


def _standardize_risk(reference_risk, query_risk):
    reference_risk, query_risk = np.asarray(reference_risk), np.asarray(query_risk)
    mean, scale = float(reference_risk.mean()), float(reference_risk.std())
    return (query_risk-mean)/max(scale, 1e-8)


def fit_one(method, config, x, t, e, query, grid, seed, split_seed, threads):
    from .data import preprocess, neural_partitions
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method in {'RefineRiskCal','RefineTimeBasis','RefineEventBins','RefineHybridScore'}:
        from .research_extra import fit_extra
        return fit_extra(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Refine'):
        from .research_round2 import fit_refined
        return fit_refined(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Prior'):
        from .ibs_upgrade import fit_upgrade
        return fit_upgrade(method,config,x,t,e,query,grid,seed,split_seed)
    if method in {'TCGAGeneFeedback','TCGANoGeneFeedback'}:
        from .tcga_knowledge import fit_tcga
        return fit_tcga(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Metric'):
        from .single_metric import fit_single_metric
        return fit_single_metric(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Frontier'):
        from .frontier import fit_frontier
        return fit_frontier(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Explore'):
        from .exploratory import fit_exploratory
        return fit_exploratory(method,config,x,t,e,query,grid,seed,split_seed)
    if method in {'HarrellScaled','UnoScaled'}:
        from .scaled_predecessor import fit_scaled_predecessor
        return fit_scaled_predecessor(method,config,x,t,e,query,grid,seed,split_seed)
    if method=='NeuralLognormalAFT':
        from .neural_aft import fit_neural_aft
        return fit_neural_aft(config,x,t,e,query,grid,seed,split_seed)
    if method in {'RandomSurvivalForest','GradientBoostingCox'}:
        from .tree_baselines import fit_tree
        return fit_tree(method,config,x,t,e,query,grid,seed,threads)
    if method=='BoostCI':
        from .boost_ci import fit_boost_ci
        return fit_boost_ci(config,x,t,e,query,grid,seed)
    if method.startswith('Strength'):
        from .strengthened import fit_strengthened
        return fit_strengthened(method,config,x,t,e,query,grid,seed,split_seed)
    if method in {'Tobit','Weibull','Logistic','Loglogistic','Lognormal'}:
        from .aft import fit_aft
        return fit_aft(method,config,x,t,e,query,grid,seed)
    if method.startswith('Improve'):
        from .improved import fit_improved
        return fit_improved(method,config,x,t,e,query,grid,seed,split_seed)
    if method.startswith('Full'):
        from .full import fit_full
        return fit_full(method,config,x,t,e,query,grid,seed,split_seed)
    if method in {'HarrellNMSE','HarrellNMAE','UnoNMSE','UnoNMAE'}:
        from .predecessor import fit_predecessor
        return fit_predecessor(method,config,x,t,e,query,grid,seed,split_seed)
    if method in {"Current", "Candidate", "CandidateRank"}:
        return _texgisa(method, config, x, t, e, query, grid, seed, split_seed)
    if method in NEURAL:
        return _pycox(method, config, x, t, e, query, grid, seed, split_seed)
    from sksurv.util import Surv
    from lifelines import CoxPHFitter
    import pandas as pd
    from .data import classical_design
    (xt, xq), design_metadata = classical_design(x, query)
    y = Surv.from_arrays(e.astype(bool), t)
    if method == "Cox":
        frame = pd.DataFrame(xt)
        frame["T"], frame["E"] = t, e
        model = CoxPHFitter(penalizer=config["pen"], l1_ratio=config["l1"])
        model.fit(frame, "T", "E", fit_options={"step_size": .25, "max_steps": 2000})
        risk = _standardize_risk(model.predict_log_partial_hazard(pd.DataFrame(xt)).values,
                                model.predict_log_partial_hazard(pd.DataFrame(xq)).values)
        survival = model.predict_survival_function(pd.DataFrame(xq), times=grid).values.T
    else:
        if method == "Coxnet":
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            model = CoxnetSurvivalAnalysis(alphas=[config["alpha"]], l1_ratio=config["l1_ratio"],
                                           max_iter=100000, fit_baseline_model=True)
        else:
            raise ValueError(f"Unknown method {method}")
        model.fit(xt, y)
        risk = _standardize_risk(model.predict(xt), model.predict(xq))
        functions = model.predict_survival_function(xq)
        # Step functions hold endpoints outside their own fitted support.
        survival = np.stack([np.where(grid < fn.x[0], 1., fn(np.clip(grid,fn.x[0],fn.x[-1])))
                             for fn in functions])
    native=np.unique(np.r_[0.,t])
    if method=='Cox': native_survival=model.predict_survival_function(pd.DataFrame(xq),times=native).values.T
    else: native_survival=np.stack([np.where(native<fn.x[0],1.,fn(np.clip(native,fn.x[0],fn.x[-1]))) for fn in functions])
    point,unresolved=restricted_median(native,native_survival)
    return Prediction(risk, survival, dict(point_time_prediction=point.tolist(),point_readout='median restricted to native fitting support',unresolved_median_count=unresolved,n_fit=len(x), n_stop=0, n_calibration=0,
                                           seed=seed, transformed_features=xt.shape[1],design=design_metadata))


def _texgisa(method, config, x, t, e, query, grid, seed, split_seed):
    from .data import preprocess, neural_partitions
    from texgisa_survival.models.texgisa import TEXGISAModel
    from .candidate import CandidateModel
    from .data import duplicate_groups
    from .data import training_partitions
    core,stop,cal=training_partitions(x,t,e,split_seed,config)
    xf, xs, xc, xq = preprocess(x.iloc[core], x.iloc[stop] if len(stop) else x.iloc[core], x.iloc[cal], query)
    kw = dict(hidden_layers=config["hidden"], num_time_bins=config["bins"],
              dropout=config["dropout"], weight_decay=config["wd"],
              activation="selu", time_binning="quantile", architecture=config["arch"],
              norm="layer", es_metric=config["es_metric"], likelihood="pch",
              lambda_cox=config["lambda_cox"], calibrate=False,
              ema_decay=config["ema_decay"], input_noise=config["input_noise"],
              input_gate=config["input_gate"], lambda_smooth=0., lambda_expert=0.,
              random_state=seed, device="cpu")
    base=CandidateModel if method in {"Candidate","CandidateRank"} else TEXGISAModel
    class Observed(base):
        def _validate(self,data):
            value=super()._validate(data)
            self.epoch_values.append(value)
            return value
        def _validation_cindex(self,data):
            value=super()._validation_cindex(data)
            self.cindex_values.append(-value)
            return value
    model=Observed(**({"ranking_target":config['ranking_target']} if base is CandidateModel else {}),**kw)
    model.epoch_values=[]
    model.cindex_values=[]
    batch = len(core) if config["batch"] == "full" else config["batch"]
    model.fit(xf, t[core], e[core], validation_data=((xs, t[stop], e[stop]) if len(stop) else None),
              validation_split=0., epochs=config.get('_stage_epochs',config['epochs']), patience=config['patience'], batch_size=batch,
              learning_rate=config["lr"], early_stopping=bool(len(stop)), verbose=0)
    if config["calibrate"]:
        model._fit_calibration((xc, t[cal], e[cal]))
    risk = _standardize_risk(model.predict_risk(xf), model.predict_risk(xq))
    survival = model.predict_survival(xq, times=grid)
    native=np.unique(np.r_[0.,t[core]])
    point,unresolved=restricted_median(native,model.predict_survival(xq,times=native))
    return Prediction(risk, survival, dict(n_fit=len(core), n_stop=len(stop),
            n_calibration=len(cal), seed=seed, point_time_prediction=point.tolist(),point_readout="median restricted to native fitting support",unresolved_median_count=unresolved,transformed_features=xf.shape[1],
            best_epochs=(int(np.argmin(model.cindex_values if config['es_metric']=='cindex' else model.epoch_values))+1 if model.epoch_values else config.get("_stage_epochs",config["epochs"])),
            selection_trace=(model.cindex_values if config['es_metric']=='cindex' else model.epoch_values),selection_criterion=('negative_stop_C' if config['es_metric']=='cindex' else 'stop_predictive_loss'),
            calibration=getattr(model, "calibration_record", {"objective": "BCE"}),
            fit_rows=core.tolist(), stop_rows=stop.tolist(), calibration_rows=cal.tolist()))


def _pycox(method, c, x, t, e, query, grid, seed, split_seed):
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH, CoxTime, DeepHitSingle, LogisticHazard, PCHazard, MTLR, PMF, CoxCC
    from pycox.models.cox_time import MLPVanillaCoxTime
    from pycox.models.loss import cox_ph_loss
    from .data import preprocess, neural_partitions
    from .data import duplicate_groups
    from .data import training_partitions
    core,stop,cal=training_partitions(x,t,e,split_seed,c)
    xf, xs, xq = preprocess(x.iloc[core], x.iloc[stop] if len(stop) else x.iloc[core], query)
    optimizer = (tt.optim.AdamW(lr=c["lr"], decoupled_weight_decay=c["wd"]*c['lr'])
                 if c["wd"] else tt.optim.Adam(c["lr"]))
    def network(outputs):
        return tt.practical.MLPVanilla(xf.shape[1], c["hidden"], outputs,
                                      batch_norm=True, dropout=c["dropout"])
    if method in {"DeepSurv","CoxCC"}:
        class SafeLoss(torch.nn.Module):
            def forward(self, log_h, durations, events):
                return log_h.sum()*0 if events.sum() < 1 else cox_ph_loss(log_h, durations, events)
        model = (CoxCC(network(1),optimizer,device='cpu') if method=='CoxCC' else CoxPH(network(1), optimizer, loss=SafeLoss(), device='cpu'))
        train_y = (t[core].astype("float32"), e[core].astype("float32"))
        val_y = (t[stop].astype("float32"), e[stop].astype("float32")) if len(stop) else None
    elif method == "CoxTime":
        label = CoxTime.label_transform()
        train_y = label.fit_transform(t[core].astype("float32"), e[core].astype("float32"))
        val_y = label.transform(t[stop].astype("float32"), e[stop].astype("float32")) if len(stop) else None
        net = MLPVanillaCoxTime(xf.shape[1], c["hidden"], batch_norm=True, dropout=c["dropout"])
        model = CoxTime(net, optimizer, labtrans=label, device='cpu')
    else:
        cls = {"DeepHit": DeepHitSingle, "LogisticHazard": LogisticHazard, "PCHazard": PCHazard,"MTLR":MTLR,"NMTLR":MTLR,"PMF":PMF}[method]
        label = cls.label_transform(c["bins"])
        train_y = label.fit_transform(t[core], e[core])
        val_y = label.transform(t[stop], e[stop]) if len(stop) else None
        extra = {"alpha": c["alpha"], "sigma": c["sigma"]} if method == "DeepHit" else {}
        net=torch.nn.Linear(xf.shape[1],label.out_features) if method=="MTLR" else network(label.out_features)
        model = cls(net, optimizer, duration_index=label.cuts, device='cpu', **extra)
    model.fit(xf, train_y, batch_size=c['batch'], epochs=c.get('_stage_epochs',c['epochs']), verbose=False,
        callbacks=([tt.callbacks.EarlyStopping(patience=c['patience'])] if len(stop) else []),
        val_data=((xs,val_y) if len(stop) else None))
    history=model.log.to_pandas()
    best_epochs=(int(np.nanargmin(history['val_loss'].values))+1 if len(stop) else c['_stage_epochs'])
    selection_trace=history['val_loss'].tolist() if len(stop) else []
    if method in {"DeepSurv", "CoxTime","CoxCC"}:
        model.compute_baseline_hazards()
    if method == 'PCHazard':
        from .pch import survival_on_grid, negative_rmst
        # Official PyCox predict() returns pre-softplus interval logits. Use
        # exact exponential interpolation, matching the Candidate convention.
        raw_fit, raw_query = np.asarray(model.predict(xf)), np.asarray(model.predict(xq))
        fit_hazard, query_hazard = np.logaddexp(0.,raw_fit), np.logaddexp(0.,raw_query)
        risk = _standardize_risk(negative_rmst(fit_hazard,label.cuts), negative_rmst(query_hazard,label.cuts))
        native_surv=survival_on_grid(query_hazard,label.cuts,label.cuts)
        point,unresolved=restricted_median(label.cuts,native_surv)
        return Prediction(risk,survival_on_grid(query_hazard,label.cuts,grid),
            dict(seed=seed,best_epochs=best_epochs,selection_trace=selection_trace,selection_criterion='stop_predictive_loss',n_fit=len(core),n_stop=len(stop),n_calibration=0,
                 point_time_prediction=point.tolist(),point_readout='median restricted to native fitting support',unresolved_median_count=unresolved,
                 transformed_features=xf.shape[1],fit_rows=core.tolist(),stop_rows=stop.tolist(),
                 prediction='exact PCH exponential survival and analytic core-horizon RMST'))
    if method == 'LogisticHazard':
        from .logistic import survival_frame
        frame = survival_frame(model.predict(xq),label.cuts,sub=10)
        reference_frame = survival_frame(model.predict(xf),label.cuts,sub=10)
    else:
        predictor = model.interpolate(10) if method in {'DeepHit','PMF','NMTLR'} else model
        frame = predictor.predict_surv_df(xq)
    if method in {"DeepSurv","CoxCC"}:
        reference, risk = model.predict(xf).ravel(), model.predict(xq).ravel()
    else:
        if method != 'LogisticHazard':
            reference_frame=predictor.predict_surv_df(xf)
        reference, risk = _rmst_risk(reference_frame, grid), _rmst_risk(frame, grid)
    point,unresolved=restricted_median(np.asarray(frame.index),np.asarray(frame.values).T)
    return Prediction(_standardize_risk(reference, risk), _frame_on_grid(frame, grid),
            dict(seed=seed,best_epochs=best_epochs,selection_trace=selection_trace,selection_criterion='stop_predictive_loss', n_fit=len(core), n_stop=len(stop), n_calibration=0,
                 point_time_prediction=point.tolist(),point_readout="median restricted to native fitting support",unresolved_median_count=unresolved,
                 transformed_features=xf.shape[1], fit_rows=core.tolist(), stop_rows=stop.tolist(),
                 prediction=('float64 softplus cumulative survival; const_pdf sub10'
                             if method=='LogisticHazard' else 'native PyCox prediction')))


def restricted_median(times,survival):
    times=np.asarray(times);survival=np.asarray(survival)
    crossed=survival<=.5;first=crossed.argmax(axis=1)
    missing=~crossed.any(axis=1)
    return np.where(missing,times[-1],times[first]),int(missing.sum())


def fit_predict(method,config,x,t,e,query,grid,seeds,split_seed,threads,budget):
    if len(seeds)!=1:raise ValueError('One model per seed, never an ensemble')
    seed=seeds[0]
    clean={k:v for k,v in config.items() if k!='_checkpoint'}
    budget.reserve(method+':select',[seed])
    first=fit_one(method,clean,x,t,e,query,grid,seed,split_seed,threads)
    if method in NEURAL:
        epochs=int(first.metadata['best_epochs'])
        if not 1<=epochs<=config['epochs']:raise ValueError('Invalid selected epoch')
        if 'training_trace' in first.metadata:
            trace=[dict(epoch=v['epoch'],eligible=v['checkpoint_eligible'],value=v['validation_predictive_objective']) for v in first.metadata['training_trace']]
            criterion=first.metadata.get('selection_criterion','stop_predictive_loss; eligible after full prior warmup')
        else:
            values=first.metadata.get('selection_trace',first.metadata.get('training_stop_objective',[]))
            trace=[dict(epoch=i,eligible=True,value=v) for i,v in enumerate(values)]
            criterion=first.metadata.get('selection_criterion','stop_predictive_loss')
        if not trace or any(v['value'] is None or not np.isfinite(v['value']) for v in trace):raise FloatingPointError('Missing/nonfinite stage1 checkpoint criterion')
        if min((v for v in trace if v['eligible']),key=lambda v:v['value'])['epoch']+1!=epochs:raise ValueError('Selected epoch differs from saved stopping trace')
        final_config=dict(config,_stage_epochs=epochs)
        budget.reserve(method+':refit',[seed])
        final=fit_one(method,final_config,x,t,e,query,grid,seed,split_seed,threads)
        final.metadata['stage1_epoch_selection']=dict(best_epochs=epochs,fit_rows=first.metadata['fit_rows'],stop_rows=first.metadata['stop_rows'],criterion=criterion,trace=trace)
    else:final=first
    return Prediction(final.risk,final.survival,dict(members=[final.metadata],seeds=[seed]))
