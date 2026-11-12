"""Experimental full TexGISa recipes; frozen public package remains untouched."""
from pathlib import Path
import json,copy,math
import numpy as np
import torch
from torch.nn import functional as F
from .candidate import CandidateModel
from .losses import pch_nll,curve_rmst,smooth_concordance,point_loss,log_softplus


class FullModel(CandidateModel):
    @staticmethod
    def _raw_pch_nll(logits,bins,events,fractions,log_temperature,bias):
        z=logits/log_temperature.exp()+bias
        rate=F.softplus(z)
        previous=torch.cat([torch.zeros_like(rate[:,:1]),rate.cumsum(1)[:,:-1]],dim=1)
        k=bins[:,None]
        return (previous.gather(1,k).squeeze(1)+fractions*rate.gather(1,k).squeeze(1)
                -events*log_softplus(z).gather(1,k).squeeze(1)).mean()

    def __init__(self,recipe,config,rules,feature_names,**kwargs):
        super().__init__(ranking_target='cox',**kwargs)
        self.recipe=recipe;self.config=config;self.rules=rules;self.feature_names=feature_names
        self.trace=[];self.prior_audit={};self.calibration_record={'status':'not_requested'}

    def curve_logits_risk(self,x):
        f=self.model._features(x)
        cox=self.model.cox(f).squeeze(1)
        coupling=.25 if self.recipe=='FullDecoupled' else 1.
        logits=self.model.base.logits(f)+coupling*cox[:,None]
        return logits,cox

    def _build_network(self,input_dim):
        network=super()._build_network(input_dim)
        # All inherited prediction/attribution operations must share this head.
        if self.recipe=='FullDecoupled':
            import types
            def weak_logits(module,f):
                return module.base.logits(f)+.25*module.cox(f)
            network._logits=types.MethodType(weak_logits,network)
        return network

    def predictive_loss(self,x,t,e,edges):
        logits,cox=self.curve_logits_risk(x)
        nll=pch_nll(logits,t,e,edges)
        normalized_time=curve_rmst(logits,edges)/edges[-1]
        cox_loss=self._cox_loss(cox,t,e)
        ranking=smooth_concordance(t,e,normalized_time,sigma=.05)
        # RMST is bounded; times beyond its horizon are censored at that horizon.
        clipped=t.clamp(max=float(edges[-1]))/edges[-1]
        observed=e*(t<=edges[-1])
        point=point_loss(clipped,observed,normalized_time,sigma=.05,code_squared_weights=False)
        if self.recipe=='FullBase':
            total=nll+.3*cox_loss
        elif self.recipe=='FullRank':
            total=nll+.1*cox_loss+.2*ranking
        else:
            total=nll+.1*cox_loss+.2*ranking+.1*point
        return total,dict(pch=nll,cox=cox_loss,ranking=ranking,point=point)

    def prior_loss(self,x,edges):
        from texgisa_survival.texgi_utils import texgi_time_series
        if self.recipe=='FullNoPrior':
            return x.sum()*0,dict(attr_smooth=0.,direction=0.,active_rule_rows=0)
        x=x[:self.config['prior_rows']]
        baseline=torch.as_tensor(self.baseline_reference_,dtype=x.dtype).expand_as(x)
        training=self.model.training;self.model.eval()
        try:
            derivative=texgi_time_series(self.model,x,baseline,M=self.config['ig_steps'],
                t_sample=self.config['prior_time_samples'],create_graph=True,multiply_by_inputs=False)
            phi=derivative*(x-baseline).unsqueeze(0)
            step=max(1,self.num_time_bins//self.config['prior_time_samples'])
            bins=list(range(0,self.num_time_bins,step))[:self.config['prior_time_samples']]
            centers=(edges[1:]+edges[:-1])/2
            gaps=torch.diff(centers[bins]);gaps=gaps/gaps.mean()
            smooth=(torch.diff(phi,dim=0)/gaps[:,None,None]).abs().mean()
            direction=x.sum()*0;active=0
            z=x.detach().clone().requires_grad_(True)
            logits,cox=self.curve_logits_risk(z)
            risk=cox if self.recipe=='FullBase' else -curve_rmst(logits,edges)/edges[-1]
            risk_gradient=torch.autograd.grad(risk.sum(),z,create_graph=True)[0]
            for rule in self.rules:
                j=self.feature_names.index(rule['feature'])
                mask=(x[:,j]>float(self.rule_reference_['median'][j]) if rule['condition']=='above_training_median'
                      else torch.ones(len(x),dtype=torch.bool))
                if mask.any():
                    active+=int(mask.sum())
                    direction=direction+F.relu(.001-rule['sign']*derivative[:,mask,j]).mean()
                    direction=direction+F.relu(.001-rule['sign']*risk_gradient[mask,j]).mean()
            total=self.config['lambda_attr_smooth']*smooth+self.config['lambda_expert']*direction
            return total,dict(attr_smooth=float(smooth.detach()),direction=float(direction.detach()),active_rule_rows=active)
        finally:
            self.model.train(training)

    def fit_full(self,x,t,e,stop,seed):
        self.scaler.fit(x)
        xt=torch.as_tensor(self.scaler.transform(x),dtype=torch.float32)
        xs=torch.as_tensor(self.scaler.transform(stop[0]),dtype=torch.float32)
        self.baseline_reference_=np.quantile(xt.numpy(),.98,axis=0).astype(np.float32)
        self.rule_reference_={'median':np.median(xt.numpy(),axis=0)}
        self.time_bins=self._make_time_bins(np.asarray(t),np.asarray(e))
        self.num_time_bins=len(self.time_bins)-1
        self.model=self._build_network(xt.shape[1])
        edges=torch.as_tensor(self.time_bins,dtype=torch.float32)
        tt=torch.as_tensor(t,dtype=torch.float32);ee=torch.as_tensor(e,dtype=torch.float32)
        st=torch.as_tensor(stop[1],dtype=torch.float32);se=torch.as_tensor(stop[2],dtype=torch.float32)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config['lr'],weight_decay=self.config['wd'])
        best=float('inf');best_state=None;stale=0
        for epoch in range(self.config['epochs']):
            self.model.train();permutation=torch.randperm(len(xt));components=[]
            for ids in permutation.split(self.config['batch']):
                optimizer.zero_grad()
                predictive,parts=self.predictive_loss(xt[ids],tt[ids],ee[ids],edges)
                prior,diagnostics=self.prior_loss(xt[ids],edges)
                total=predictive+prior
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite full objective')
                if not self.prior_audit and self.recipe!='FullNoPrior':
                    gradients=torch.autograd.grad(prior,tuple(self.model.parameters()),retain_graph=True,allow_unused=True)
                    norm=math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
                    self.prior_audit=dict(first_prior_parameter_gradient_norm=norm,**diagnostics,
                        lambda_attr_smooth=self.config['lambda_attr_smooth'],lambda_direction=self.config['lambda_expert'],
                        directional_rule_count=len(self.rules),feature_names=self.feature_names,
                        baseline='fitting-only 0.98 per-feature quantile',ig_steps=self.config['ig_steps'])
                total.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);optimizer.step()
                components.append(dict(total=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()},**diagnostics))
            self.model.eval()
            with torch.no_grad(): val=float(self.predictive_loss(xs,st,se,edges)[0])
            if not np.isfinite(val):raise FloatingPointError('Nonfinite full checkpoint objective')
            trace={k:float(np.mean([r[k] for r in components])) for k in components[0]}
            self.trace.append(dict(epoch=epoch,validation_predictive_objective=val,**trace))
            if val<best:
                best=val;best_state=copy.deepcopy(self.model.state_dict());stale=0
            else:stale+=1
            if stale>=self.config['patience']:break
        if best_state is None:raise FloatingPointError('No finite checkpoint')
        self.model.load_state_dict(best_state);self.is_fitted=True
        self._init_explainer(xt[:min(200,len(xt))])

    def predict_risk(self,x):
        if self.recipe=='FullBase':return self._cox_risk(x)
        return self._analytic_rmst(x)

    def final_prior_diagnostics(self,x):
        from texgisa_survival.texgi_utils import texgi_time_series
        z=torch.as_tensor(self.scaler.transform(x[:12]),dtype=torch.float32).requires_grad_(True)
        self.model.eval()
        path=texgi_time_series(self.model,z,torch.as_tensor(self.baseline_reference_).expand_as(z),
            M=self.config['ig_steps'],t_sample=self.config['prior_time_samples'],multiply_by_inputs=False)
        raw=self.model.raw_logits(z)/self.model.temperature+self.model.logit_bias
        risk=(self.model.risk(z) if self.recipe=='FullBase'
              else -curve_rmst(raw,torch.as_tensor(self.time_bins,dtype=z.dtype))/float(self.time_bins[-1]))
        gradient=torch.autograd.grad(risk.sum(),z)[0]
        if not torch.isfinite(path).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite post-calibration attribution or risk gradient')
        records=[]
        for rule in self.rules:
            j=self.feature_names.index(rule['feature'])
            mask=(z[:,j]>self.rule_reference_['median'][j] if rule['condition']=='above_training_median'
                  else torch.ones(len(z),dtype=torch.bool))
            records.append(dict(feature=rule['feature'],active_rows=int(mask.sum()),
                path_sign_violation=(float((rule['sign']*path[:,mask,j]<0).float().mean()) if mask.any() else None),
                final_risk_sign_violation=(float((rule['sign']*gradient[mask,j]<0).float().mean()) if mask.any() else None),
                path_margin_shortfall=(float(F.relu(.001-rule['sign']*path[:,mask,j]).mean()) if mask.any() else None),
                risk_margin_shortfall=(float(F.relu(.001-rule['sign']*gradient[mask,j]).mean()) if mask.any() else None)))
        step=max(1,self.num_time_bins//self.config['prior_time_samples'])
        indices=list(range(0,self.num_time_bins,step))[:self.config['prior_time_samples']]
        return dict(n_rows=len(z),time_gradient_shape=list(path.shape),finite=bool(torch.isfinite(path).all()),
                    time_bin_indices=indices,risk_gradient_finite=True,
                    risk_gradient_units='raw Cox head, or RMST divided by fitted horizon; before positive risk z-scoring',
                    rules=records,stage='after calibration; diagnostic only, never used for model selection')


def fit_full(method,c,x,t,e,query,grid,seed,split_seed):
    from .adapters import Prediction,_standardize_risk
    from .data import preprocess,neural_partitions,duplicate_groups
    core,stop,cal=neural_partitions(e,split_seed,True,groups=duplicate_groups(x,t,e))
    (xf,xs,xc,xq),transformer=preprocess(x.iloc[core],x.iloc[stop],x.iloc[cal],query,return_transformer=True)
    # Author CSVs are all numerical and their order is preserved by preprocessing.
    names=list(x.columns)
    if xf.shape[1]!=len(names):raise ValueError('Named prior feature mapping changed')
    priors=json.loads((Path(__file__).resolve().parents[1]/'priors.json').read_text())
    dataset=x.attrs['dataset'];rules=priors[dataset]['rules']
    model=FullModel(method,c,rules,names,hidden_layers=c['hidden'],num_time_bins=c['bins'],
        dropout=c['dropout'],architecture='mlp',norm='layer',activation='selu',likelihood='pch',
        lambda_cox=.3,calibrate=False,ema_decay=0.,input_noise=0.,random_state=seed,device='cpu')
    model.fit_full(xf,t[core],e[core],(xs,t[stop],e[stop]),seed)
    model._fit_calibration((xc,t[cal],e[cal]))
    final_prior=dict(fitting=model.final_prior_diagnostics(xf),query=model.final_prior_diagnostics(xq))
    risk=_standardize_risk(model.predict_risk(xf),model.predict_risk(xq))
    survival=model.predict_survival(xq,times=grid)
    if c.get('_checkpoint'):
        torch.save(dict(state=model.model.state_dict(),scaler=model.scaler,preprocessor=transformer,
            time_bins=model.time_bins,feature_names=names,recipe=method,config=c,
            baseline_reference=model.baseline_reference_,rule_reference=model.rule_reference_,prior_source=priors[dataset],
            temperature=float(model.model.temperature),logit_bias=float(model.model.logit_bias)),c['_checkpoint'])
    return Prediction(risk,survival,dict(seed=seed,n_fit=len(core),n_stop=len(stop),n_calibration=len(cal),
        fit_rows=core.tolist(),stop_rows=stop.tolist(),calibration_rows=cal.tolist(),
        recipe=method,prior_status=priors[dataset],prior_audit=model.prior_audit,training_trace=model.trace,
        attribution_prior_training_enabled=method!='FullNoPrior',
        directional_prior_training_enabled=method!='FullNoPrior' and bool(rules),
        baseline_reference=model.baseline_reference_.tolist(),rule_reference_median=model.rule_reference_['median'].tolist(),
        final_prior_diagnostics=final_prior,pair_sampling='All comparable pairs within shuffled minibatches; not full-core risk sets',
        calibration=model.calibration_record,risk_target=('Cox head' if method=='FullBase' else 'calibrated curve analytic RMST')))
