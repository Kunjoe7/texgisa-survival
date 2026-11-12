"""Four own-method hypotheses inspired by recent tabular/calibration work.
No baseline/package implementation is imported or modified here.
"""
import math
import numpy as np
import torch
from torch import nn
from .ibs_upgrade import UpgradeModel,UpgradeNetwork
from .frontier import _tower
from .strengthened import fit_strengthened
METHODS={'RefineSmooth','RefineSparse','RefineCDF','RefineSmoothCDF'}

class SmoothFeatures(nn.Module):
    def __init__(self,features):super().__init__();self.output_features=4*features
    def forward(self,x):return torch.cat([x,torch.tanh(.5*x),torch.tanh(x),torch.tanh(2*x)],dim=1)

def warp_log_rates(log_rates,alpha,beta):
    """Boundary cumulative hazard f(H)=alpha*H+beta*log1p(H).
    Positive alpha, nonnegative beta guarantee positive interval increments.
    Exact at boundaries; within bins the output remains a PCH model.
    """
    increment=log_rates.double().exp();cumulative=increment.cumsum(1)
    previous=torch.cat([torch.zeros_like(cumulative[:,:1]),cumulative[:,:-1]],1)
    transformed=alpha*increment+beta*torch.log1p(increment/(1.+previous))
    return transformed.log()

class Round02Network(UpgradeNetwork):
    def __init__(self,features,config,initial,edges,horizon,point_scaling,important):
        config=dict(config,upgrade_variant='PriorIBSAlign');super().__init__(features,config,initial,edges,horizon,point_scaling,important)
        self.research_variant=config['round02_variant'];self.use_warp=self.research_variant in {'RefineCDF','RefineSmoothCDF'}
        if self.research_variant in {'RefineSmooth','RefineSmoothCDF','RefineSparse'}:
            self.embedding=nn.Identity() if self.research_variant=='RefineSparse' else SmoothFeatures(features)
            width=features if self.research_variant=='RefineSparse' else self.embedding.output_features
            self.rank_features=_tower(width,config);self.curve_features=_tower(width,config)
        if self.research_variant=='RefineSparse':
            logit=math.log(config.get('residual_gate_init',.1)/(1.-config.get('residual_gate_init',.1)))
            self.rank_residual_gate=nn.Parameter(torch.tensor(logit));self.curve_residual_gate=nn.Parameter(torch.tensor(logit))
        self.register_buffer('warp_alpha',torch.tensor(1.,dtype=torch.float64));self.register_buffer('warp_beta',torch.tensor(0.,dtype=torch.float64))
    def rank_parameters(self):
        params=list(super().rank_parameters())
        if self.research_variant=='RefineSparse':params.append(self.rank_residual_gate)
        return tuple(params)
    def _rank(self,x,embedded):
        if self.research_variant!='RefineSparse':return super()._rank(x,embedded)
        raw=(torch.sigmoid(self.rank_residual_gate)*self.rank_head(self.rank_features(embedded))+self.rank_skip(x)).squeeze(1)
        return self.risk_scale*self.rank_bound*torch.tanh(raw/self.rank_bound)
    def latent_components(self,x):
        if self.research_variant!='RefineSparse':return super().latent_components(x)
        rank=self._rank(x,self.rank_encoded(x));hidden=self.curve_features(self.embedding(x));gate=torch.sigmoid(self.curve_residual_gate)
        point=self.contrast_bound*torch.tanh((gate*self.point_head(hidden)+self.point_skip(x)).squeeze(1)/self.contrast_bound)
        residual=gate*self.residual_bound*torch.tanh(self.residual_head(hidden)/self.residual_bound)
        baseline=self.initial_log_rates+self.baseline_shift_bound*torch.tanh(self.baseline_delta/self.baseline_shift_bound)
        return baseline,rank,point,residual
    def from_components(self,components,temperature=None,bias=None,*,point_temperature=None,point_bias=None,detach_rank=False):
        rates,point,contrast=super().from_components(components,temperature,bias,point_temperature=point_temperature,point_bias=point_bias,detach_rank=detach_rank)
        if self.use_warp:rates=warp_log_rates(rates,self.warp_alpha,self.warp_beta)
        return rates,point,contrast

class Round02Model(UpgradeModel):
    def __init__(self,method,config,rules,feature_names,random_state=0):
        super().__init__(method,dict(config,upgrade_variant='PriorIBSAlign'),rules,feature_names,random_state)
        self.research_variant=config['round02_variant']
    def _create_network(self,features,initial,edges,horizon):return Round02Network(features,self.config,initial,edges,horizon,self.point_scaling_,self.gene_mask)
    def prior_loss(self,x):
        loss,diagnostic=super().prior_loss(x)
        if self.research_variant=='RefineSparse':
            weights=torch.where(torch.as_tensor(self.gene_mask,device=x.device),.5,1.)
            sparse=((self.model.rank_skip.weight.abs()+self.model.point_skip.weight.abs())*weights[None,:]).mean()
            loss=loss+float(self.config.get('lambda_skip_l1',.01))*sparse;diagnostic['own_skip_l1']=float(sparse.detach())
        return loss,diagnostic
    def _fit_calibration(self,validation_data,steps=None):
        super()._fit_calibration(validation_data,steps)
        if not self.model.use_warp:return
        previous_record=self.calibration_record;x,t,e=validation_data;self.model.eval();xt=torch.as_tensor(self.scaler.transform(x),dtype=torch.float32)
        with torch.no_grad():base=self.model.log_rates(xt).detach();rank_before=self.model.risk(xt).clone()
        log_a=torch.tensor(0.,dtype=torch.float64,requires_grad=True);beta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.Adam([log_a,beta],lr=.03);shrink=float(self.config.get('warp_shrinkage',.0001))
        def objective():
            ibs=self._brier(warp_log_rates(base,log_a.exp(),beta),t,e)
            return ibs+shrink*((log_a.exp()-1.).square()+beta.square()),ibs
        with torch.no_grad():initial,initial_ibs=[float(v) for v in objective()]
        best=initial;best_ibs=initial_ibs;saved=(0.,0.);accepted=0
        for _ in range(int(self.config.get('warp_steps',120) if steps is None else steps)):
            optimizer.zero_grad();loss=objective()[0]
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite own monotone warp')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad) for p in [log_a,beta]):raise FloatingPointError('Nonfinite monotone warp derivative')
            optimizer.step()
            with torch.no_grad():
                log_a.clamp_(math.log(.25),math.log(4.));beta.clamp_(0.,4.);value,ibs=[float(v) for v in objective()]
                if np.isfinite(value) and value<best:best,best_ibs=value,ibs;saved=(float(log_a),float(beta));accepted+=1
        with torch.no_grad():
            self.model.warp_alpha.fill_(math.exp(saved[0]));self.model.warp_beta.fill_(saved[1]);rank_after=self.model.risk(xt)
        assert torch.equal(rank_before,rank_after)
        self.calibration_record=dict(objective='training-reserved calibration IPCW IBS + identity shrinkage',initial_objective=initial,final_objective=best,initial_brier=initial_ibs,final_brier=best_ibs,
            alpha=math.exp(saved[0]),beta=saved[1],accepted_improvements=accepted,rank_unchanged=True,previous_scalar_calibration=previous_record,
            transformation='H -> alpha*H + beta*log1p(H), alpha>0,beta>=0; positive PCH increments; not KSP/CSD or a conformal guarantee')
    def extra_metadata(self):
        record=super().extra_metadata();record.update(experiment_variant=self.research_variant,gene_log1p=False,low_rank_active=False,full_input_feature_identity_preserved=True,
            representation=('raw + smooth tanh(.5x,x,2x)' if self.research_variant in {'RefineSmooth','RefineSmoothCDF'} else 'raw with sparse/gated residual' if self.research_variant=='RefineSparse' else 'previous fixed Fourier representation'),
            proper_survival_monotone_warp=self.model.use_warp,expert_and_temporal_priors_retained=True)
        return record

def fit_refined(method,config,x,t,e,query,grid,seed,split_seed):
    assert method in METHODS;config=dict(config,round02_variant=method,upgrade_variant='PriorIBSAlign',_gene_input=x.attrs['dataset']=='tcga')
    return fit_strengthened('MetricOutcome',config,x,t,e,query,grid,seed,split_seed,_model_class=Round02Model)
