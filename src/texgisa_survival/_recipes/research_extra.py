"""Additional own-only ideas. Existing original-package baselines stay frozen."""
import math
import numpy as np
import torch
from torch import nn
from .research_round2 import Round02Network,Round02Model
from .improved import survival_from_log_rates
from .strengthened import fit_strengthened
METHODS={'RefineRiskCal','RefineTimeBasis','RefineEventBins','RefineHybridScore'}

def integrated_bernoulli_score(survival,times,events,grid,event_g,grid_g):
    dtype=survival.dtype;device=survival.device
    t=torch.as_tensor(times,dtype=dtype,device=device);e=torch.as_tensor(events,dtype=dtype,device=device);u=torch.as_tensor(grid,dtype=dtype,device=device)
    g=torch.as_tensor(event_g,dtype=dtype,device=device);gu=torch.as_tensor(grid_g,dtype=dtype,device=device)
    observed=(e[:,None]==1)&(t[:,None]<=u);alive=t[:,None]>u
    if ((g<=0)&observed.any(1)).any() or (gu<=0).any():raise ValueError('Log-score censoring positivity violated')
    safe=torch.where(g>0,g,torch.ones_like(g));prob=survival.clamp(1e-8,1.-1e-8)
    loss=-observed.to(dtype)*torch.log1p(-prob)/safe[:,None]-alive.to(dtype)*prob.log()/gu[None,:]
    return torch.trapezoid(loss.mean(0),u)/(u[-1]-u[0])

class ExtraNetwork(Round02Network):
    def __init__(self,features,config,initial,edges,horizon,point_scaling,important):
        super().__init__(features,config,initial,edges,horizon,point_scaling,important)
        self.register_buffer('riskcal_coefficients',torch.zeros(2,dtype=torch.float64))
        self.register_buffer('interval_log_width',torch.as_tensor(np.log(np.diff(edges)),dtype=torch.float64))
        if self.research_variant=='RefineTimeBasis':
            count=len(initial);k=min(int(config.get('temporal_basis_rank',6)),max(1,count-1))
            center=(np.asarray(edges[:-1])+np.asarray(edges[1:]))/(2*float(edges[-1]));matrix=torch.cos(torch.as_tensor(center[:,None]*np.arange(1,k+1)[None,:]*math.pi,dtype=torch.float64))
            basis=torch.linalg.qr(matrix,mode='reduced').Q.float();self.register_buffer('extra_time_basis',basis)
            self.residual_head=nn.Linear(config['hidden'][-1],k);nn.init.normal_(self.residual_head.weight,std=.01);nn.init.zeros_(self.residual_head.bias)
    def latent_components(self,x):
        components=list(super().latent_components(x))
        if self.research_variant=='RefineTimeBasis':components[3]=(components[3]@self.extra_time_basis.T)/math.sqrt(self.extra_time_basis.shape[1])
        return tuple(components)
    def from_components(self,components,temperature=None,bias=None,*,point_temperature=None,point_bias=None,detach_rank=False):
        rates,point,contrast=super().from_components(components,temperature,bias,point_temperature=point_temperature,point_bias=point_bias,detach_rank=detach_rank)
        if self.research_variant=='RefineRiskCal':
            rank=components[1].detach() if detach_rank else components[1]
            z=torch.tanh(rank.double()/self.risk_scale)
            rates=rates+self.riskcal_coefficients[0]+self.riskcal_coefficients[1]*z[:,None]
        return rates,point,contrast

class ExtraModel(Round02Model):
    def _create_network(self,features,initial,edges,horizon):return ExtraNetwork(features,self.config,initial,edges,horizon,self.point_scaling_,self.gene_mask)
    def _make_time_bins(self,times,events):
        if self.research_variant!='RefineEventBins':return super()._make_time_bins(times,events)
        times=np.asarray(times,float);events=np.asarray(events,int);observed=times[events==1]
        if not len(observed):raise ValueError('Event-quantile grid requires optimization-training events')
        bins=int(self.config.get('event_bins',32));edges=np.unique(np.r_[0.,np.quantile(observed,np.linspace(0,1,bins+1)),float(times.max())])
        if len(edges)<2 or edges[0]!=0 or np.any(np.diff(edges)<=0):raise ValueError('Invalid fitting-only event grid')
        self.event_grid_audit=dict(source='current optimization-training event times only; endpoint=max optimization duration',event_count=len(observed),requested_bins=bins,actual_intervals=len(edges)-1)
        return edges
    def _curve_parts(self,x,times,events):
        total,nll,brier,point=super()._curve_parts(x,times,events)
        if self.research_variant=='RefineTimeBasis':
            rates=self.model.loss_representation(x)[0];intensity=rates.double()-self.model.interval_log_width[None,:]
            if intensity.shape[1]>1:total=total+float(self.config.get('intensity_smoothing',.001))*torch.diff(intensity,dim=1).square().mean()
        if self.research_variant=='RefineHybridScore':
            rates=self.model.loss_representation(x)[0];survival=survival_from_log_rates(rates,self.time_bins,self.brier_grid_)
            g=self.censoring_.predict(torch.as_tensor(times).detach().cpu().numpy(),left=True)
            score=integrated_bernoulli_score(survival,times,events,self.brier_grid_,g,self.grid_g_)
            total=total+float(self.config.get('integrated_log_weight',.25))*score
        return total,nll,brier,point
    def _fit_calibration(self,validation_data,steps=None):
        super()._fit_calibration(validation_data,steps)
        if self.research_variant!='RefineRiskCal':return
        prior=self.calibration_record;x,t,e=validation_data;xt=torch.as_tensor(self.scaler.transform(x),dtype=torch.float32);self.model.eval()
        with torch.no_grad():base=self.model.log_rates(xt).detach();rank=self.model.risk(xt).detach();z=torch.tanh(rank.double()/self.model.risk_scale)
        coefficients=torch.zeros(2,dtype=torch.float64,requires_grad=True);optimizer=torch.optim.Adam([coefficients],lr=.03);weight=float(self.config.get('riskcal_shrinkage',.01))
        def objective():
            rates=base+coefficients[0]+coefficients[1]*z[:,None];ibs=self._brier(rates,t,e)
            return ibs+weight*coefficients.square().mean(),ibs
        with torch.no_grad():initial,initial_ibs=[float(v) for v in objective()]
        best,best_ibs=initial,initial_ibs;saved=torch.zeros(2,dtype=torch.float64);accepted=0
        for _ in range(int(self.config.get('calibration_steps',80) if steps is None else steps)):
            optimizer.zero_grad();loss=objective()[0]
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite risk-conditioned calibration')
            loss.backward()
            if coefficients.grad is None or not torch.isfinite(coefficients.grad).all():raise FloatingPointError('Nonfinite risk calibration gradient')
            optimizer.step()
            with torch.no_grad():
                coefficients[0].clamp_(-2.,2.);coefficients[1].clamp_(-.5,.5);value,ibs=[float(v) for v in objective()]
                if np.isfinite(value) and value<best:best,best_ibs=value,ibs;saved=coefficients.detach().clone();accepted+=1
        with torch.no_grad():self.model.riskcal_coefficients.copy_(saved);after=self.model.risk(xt)
        assert torch.equal(rank,after)
        self.calibration_record=dict(objective='reserved training IBS + risk-conditioning shrinkage',initial_objective=initial,final_objective=best,initial_brier=initial_ibs,final_brier=best_ibs,coefficients=saved.tolist(),accepted_improvements=accepted,rank_unchanged=True,previous_scalar_calibration=prior,conditional_calibration_guarantee_claimed=False)
    def extra_metadata(self):
        r=super().extra_metadata();r.update(extra_idea=self.research_variant,event_grid=getattr(self,'event_grid_audit',None),temporal_basis_rank=int(self.model.extra_time_basis.shape[1]) if hasattr(self.model,'extra_time_basis') else None,
            hybrid_log_score=self.research_variant=='RefineHybridScore',risk_conditioning=self.research_variant=='RefineRiskCal',input_feature_projection=False,additional_baseline_training=False)
        return r

def fit_extra(method,config,x,t,e,query,grid,seed,split_seed):
    assert method in METHODS;config=dict(config,round02_variant=method,upgrade_variant='PriorIBSAlign',_gene_input=x.attrs['dataset']=='tcga')
    return fit_strengthened('MetricOutcome',config,x,t,e,query,grid,seed,split_seed,_model_class=ExtraModel)
