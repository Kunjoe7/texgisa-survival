"""Own-only nine-task hypotheses, preserving expert and time-gradient priors."""
from pathlib import Path
import copy,json,math
import numpy as np
import torch
from torch import nn
from .single_metric import SingleMetricModel,SingleMetricNetwork
from .frontier import NumericEmbedding,_tower
from .strengthened import fit_strengthened
from .improved import pch_log_rate_nll,_LogRateView
METHODS={'PriorIBSAlign','PriorCurveSeparate','PriorGeneLowRank'}

class GatedLowRankEmbedding(nn.Module):
    def __init__(self,features,config,important):
        super().__init__();width=min(features,int(config.get('low_rank_features',32)))
        target=torch.where(torch.as_tensor(important),.8,.2) if any(important) else torch.full((features,),.5)
        self.register_buffer('gate_target',target);self.gate_raw=nn.Parameter(torch.logit(target))
        self.project=nn.Linear(features,width,bias=False);nn.init.orthogonal_(self.project.weight)
        self.numeric=NumericEmbedding(width,config.get('embedding_frequencies',4));self.output_features=self.numeric.output_features
    def forward(self,x):return self.numeric(self.project(x*(.05+.95*torch.sigmoid(self.gate_raw))))
    def gate_penalty(self):return (torch.sigmoid(self.gate_raw)-self.gate_target).square().mean()

class UpgradeNetwork(SingleMetricNetwork):
    def __init__(self,features,config,initial,edges,horizon,point_scaling,important):
        super().__init__(features,config,'MetricOutcome',initial,edges,horizon,point_scaling)
        self.upgrade_variant=config['upgrade_variant'];self.separate=self.upgrade_variant!='PriorIBSAlign';self.rank_embedding=None
        if self.upgrade_variant=='PriorGeneLowRank' and features>int(config.get('low_rank_features',32)):
            self.embedding=GatedLowRankEmbedding(features,config,important);self.rank_embedding=GatedLowRankEmbedding(features,config,important)
            self.rank_features=_tower(self.embedding.output_features,config);self.curve_features=_tower(self.embedding.output_features,config)
        if self.separate:
            self.point_features=copy.deepcopy(self.curve_features);self.curve_contrast=nn.Linear(config['hidden'][-1],1)
            nn.init.normal_(self.curve_contrast.weight,std=.01);nn.init.zeros_(self.curve_contrast.bias)
        centers=torch.as_tensor((np.asarray(edges[:-1])+np.asarray(edges[1:]))/2,dtype=torch.float64);axis=centers/max(float(edges[-1]),1e-8)
        basis=torch.stack([torch.cos(math.pi*k*axis) for k in [1,2,3]],1);basis-=basis.mean(0)
        self.register_buffer('calibration_basis',basis);self.register_buffer('calibration_shape',torch.zeros(3,dtype=torch.float64))
    def rank_encoded(self,x):
        return self.rank_embedding(self.rank_localizer(x)) if self.rank_embedding is not None else super().rank_encoded(x)
    def rank_parameters(self):
        params=list(super().rank_parameters())
        if self.rank_embedding is not None:params.extend(self.rank_embedding.parameters())
        return tuple(params)
    def latent_components(self,x):
        if not self.separate:return super().latent_components(x)
        rank=self._rank(x,self.rank_encoded(x));embedded=self.embedding(self.curve_localizer(x));hidden=self.curve_features(embedded);phidden=self.point_features(embedded)
        point=self.contrast_bound*torch.tanh((self.point_head(phidden)+self.point_skip(x)).squeeze(1)/self.contrast_bound)
        residual=self.residual_bound*torch.tanh(self.residual_head(hidden)/self.residual_bound)
        baseline=self.initial_log_rates+self.baseline_shift_bound*torch.tanh(self.baseline_delta/self.baseline_shift_bound)
        contrast=self.contrast_bound*torch.tanh(self.curve_contrast(hidden).squeeze(1)/self.contrast_bound)
        return baseline,rank,point,residual,contrast
    def from_components(self,components,temperature=None,bias=None,*,point_temperature=None,point_bias=None,detach_rank=False):
        rates,point,contrast=super().from_components(components[:4],temperature,bias,point_temperature=point_temperature,point_bias=point_bias,detach_rank=detach_rank)
        if self.separate:
            baseline,rank,_,residual,curve_contrast=components;rank=rank.detach() if detach_rank else rank
            temp=self.temperature if temperature is None else temperature;shift=self.logit_bias if bias is None else bias
            offset=2.*torch.sigmoid(self.offset_logit)*rank/self.risk_scale
            rates=baseline[None,:]+offset[:,None]+(curve_contrast[:,None]+residual)/temp+shift
        return rates+(self.calibration_basis@self.calibration_shape)[None,:],point,contrast

class UpgradeModel(SingleMetricModel):
    def __init__(self,method,config,rules,feature_names,random_state=0):
        super().__init__('MetricOutcome',config,rules,feature_names,random_state);self.upgrade_variant=config['upgrade_variant'];self.gene_feedback=bool(config.get('_gene_input',False))
        selected=set(json.loads((Path(__file__).resolve().parents[1]/'data/gene_prior.json').read_text())['genes'])
        self.gene_mask=[self.gene_feedback and name in selected for name in feature_names]
        if self.gene_feedback and sum(self.gene_mask)!=474:raise ValueError('TCGA474-gene alignment changed')
    def _create_network(self,features,initial,edges,horizon):return UpgradeNetwork(features,self.config,initial,edges,horizon,self.point_scaling_,self.gene_mask)
    def _curve_parts(self,x,times,events):
        rates,point,_=self.model.loss_representation(x);nll=pch_log_rate_nll(rates,times,events,self.time_bins);brier=self._brier(rates,times,events);point_loss=self._point_loss(point,times,events)
        return float(self.config.get('curve_nll_weight',1.))*nll+self.brier_weight*brier+self.point_weight*point_loss,nll,brier,point_loss
    def _checkpoint_objective(self,x,times,events):return self._brier(self.model.log_rates(x),times,events)
    def prior_loss(self,x):
        total,diagnostics=super().prior_loss(x)
        if self.gene_feedback:
            from texgisa_survival.texgi_utils import texgi_time_series
            subset=x[:int(self.config['prior_rows'])];baseline=torch.as_tensor(self.baseline_reference_,dtype=x.dtype,device=x.device).expand_as(subset);training=self.model.training;self.model.eval()
            try:
                path=texgi_time_series(_LogRateView(self.model),subset,baseline,M=int(self.config['ig_steps']),t_sample=int(self.config['prior_time_samples']),create_graph=True,multiply_by_inputs=False)
                norms=torch.linalg.vector_norm(path*(subset-baseline).unsqueeze(0),dim=(0,1));mask=torch.as_tensor(self.gene_mask,device=x.device)
                promote=torch.relu(norms.mean()-norms[mask]);suppress=norms[~mask];normalized=self.upgrade_variant!='PriorIBSAlign'
                promote=promote.mean() if normalized else promote.sum();suppress=suppress.mean() if normalized else suppress.sum()
                total=total+float(self.config['gene_lambda_promote'])*promote+float(self.config['gene_lambda_suppress'])*suppress
                diagnostics.update(gene_promote=float(promote.detach()),gene_suppress=float(suppress.detach()),gene_count=474,gene_terms_count_normalized=normalized)
            finally:self.model.train(training)
        if isinstance(self.model.embedding,GatedLowRankEmbedding):
            gate=self.model.embedding.gate_penalty()+self.model.rank_embedding.gate_penalty();total=total+.001*gate;diagnostics['expert_gate_penalty']=float(gate.detach())
        return total,diagnostics
    def _fit_calibration(self,validation_data,steps=None):
        # Existing separate point calibration is retained. Curve calibration
        # starts at the identity and uses reserved training rows exclusively.
        super()._fit_calibration(validation_data,steps);point_record=self.calibration_record
        self.model.temperature.fill_(1.);self.model.logit_bias.zero_();self.model.calibration_shape.zero_()
        x,t,e=validation_data;xt=torch.as_tensor(self.scaler.transform(x),dtype=torch.float32);self.model.eval()
        with torch.no_grad():components=tuple(v.detach() for v in self.model.latent_components(xt));rank_before=self.model.risk(xt).clone()
        log_t=torch.tensor(0.,dtype=torch.float64,requires_grad=True);bias=torch.tensor(0.,dtype=torch.float64,requires_grad=True);shape=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        flex=self.upgrade_variant!='PriorIBSAlign';parameters=[log_t,bias]+([shape] if flex else []);optimizer=torch.optim.Adam(parameters,lr=.03)
        def objective():
            rates=self.model.from_components(components,log_t.exp(),bias)[0]
            if flex:rates=rates+(self.model.calibration_basis@shape)[None,:]
            ibs=self._brier(rates,t,e);nll=pch_log_rate_nll(rates,t,e,self.time_bins)
            regularizer=.001*(log_t.square()+bias.square())+(.01*shape.square().mean() if flex else 0.)
            return ibs+float(self.config.get('calibration_nll_weight',.05))*nll+regularizer,ibs,nll
        with torch.no_grad():initial=tuple(float(v) for v in objective())
        best=initial[0];best_values=initial;saved=(0.,0.,torch.zeros(3,dtype=torch.float64));accepted=0
        for _ in range(int(self.config.get('calibration_steps',80) if steps is None else steps)):
            optimizer.zero_grad();value=objective()[0]
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite own calibration')
            value.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('Nonfinite calibration gradient')
            optimizer.step()
            with torch.no_grad():
                log_t.clamp_(math.log(.2),math.log(5.));bias.clamp_(-3.,3.);shape.clamp_(-2.,2.);candidate=tuple(float(v) for v in objective())
                if np.isfinite(candidate).all() and candidate[0]<best:best=candidate[0];best_values=candidate;saved=(float(log_t),float(bias),shape.detach().clone());accepted+=1
        with torch.no_grad():
            self.model.temperature.fill_(math.exp(saved[0]));self.model.logit_bias.fill_(saved[1]);self.model.calibration_shape.copy_(saved[2]);rank_after=self.model.risk(xt)
        assert torch.equal(rank_before,rank_after)
        self.calibration_record=dict(objective='reserved-calibration IPCW IBS +0.05 PCH NLL + shrinkage',initial_objective=initial[0],final_objective=best,initial_brier=initial[1],final_brier=best_values[1],temperature=math.exp(saved[0]),bias=saved[1],time_shape=saved[2].tolist(),flexible_time_shape=flex,accepted_improvements=accepted,n_calibration=len(t),rank_unchanged=True,point_calibration=point_record)
    def extra_metadata(self):
        r=super().extra_metadata();r.update(upgrade_variant=self.upgrade_variant,selection_criterion='reserved_stop_IPCW_IBS_after_full_prior_warmup',checkpoint_objective='IPCW IBS',own_reverse_KM_ties_corrected=True,gene_feedback=self.gene_feedback,gene_selected_features=sum(self.gene_mask),gene_log1p=self.config.get('_gene_log1p',False),low_rank_active=isinstance(self.model.embedding,GatedLowRankEmbedding),curve_point_towers_separate=self.upgrade_variant!='PriorIBSAlign',baseline_code_or_outputs_changed=False)
        return r

def fit_upgrade(method,c,x,t,e,query,grid,seed,split_seed):
    c=dict(c,upgrade_variant=method,_gene_input=x.attrs['dataset']=='tcga');assert method in METHODS
    if method=='PriorGeneLowRank' and c['_gene_input']:
        if (x.to_numpy()<0).any() or (query.to_numpy()<0).any():raise ValueError('FPKM log1p requires nonnegative expression')
        x=x.copy();query=query.copy();x.iloc[:,:]=np.log1p(x.to_numpy());query.iloc[:,:]=np.log1p(query.to_numpy());c['_gene_log1p']=True
    return fit_strengthened('MetricOutcome',c,x,t,e,query,grid,seed,split_seed,_model_class=UpgradeModel)
