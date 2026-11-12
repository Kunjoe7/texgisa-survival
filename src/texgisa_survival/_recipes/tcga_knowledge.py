"""Current full model plus the source paper's Equation18 importance mechanism."""
import json
from pathlib import Path
import torch
from .single_metric import SingleMetricModel
from .strengthened import fit_strengthened
from .improved import _LogRateView
def importance_penalty(phi,important,lambda1=.01,lambda2=.01):
    if phi.ndim!=3:raise ValueError('Phi must be time x sample x feature')
    important=torch.as_tensor(important,dtype=torch.bool,device=phi.device)
    if important.shape!=(phi.shape[2],) or not important.any() or important.all():raise ValueError('A nonempty proper feature subset is required')
    norms=torch.linalg.vector_norm(phi,dim=(0,1));average=norms.mean()
    promote=torch.relu(average-norms[important]).sum();suppress=norms[~important].sum()
    return float(lambda1)*promote+float(lambda2)*suppress,dict(gene_promote=float(promote.detach()),gene_suppress=float(suppress.detach()),gene_attribution_norm_mean=float(average.detach()))
class TCGAKnowledgeModel(SingleMetricModel):
    def __init__(self,method,config,rules,feature_names,random_state=0):
        super().__init__('MetricOutcome',config,rules,feature_names,random_state)
        self.gene_feedback=bool(config.get('gene_feedback',False))
        prior=json.loads((Path(__file__).resolve().parents[1]/'data/gene_prior.json').read_text())
        self.gene_prior_source=prior['definition'];selected=set(prior['genes']);self.gene_mask=[v in selected for v in feature_names]
        if not any(self.gene_mask) or all(self.gene_mask):raise ValueError('Invalid gene-prior feature alignment')
    def prior_loss(self,x):
        total,diagnostics=super().prior_loss(x)
        diagnostics.update(gene_feedback_enabled=self.gene_feedback,gene_selected_features=sum(self.gene_mask))
        if not self.gene_feedback:return total,diagnostics
        from texgisa_survival.texgi_utils import texgi_time_series
        subset=x[:int(self.config['prior_rows'])];baseline=torch.as_tensor(self.baseline_reference_,dtype=x.dtype,device=x.device).expand_as(subset)
        training=self.model.training;self.model.eval()
        try:
            path=texgi_time_series(_LogRateView(self.model),subset,baseline,M=int(self.config['ig_steps']),t_sample=int(self.config['prior_time_samples']),create_graph=True,multiply_by_inputs=False)
            phi=path*(subset-baseline).unsqueeze(0)
            penalty,record=importance_penalty(phi,self.gene_mask,self.config.get('gene_lambda_promote',.01),self.config.get('gene_lambda_suppress',.01));diagnostics.update(record)
            return total+penalty,diagnostics
        finally:self.model.train(training)
    def extra_metadata(self):
        record=super().extra_metadata();record.update(tcga_gene_feedback=dict(enabled=self.gene_feedback,selected_features=sum(self.gene_mask),source=self.gene_prior_source,
            form='Paper Eq18: sum ReLU(mean feature attribution L2 norm - selected feature norm), plus sum unselected norms',
            lambda_promote=self.config.get('gene_lambda_promote',.01),lambda_suppress=self.config.get('gene_lambda_suppress',.01),
            attribution_target='Current model calibrated PCH log-rate; documented adaptation of the paper mechanism',original_clinician_474_gene_list=False))
        return record
def fit_tcga(method,c,x,t,e,query,grid,seed,split_seed):
    c=dict(c,gene_feedback=method=='TCGAGeneFeedback')
    return fit_strengthened('MetricOutcome',c,x,t,e,query,grid,seed,split_seed,_model_class=TCGAKnowledgeModel)
