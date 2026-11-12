# Copyright (c) 2026 DHAI Lab. Licensed under the MIT License.
"""Fitted research recipes with explicit risk, curve and time prediction outputs.

The recipe mathematics are copied from the completed development campaigns.
This interface retains the fitted object rather than repeating fitting on queries.
"""
from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def available_recipes():
    return ('RefineCDF', 'RefineRiskCal', 'RefineSmooth', 'RefineSparse',
            'RefineSmoothCDF', 'RefineTimeBasis', 'RefineEventBins', 'RefineHybridScore')


class SurvivalRecipe(BaseEstimator):
    """Right-censored numeric-data recipe; all fitting partitions are internal.

    Parameters
    ----------
    recipe : str
        A name returned by :func:`available_recipes`.
    dataset : str
        Named benchmark task for its frozen prior, or ``custom`` (structural
        time-gradient priors without invented feature directions).
    config_id : int
        Index 0 or 1 in the frozen search grid. This does not tune on test data.
    config : dict or None
        Explicit training overrides. The selected effective configuration is
        saved in ``config_``. Benchmark reproduction uses its own saved splits.
    random_state : int
        Initialization and internal fitting partition seed.
    threads : int
        CPU thread count. Current recipes are validated on CPU.
    """
    def __init__(self, recipe='RefineCDF', dataset='custom', config_id=0,
                 config=None, random_state=271, threads=1):
        self.recipe=recipe; self.dataset=dataset; self.config_id=config_id
        self.config=config; self.random_state=random_state; self.threads=threads

    def _frame(self, X, fitting=False):
        if isinstance(X, pd.DataFrame):
            frame=X.copy()
        else:
            array=np.asarray(X, dtype=float)
            if array.ndim != 2:
                raise ValueError('X must be a numeric matrix')
            frame=pd.DataFrame(array, columns=[f'x{i}' for i in range(array.shape[1])])
        if frame.columns.has_duplicates or frame.shape[1]==0:
            raise ValueError('Unique, nonempty feature names are required')
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Provide finite numeric features; encode categories explicitly')
        if not fitting and list(frame.columns)!=list(self.feature_names_in_):
            raise ValueError('Prediction feature names/order differ from fitting')
        frame.attrs['dataset']=self.dataset
        return frame

    def fit(self, X, time, event):
        import torch
        from ._recipes.strengthened import fit_strengthened
        from ._recipes.research_round2 import Round02Model
        from ._recipes.research_extra import ExtraModel
        from ._recipes.metrics import evaluation_grid
        if self.recipe not in available_recipes():
            raise ValueError('Unknown recipe: '+str(self.recipe))
        root=Path(__file__).resolve().parent
        priors=json.loads((root/'priors.json').read_text())
        if self.dataset not in priors:
            raise ValueError('Unknown prior task; use dataset="custom" for other data')
        if self.config_id not in (0,1):
            raise ValueError('config_id must be 0 or 1')
        frame=self._frame(X, fitting=True)
        time=np.asarray(time,dtype=float);event=np.asarray(event)
        if time.shape!=(len(frame),) or event.shape!=time.shape or not np.isfinite(time).all() or (time<0).any() or not np.isin(event,[0,1]).all():
            raise ValueError('Aligned nonnegative times and binary events are required')
        for rule in priors[self.dataset]['rules']:
            if rule['feature'] not in frame.columns:
                raise ValueError('Missing named prior feature: '+rule['feature'])
        grids=json.loads((root/'recipe_configs.json').read_text())
        config=copy.deepcopy(grids[self.recipe][self.config_id]);config.update(self.config or {})
        config.update(round02_variant=self.recipe,upgrade_variant='PriorIBSAlign',_gene_input=self.dataset=='tcga')
        if '_checkpoint' in config or '_stage_epochs' in config:
            raise ValueError('Checkpoint/refit controls belong to the reproduction runner')
        torch.set_num_threads(self.threads);torch.manual_seed(self.random_state)
        np.random.seed(self.random_state);random.seed(self.random_state)
        grid=evaluation_grid(time,event)
        cls=Round02Model if self.recipe in {'RefineCDF','RefineSmooth','RefineSparse','RefineSmoothCDF'} else ExtraModel
        first,model,transformer,center,scale=fit_strengthened(
            'MetricOutcome',config,frame,time,event,frame.iloc[:1],grid,
            self.random_state,self.random_state+271828,_model_class=cls,_return_fitted=True)
        # Match the completed campaign: fresh fixed-epoch refit on core+stop;
        # calibration rows remain disjoint and are never merged into fitting.
        epochs=int(first.metadata['best_epochs'])
        torch.manual_seed(self.random_state);np.random.seed(self.random_state);random.seed(self.random_state)
        prediction,model,transformer,center,scale=fit_strengthened(
            'MetricOutcome',dict(config,_stage_epochs=epochs),frame,time,event,
            frame.iloc[:1],grid,self.random_state,self.random_state+271828,
            _model_class=cls,_return_fitted=True)
        prediction.metadata['stage1_epoch_selection']={
            'best_epochs':epochs,'fit_rows':first.metadata['fit_rows'],
            'stop_rows':first.metadata['stop_rows'],
            'criterion':first.metadata['selection_criterion']}
        self.model_=model;self.preprocessor_=transformer
        self.risk_center_=center;self.risk_scale_=scale
        self.feature_names_in_=np.asarray(frame.columns,dtype=object)
        self.n_features_in_=frame.shape[1];self.time_grid_=grid
        self.config_=config;self.metadata_=prediction.metadata
        return self

    def _transform(self, X):
        check_is_fitted(self,'model_')
        return self.preprocessor_.transform(self._frame(X)).astype('float32')

    def predict_risk(self,X):
        """Risk score: larger means earlier event; independent of curve calibration."""
        return (self.model_.predict_risk(self._transform(X))-self.risk_center_)/self.risk_scale_

    def predict_survival(self,X,times=None):
        """Return a subjects-by-times probability matrix on a sorted nonnegative grid."""
        grid=np.asarray(self.time_grid_ if times is None else times,dtype=float)
        if grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or (grid<0).any() or (np.diff(grid)<0).any():
            raise ValueError('times must be a finite sorted nonnegative vector')
        return self.model_.predict_survival(self._transform(X),times=grid)

    def predict_time(self,X):
        """Positive auxiliary point estimate in original endpoint units, not curve median."""
        return self.model_.predict_point_time(self._transform(X))

    def explain(self,X,integration_steps=20,time_samples=None):
        """TEXGI of calibrated log interval cumulative hazards, shape time x row x feature."""
        import torch
        from .texgi_utils import texgi_time_series
        from ._recipes.improved import _LogRateView
        xt=self.model_._x_tensor(self._transform(X))
        baseline=torch.as_tensor(self.model_.baseline_reference_,dtype=xt.dtype).expand_as(xt)
        self.model_.model.eval()
        values=texgi_time_series(_LogRateView(self.model_.model),xt,baseline,
            M=integration_steps,t_sample=time_samples,create_graph=False,multiply_by_inputs=True)
        return values.detach().cpu().numpy()

    def save(self,path):
        """Save a trusted local fitted object; use joblib.load only on trusted artifacts."""
        import joblib
        check_is_fitted(self,'model_');joblib.dump(self,path)

