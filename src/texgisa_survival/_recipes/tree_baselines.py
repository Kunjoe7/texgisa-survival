"""Official scikit-survival forest and Cox gradient boosting comparators."""
import numpy as np
from .adapters import Prediction,_standardize_risk,restricted_median
from .data import preprocess

def fit_tree(method,c,x,t,e,query,grid,seed,threads):
    from sksurv.util import Surv
    from sksurv.ensemble import RandomSurvivalForest,GradientBoostingSurvivalAnalysis
    xf,xq=preprocess(x,query,dtype=np.float64)
    if method=='RandomSurvivalForest':
        model=RandomSurvivalForest(n_estimators=c['n_estimators'],max_depth=c['max_depth'],min_samples_leaf=c['min_samples_leaf'],
            max_features='sqrt',n_jobs=threads,random_state=seed,low_memory=False)
    else:
        model=GradientBoostingSurvivalAnalysis(loss='coxph',n_estimators=c['n_estimators'],learning_rate=c['learning_rate'],
            max_depth=c['max_depth'],min_samples_leaf=c['min_samples_leaf'],subsample=c['subsample'],random_state=seed)
    model.fit(xf,Surv.from_arrays(e.astype(bool),t))
    reference,risk=model.predict(xf),model.predict(xq)
    functions=model.predict_survival_function(xq)
    native=np.unique(np.r_[0.,model.unique_times_])
    def on_grid(values):
        return np.stack([np.where(values<fn.x[0],1.,fn(np.clip(values,fn.x[0],fn.x[-1]))) for fn in functions])
    survival=on_grid(np.asarray(grid));point,missing=restricted_median(native,on_grid(native))
    return Prediction(_standardize_risk(reference,risk),survival,dict(seed=seed,n_fit=len(x),n_stop=0,n_calibration=0,
        point_time_prediction=point.tolist(),point_readout='median restricted to native fitted support',unresolved_median_count=missing,
        implementation='official scikit-survival '+type(model).__name__,fitting_stage_definition='one end-to-end estimator.fit; internal trees are iterative model components, like neural optimizer steps'))
