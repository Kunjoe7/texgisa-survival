# Licensed under the MIT License.
import numpy as np
import pandas as pd
import pytest
from texgisa_survival.recipes import SurvivalRecipe,available_recipes
from texgisa_survival.backends.runner import available_methods,fit_predict


@pytest.fixture
def sample():
    rng=np.random.default_rng(19)
    x=pd.DataFrame(rng.normal(size=(160,4)),columns=list('abcd'))
    event_time=rng.exponential(10*np.exp(-.3*x.a.to_numpy()))+.1
    censor=rng.exponential(20,size=len(x))+.1
    return x,np.minimum(event_time,censor),(event_time<=censor).astype(int)


@pytest.mark.parametrize('name',available_recipes())
def test_current_recipe_fit_and_predictions(sample,name,tmp_path):
    x,t,e=sample
    cfg=dict(epochs=3,prior_warmup=1,patience=2,ig_steps=2,prior_time_samples=2,
             prior_rows=2,hidden=[8,4],bins=8,batch=64,brier_points=8,
             calibration_steps=2,warp_steps=2)
    model=SurvivalRecipe(name,config=cfg,random_state=19).fit(x,t,e)
    risk=model.predict_risk(x[:5]);curve=model.predict_survival(x[:5],times=[0,1,3,5])
    point=model.predict_time(x[:5]);attr=model.explain(x[:2],integration_steps=2,time_samples=2)
    assert risk.shape==(5,) and curve.shape==(5,4) and point.shape==(5,)
    assert np.isfinite(risk).all() and np.isfinite(attr).all() and (point>0).all()
    assert ((curve>=0)&(curve<=1)).all() and (np.diff(curve,axis=1)<=1e-10).all()
    import joblib
    saved=tmp_path/'model.joblib';model.save(saved);loaded=joblib.load(saved)
    np.testing.assert_allclose(loaded.predict_risk(x[:5]),risk)
    with pytest.raises(ValueError):model.predict_risk(x[list('bacd')][:5])
    assert model.metadata_['expert_and_temporal_priors_retained']


def test_native_dispatch_uses_original_output(sample):
    x,t,e=sample
    out=fit_predict('Cox',x,t,e,x[:4],config={'pen':.01,'l1':0})
    assert out['risk'].shape==(4,) and out['curve'].shape[1]==4
    assert out['metadata']['source_modified'] is False
    assert out['metadata']['two_stage_refit'] is False
    assert out['metadata']['output_clipping'] is False


def test_registry_and_explicit_errors():
    assert len(available_recipes())==8
    assert {'DRO','Kernet','SODEN','HarrellNMSE','UnoNMAE','DeepHit','RandomSurvivalForest'}<=set(available_methods())
    with pytest.raises(ValueError):SurvivalRecipe('missing').fit([[1]], [1], [1])
