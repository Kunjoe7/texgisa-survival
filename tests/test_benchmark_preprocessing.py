# Copyright (c) 2026 DHAI Lab
# Licensed under the MIT License; see LICENSE.

"""The revised benchmark must learn transformations inside fitting splits."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_preprocessing_excludes_validation_and_test_observations():
    pytest.importorskip('pycox')
    path=Path(__file__).resolve().parents[1]/'benchmarks/run_benchmark.py'
    spec=importlib.util.spec_from_file_location('benchmark_preprocessing',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training=pd.DataFrame({'value':[1.,3.,np.nan,999.], 'category':['a','b','a','new']})
    heldout=pd.DataFrame({'value':[1000.], 'category':['unseen']})
    a,b=module.preprocess(training,heldout,fit_rows=[0,1,2])
    assert np.allclose(a[:3,0],[-np.sqrt(1.5),np.sqrt(1.5),0.])
    assert a.shape==(4,2) and b.shape==(1,2)
    assert a[3,1]==0 and b[0,1]==0
