"""Unmodified public-package/ICDM2021 models; no old engine dependency.

This module does format conversion, documented train-only StandardScaler,
normal public-API calls and output packaging. It does not implement losses,
repair probabilities, derive missing survival curves, or refit a second stage.
No fitting or author-module loading happens on import.
"""
from pathlib import Path
import hashlib
import os
import importlib
import importlib.metadata
import random
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PYCOX_METHODS = ('MTLR', 'DeepSurv', 'CoxTime', 'DeepHit', 'LogisticHazard',
                 'PCHazard', 'CoxCC', 'PMF', 'NMTLR')
AFT_METHODS = ('Weibull', 'Lognormal', 'Loglogistic')
SKSURV_METHODS = ('Coxnet', 'RandomSurvivalForest', 'GradientBoostingCox')
AUTHOR_METHODS = {'HarrellNMSE': ('Harrell', 'mse'), 'HarrellNMAE': ('Harrell', 'mae'),
                  'UnoNMSE': ('ipcw', 'mse'), 'UnoNMAE': ('ipcw', 'mae')}
PACKAGE_METHODS = ('Cox',) + AFT_METHODS + SKSURV_METHODS + PYCOX_METHODS
METHODS = PACKAGE_METHODS + tuple(AUTHOR_METHODS)
AUTHOR_COMMIT = '2eee054ccbf8b8ab9750c79d640ce72ef8f836d1'
AUTHOR_HASHES = {
    'model.py': '65a14e660fbebc8bb396cbd275ac6ca7fce2167eb253ff7913b46f3b3970c693',
    'SoftCindex.py': '8acc11e5dd4d9f11869692ced411a8606faed4d84b74e1726ca8059b969c58f5',
    'surv_point_loss.py': 'f8b5232a3a7ae7effb21757d83c8c75bce15e4229c40ce6b1f9166f6c711fd76',
    'surv_data.py': 'f9b7f9886ac0648a0d9cee3fa0444b20509160585c3552863166e54aa296bb4c',
    'nonparametric.py': 'f3ece41e84801b2e4f2ac1659d3cc430f7a883bfe32f26a424cdca3014c97361',
    'util.py': 'b154dd9f19d24414dbb30caff10c5c5e9158f17360b86ee9a286318600c3c7b9',
}


def _author_folder():
    if os.environ.get('TEXGISA_ICDM2021_SOURCE'):
        return Path(os.environ['TEXGISA_ICDM2021_SOURCE']).resolve()
    bundled = ROOT / 'sources' / 'ICDM2021'
    if bundled.is_dir():
        return bundled.resolve()
    original = ROOT.parents[1] / 'work' / 'precursor_paper' / 'official_source' / ('local_global_survival-' + AUTHOR_COMMIT)
    if not original.is_dir():
        raise FileNotFoundError('Frozen ICDM2021 author source is missing; provide sources/ICDM2021')
    return original.resolve()


def mechanism_source_audit():
    """Read-only provenance; this does not import author code or fit a model."""
    folder = _author_folder()
    hashes = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in AUTHOR_HASHES}
    if hashes != AUTHOR_HASHES:
        raise RuntimeError('ICDM2021 author source differs from the frozen original')
    versions = {}
    for name in ('numpy', 'pandas', 'torch', 'torchtuples', 'pycox', 'lifelines', 'scikit-survival', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(author_repository='https://github.com/yanlirock/local_global_survival',
        author_commit=AUTHOR_COMMIT, author_source=str(folder), author_sha256=hashes,
        package_versions=versions, source_modified=False, runtime_monkey_patching=False,
        package_methods=list(PACKAGE_METHODS), author_methods=list(AUTHOR_METHODS),
        policy='BASELINE_POLICY.md; original APIs, native losses/outputs, single training stage')


def _author_modules():
    if int(np.__version__.split('.')[0]) >= 2:
        raise RuntimeError('Unmodified ICDM2021 author code requires its NumPy1.x environment; no alias injection allowed')
    folder = _author_folder()
    for name, expected in AUTHOR_HASHES.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('ICDM2021 source integrity failure: ' + name)
    # Ordinary source imports, not AST extraction, exec rewriting or aliasing.
    for name in ('model', 'SoftCindex', 'surv_point_loss', 'surv_data', 'nonparametric', 'util'):
        if name in sys.modules:
            loaded = getattr(sys.modules[name], '__file__', None)
            if loaded is None or Path(loaded).resolve().parent != folder:
                raise RuntimeError('Conflicting author import namespace: ' + name)
    sys.path.insert(0, str(folder))
    try:
        model = importlib.import_module('model')
        data = importlib.import_module('surv_data')
    finally:
        sys.path.remove(str(folder))
    for name in ('model', 'SoftCindex', 'surv_point_loss', 'surv_data', 'nonparametric', 'util'):
        if Path(sys.modules[name].__file__).resolve().parent != folder:
            raise RuntimeError('Author dependency imported outside the frozen source: ' + name)
    return model, data


def _matrix(frame):
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError('Public-package input must be a two-dimensional numeric matrix')
    return array


def _scaled(fitting, *others):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(_matrix(fitting))
    return [scaler.transform(_matrix(value)).astype('float32') for value in (fitting,) + others]


def _validation_rows(count, validation_indices):
    if validation_indices is None:
        raise ValueError('Native neural package fit requires runner-provided 80/20 validation indices')
    if isinstance(validation_indices, tuple) and len(validation_indices) == 2:
        fitting, validation = map(np.asarray, validation_indices)
    else:
        validation = np.asarray(validation_indices)
        fitting = np.setdiff1d(np.arange(count), validation)
    for rows in (fitting, validation):
        if rows.ndim != 1 or not len(rows) or rows.dtype.kind not in 'iu' or (rows < 0).any() or (rows >= count).any() or len(np.unique(rows)) != len(rows):
            raise ValueError('Invalid native fitting/validation row indices')
    if np.intersect1d(fitting, validation).size or not np.array_equal(np.sort(np.r_[fitting, validation]), np.arange(count)):
        raise ValueError('Native fitting/validation must partition all supplied training rows')
    return fitting, validation


def _fit_lifelines(method, config, X, T, E, query):
    from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    columns = ['x' + str(i) for i in range(_matrix(X).shape[1])]
    frame = pd.DataFrame(_matrix(X), columns=columns)
    frame['duration'], frame['event'] = T, E
    qframe = pd.DataFrame(_matrix(query), columns=columns)
    if method == 'Cox':
        model = CoxPHFitter(penalizer=config['pen'], l1_ratio=config['l1'])
    else:
        cls = {'Weibull': WeibullAFTFitter, 'Lognormal': LogNormalAFTFitter,
               'Loglogistic': LogLogisticAFTFitter}[method]
        model = cls(penalizer=config['pen'])
    # No QR/rank reduction, Newton override, positive-time shift or fit retry.
    model.fit(frame, duration_col='duration', event_col='event')
    curve = model.predict_survival_function(qframe)
    point = model.predict_median(qframe).to_numpy()
    risk = model.predict_log_partial_hazard(qframe).to_numpy() if method == 'Cox' else -point
    return dict(risk=risk, survival=curve, point=point, metadata=dict(
        package='lifelines', estimator=type(model).__name__, preprocessing='none; estimator defaults',
        fit_rows=list(range(len(T))), validation_rows=[],
        risk_output='predict_log_partial_hazard' if method == 'Cox' else 'negative native predict_median',
        point_output='native predict_median; infinities retained as returned',
        survival_output='native predict_survival_function on its original timeline',
        zero_time_policy='unchanged durations; package rejection is retained'))


def _fit_sksurv(method, config, X, T, E, query, seed, threads):
    from sksurv.util import Surv
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    if method == 'Coxnet':
        fitting, q = _scaled(X, query)
        model = CoxnetSurvivalAnalysis(alphas=[config['alpha']], l1_ratio=config['l1_ratio'], fit_baseline_model=True)
        preprocessing = 'sklearn StandardScaler fitted on training rows; no imputation/QR'
    elif method == 'RandomSurvivalForest':
        fitting, q = _matrix(X), _matrix(query)
        model = RandomSurvivalForest(**config, random_state=seed, n_jobs=threads)
        preprocessing = 'none'
    else:
        fitting, q = _matrix(X), _matrix(query)
        model = GradientBoostingSurvivalAnalysis(**config, loss='coxph', random_state=seed)
        preprocessing = 'none'
    model.fit(fitting, Surv.from_arrays(np.asarray(E, dtype=bool), T))
    risk = model.predict(q)
    # Public return_array and unique_times_ only convert the native result to
    # the common DataFrame container. No new times, extrapolation or clipping.
    curve = pd.DataFrame(model.predict_survival_function(q, return_array=True).T, index=model.unique_times_)
    return dict(risk=risk, survival=curve, point=None, metadata=dict(
        package='scikit-survival', estimator=type(model).__name__, preprocessing=preprocessing,
        fit_rows=list(range(len(T))), validation_rows=[], risk_output='native predict',
        point_output=None, point_unavailable_reason='No native point-time prediction API invoked',
        survival_output='native predict_survival_function(return_array=True) at unique_times_'))


def _fit_pycox(method, config, X, T, E, query, validation_indices):
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH, CoxCC, CoxTime, MTLR, PMF, DeepHitSingle, LogisticHazard, PCHazard
    from pycox.models.cox_time import MLPVanillaCoxTime
    fitting, validation = _validation_rows(len(T), validation_indices)
    xf, xv, xq = _scaled(np.asarray(X)[fitting], np.asarray(X)[validation], query)
    tf, ef, tv, ev = T[fitting], E[fitting], T[validation], E[validation]
    optimizer = (tt.optim.AdamW(lr=config['lr'], decoupled_weight_decay=config['wd'])
                 if config.get('wd') else tt.optim.Adam(lr=config['lr']))
    def network(outputs):
        return tt.practical.MLPVanilla(xf.shape[1], config['hidden'], outputs,
                                      batch_norm=True, dropout=config['dropout'])
    if method in ('DeepSurv', 'CoxCC'):
        cls = CoxPH if method == 'DeepSurv' else CoxCC
        model = cls(network(1), optimizer, device='cpu')
        yf, yv = (tf.astype('float32'), ef.astype('float32')), (tv.astype('float32'), ev.astype('float32'))
    elif method == 'CoxTime':
        label = CoxTime.label_transform()
        yf, yv = label.fit_transform(tf.astype('float32'), ef.astype('float32')), label.transform(tv.astype('float32'), ev.astype('float32'))
        body = MLPVanillaCoxTime(xf.shape[1], config['hidden'], batch_norm=True, dropout=config['dropout'])
        model = CoxTime(body, optimizer, labtrans=label, device='cpu')
    else:
        cls = {'MTLR': MTLR, 'NMTLR': MTLR, 'PMF': PMF, 'DeepHit': DeepHitSingle,
               'LogisticHazard': LogisticHazard, 'PCHazard': PCHazard}[method]
        label = cls.label_transform(config['bins'])
        yf, yv = label.fit_transform(tf, ef), label.transform(tv, ev)
        body = torch.nn.Linear(xf.shape[1], label.out_features) if method == 'MTLR' else network(label.out_features)
        extra = dict(alpha=config['alpha'], sigma=config['sigma']) if method == 'DeepHit' else {}
        if method == 'PCHazard':
            extra['sub'] = 10
        model = cls(body, optimizer, duration_index=label.cuts, device='cpu', **extra)
    history = model.fit(xf, yf, batch_size=config['batch'], epochs=config['epochs'], verbose=False,
        callbacks=[tt.callbacks.EarlyStopping(patience=config['patience'])], val_data=(xv, yv))
    if method in ('DeepSurv', 'CoxCC', 'CoxTime'):
        model.compute_baseline_hazards()
    predictor = model.interpolate(10) if method in ('MTLR', 'NMTLR', 'PMF', 'DeepHit', 'LogisticHazard') else model
    curve = predictor.predict_surv_df(xq)
    risk = model.predict(xq).reshape(-1) if method in ('DeepSurv', 'CoxCC') else None
    log = history.to_pandas()
    return dict(risk=risk, survival=curve, point=None, metadata=dict(
        package='pycox/torchtuples', estimator=type(model).__name__,
        architecture='linear MTLR' if method == 'MTLR' else 'native network construction',
        fit_rows=fitting.tolist(), validation_rows=validation.tolist(),
        preprocessing='sklearn StandardScaler fitted only on80% fitting rows; no imputation/QR',
        optimizer='native AdamW decoupled_weight_decay=config.wd' if config.get('wd') else 'native Adam',
        loss='unchanged package default', training='single stage; native EarlyStopping restores best validation weights; no refit',
        actual_epochs=len(log), native_training_log=log.reset_index().to_dict(orient='records'),
        risk_output='native scalar predict' if risk is not None else None,
        risk_unavailable_reason=None if risk is not None else 'Native multi-time model has no single risk head; no custom scalar readout added',
        point_output=None, point_unavailable_reason='No native point-time prediction API invoked',
        survival_output='native predict_surv_df; native interpolate10 or PCHazard constructor sub10 where documented',
        probability_policy='Original values returned unchanged; invalid values are not repaired'))


def _fit_author(method, config, X, T, E, query):
    import torch
    import torchtuples as tt
    from torch.utils.data import DataLoader
    author, author_data = _author_modules()
    if config and config != {'epochs': 200}:
        raise ValueError('ICDM2021 methods keep fixed original main settings; only epochs=200 config is accepted')
    fitting, q = _scaled(X, query)
    cindex_type, measure = AUTHOR_METHODS[method]
    dtype = torch.FloatTensor
    net = tt.practical.MLPVanilla(fitting.shape[1], [32, 32], 1, True, .1, output_bias=False)
    model = author.CombinedLossSurvModel(net, sigma=.01, Cindex_type=cindex_type,
        event_train=E, time_train=T, alpha=.5, measure=measure, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
    # Constructed in original main.py but never stepped there. Do not activate
    # a scheduler that the published source did not actually apply.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.95)
    dataset = author_data.SurvData(fitting, np.asarray(T, dtype=float), np.asarray(E, dtype=float))
    def cycle(loader):
        while True:
            yield from loader
    loader = iter(cycle(DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=author_data.collate_fn)))
    trace = []
    for epoch in range(200):
        losses = []
        for _ in range(len(dataset)//1024 + 1):
            model.train()
            xb, tb, eb = next(loader)
            loss, _, _, _ = model(torch.from_numpy(xb).type(dtype), tb, eb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(dict(epoch=epoch, last_original_batch_loss=losses[-1]))
    model.eval()
    with torch.no_grad():
        point = model.model(torch.from_numpy(q).type(dtype)).reshape(-1).cpu().numpy()
    return dict(risk=-point, survival=None, point=point, metadata=dict(
        package='ICDM2021 author code', source_commit=AUTHOR_COMMIT,
        source_folder=str(_author_folder()), source_sha256=dict(AUTHOR_HASHES),
        fit_rows=list(range(len(T))), validation_rows=[],
        preprocessing='sklearn StandardScaler fitted only on provided training rows; author main standardizes all columns before its CV, replaced with train-only public standardization',
        source_protocol_boundary='Common outer folds/seeds; original loss/classes/network/optimizer/iteration settings; no official main top-level dataset execution or test-epoch peeking',
        censoring_reference='Provided training outcomes only; original main passes full data outcomes, which would leak the outer test into IPCW',
        native_settings=dict(sigma=.01, alpha=.5, hidden=[32,32], batch_norm=True, dropout=.1,
            output_bias=False, optimizer='SGD', lr=.05, momentum=.9, batch=1024,
            epochs=200, clip=5., iterations_per_epoch=len(dataset)//1024+1,
            scheduler='StepLR(5,.95) constructed, never stepped, matching author main'),
        actual_epochs=200, original_training_log=trace,
        risk_output='negative original author scalar time prediction', point_output='original author scalar time prediction, unchanged',
        survival_output=None, ibs_unavailable_reason='Author method supplies no survival curve; none constructed',
        dependency_policy='Unmodified source requires isolated NumPy1.x; no np.infty injection',
        selection='fixed original main settings; final epoch; no early stopping or two-stage refit'))


def fit_native(method, config, X, T, E, query, seed, threads, validation_indices=None):
    """Fit once and expose only original public outputs, without repairing them."""
    import torch
    if method not in METHODS:
        raise ValueError('Unsupported native method: ' + str(method))
    config = {} if config is None else dict(config)
    forbidden = {'_stage_epochs', '_aux_stage_epochs', '_anchor_full', 'calibration_steps'} & set(config)
    if forbidden:
        raise ValueError('Custom pipeline configuration forbidden: ' + ','.join(sorted(forbidden)))
    T, E = np.asarray(T, dtype=np.float64), np.asarray(E)
    if len(X) != len(T) or E.shape != T.shape or not np.isin(E, [0,1]).all():
        raise ValueError('Misaligned native training outcomes')
    if _matrix(X).shape[1] != _matrix(query).shape[1]:
        raise ValueError('Native training/query feature dimensions differ')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if method in AUTHOR_METHODS:
        result = _fit_author(method, config, X, T, E, query)
    elif method in PYCOX_METHODS:
        result = _fit_pycox(method, config, X, T, E, query, validation_indices)
    elif method in SKSURV_METHODS:
        result = _fit_sksurv(method, config, X, T, E, query, seed, threads)
    else:
        result = _fit_lifelines(method, config, X, T, E, query)
    result['metadata'].update(method=method, seed=int(seed), config=config,
        source_modified=False, site_packages_modified=False, runtime_monkey_patching=False,
        custom_loss=False, custom_curve_extension=False, output_clipping=False, two_stage_refit=False)
    return result
