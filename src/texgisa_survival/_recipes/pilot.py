"""Paired group-aware nested CV, inner-only selection and durable evidence."""
from pathlib import Path
import json,hashlib,os,tempfile,time,traceback,statistics,warnings,importlib.metadata
import numpy as np
from .data import load_dataset,duplicate_groups,grouped_folds
from .adapters import fit_predict
from .metrics import checked_survival,evaluation_grid,evaluate
from .budget import FitBudget,FitBudgetError


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as handle:
        json.dump(value,handle,indent=2,allow_nan=False,ensure_ascii=False)
        handle.flush();os.fsync(handle.fileno());temporary=Path(handle.name)
    for attempt in range(41):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.1)


def source_manifest(root,protocol,spaces):
    files=[p for p in root.rglob('*.py') if 'results' not in p.relative_to(root).parts]
    files+=[root/'priors.json',root/'protocol.json',root/'search_spaces.json',root/'cache_manifest.json',root/'paper_reference.json']
    files+=list((root/'data').glob('*.csv'))
    return dict(protocol=protocol,search_spaces=spaces,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','scipy','scikit-learn','scikit-survival','lifelines','pycox','torchtuples']})


def validate_cache(root,protocol):
    from .cache import validate_cache as check
    return check(root,protocol)


def choose(trials,tolerance):
    valid=[t for t in trials if t['status']=='ok']
    if not valid:raise RuntimeError('No configuration completed every inner fold')
    best=max(t['mean_c'] for t in valid)
    return min((t for t in valid if t['mean_c']>=best-tolerance),key=lambda t:(t['mean_ibs'],-t['mean_c'],t['config_id']))


def measure(method,config,x,t,e,query,tq,eq,seed,split_seed,protocol,budget,archive=None,test_rows=None):
    grid=evaluation_grid(t,e)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prediction=fit_predict(method,config,x,t,e,query,grid,[seed],split_seed,protocol['threads'],budget)
    prediction.survival,roundoff=checked_survival(prediction.survival)
    metrics=evaluate(t,e,tq,eq,prediction.risk,prediction.survival,grid)
    member=prediction.metadata['members'][0]
    if 'point_time_prediction' in member:
        point=np.asarray(member['point_time_prediction'])
        metrics['mae_uncensored']=float(np.abs(point[eq==1]-tq[eq==1]).mean())
        from lifelines.utils import concordance_index
        metrics['c_from_point']=float(concordance_index(tq,point,eq))
    if archive is not None:
        np.savez_compressed(archive,risk=prediction.risk,survival=prediction.survival,grid=grid,
            point_time=np.asarray(member['point_time_prediction']),test_time=tq,test_event=eq,train_time=t,train_event=e,test_rows=test_rows)
    return dict(metrics=metrics,metadata=prediction.metadata,numerical_roundoff=roundoff,
        warnings=[str(w.message) for w in caught],seconds=time.perf_counter()-started)


def run(root,protocol):
    root=Path(root)
    approval=json.loads((root/'APPROVAL.json').read_text(encoding='utf-8'))
    if (not approval.get('approved_by_user') or approval.get('protocol_id')!=protocol['protocol_id']
        or 'development' not in approval.get('approved_phases',[]) or not approval.get('user_instruction')
        or protocol!=json.loads((root/'protocol.json').read_text(encoding='utf-8'))):
        raise PermissionError('Runner must match the explicitly authorized frozen pilot')
    spaces=json.loads((root/'search_spaces.json').read_text(encoding='utf-8'))
    result=root/'results';result.mkdir(exist_ok=True)
    manifest=source_manifest(root,protocol,spaces)
    path=result/'run_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError('Frozen source/data/protocol/environment changed; cannot mix runs')
    if not path.exists():write_json(path,manifest)
    budget=FitBudget(result/'fit_attempts.jsonl',protocol['hard_fit_cap'],0)
    validate_cache(root,protocol)
    print('Cached42 verified; starting six-method SINGLE-METRIC primary-C comparison',flush=True)
    for dataset in protocol['datasets']:
        x,t,e=load_dataset(dataset);groups=duplicate_groups(x,t,e)
        write_json(result/(dataset+'__dataset.json'),dict(n=len(x),events=int(e.sum()),features=list(x.columns),
            exact_duplicate_rows=len(x)-len(np.unique(groups))))
        for seed in protocol['seeds']:
            for outer_fold,(train,test) in enumerate(grouped_folds(e,groups,3,seed)):
                xt,tt,et=x.iloc[train],t[train],e[train]
                inner=grouped_folds(et,groups[train],2,seed+314159)
                split_seed=seed+271828
                for method in protocol['methods']:
                    identity=f'{dataset}__{method}__{seed}__{outer_fold}'
                    path=result/(identity+'.json')
                    if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') in {'ok','failed'}:continue
                    row=dict(status='running',dataset=dataset,method=method,seed=seed,outer_fold=outer_fold,
                        outer_train_rows=train.tolist(),outer_test_rows=test.tolist(),inner_trials=[])
                    write_json(path,row)
                    for config_id,config in enumerate(spaces[method]):
                        trial=dict(config_id=config_id,config=config,status='ok',folds=[])
                        for inner_fold,(fit,validation) in enumerate(inner):
                            try:
                                budget.context=dict(phase='main_inner',dataset=dataset,seed=seed,outer_fold=outer_fold,method=method,config_id=config_id,inner_fold=inner_fold)
                                measured=measure(method,config,xt.iloc[fit],tt[fit],et[fit],xt.iloc[validation],
                                    tt[validation],et[validation],seed,split_seed,protocol,budget)
                                trial['folds'].append(dict(inner_fold=inner_fold,fit_rows=fit.tolist(),
                                    validation_rows=validation.tolist(),**measured))
                            except FitBudgetError:raise
                            except Exception:
                                trial.update(status='failed',failure=traceback.format_exc());break
                        if trial['status']=='ok':
                            trial.update(mean_c=statistics.mean(f['metrics']['c'] for f in trial['folds']),
                                mean_ibs=statistics.mean(f['metrics']['ibs'] for f in trial['folds']))
                        row['inner_trials'].append(trial);write_json(path,row)
                    try:
                        selected=choose(row['inner_trials'],protocol['cindex_tolerance'])
                        config=dict(selected['config'])
                        if method.startswith(('Metric','Frontier','Explore','Harrell','Uno','NeuralLognormal')):
                            checkpoint=result/(identity+'.pt');config['_checkpoint']=str(checkpoint)
                        budget.context=dict(phase='main_outer',dataset=dataset,seed=seed,outer_fold=outer_fold,method=method,config_id=selected['config_id'])
                        measured=measure(method,config,xt,tt,et,x.iloc[test],t[test],e[test],seed,split_seed,
                            protocol,budget,archive=result/(identity+'.npz'),test_rows=test)
                        row.update(status='ok',selected_config_id=selected['config_id'],selected_config=selected['config'],final=measured)
                    except FitBudgetError:raise
                    except Exception:row.update(status='failed',failure=traceback.format_exc())
                    write_json(path,row);print(identity,row['status'],flush=True)
