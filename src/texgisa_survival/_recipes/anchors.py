"""Fixed ten-fold diagnostic anchors, separate from nested-CV score tables."""
import json,traceback
import numpy as np
from .data import load_dataset,duplicate_groups,grouped_folds
from .metrics import evaluation_grid
from .adapters import fit_one
from .budget import FitBudgetError

def run_anchors(root,protocol,budget):
    from .pilot import write_json
    from lifelines.utils import concordance_index
    out=root/'results/anchors';out.mkdir(exist_ok=True)
    spaces=json.loads((root/'search_spaces.json').read_text())
    for dataset in protocol['datasets']:
        x,t,e=load_dataset(dataset);groups=duplicate_groups(x,t,e)
        for seed in protocol['seeds']:
            for fold,(train,test) in enumerate(grouped_folds(e,groups,10,seed)):
                for method in protocol['anchor_protocol']['methods']:
                    identity=f'{dataset}__{method}__{seed}__{fold}';path=out/(identity+'.json')
                    if path.exists():
                        prior=json.loads(path.read_text())
                        if prior.get('status')=='ok':continue
                        if prior.get('status')=='failed':raise RuntimeError('Unresolved preserved anchor failure: '+identity)
                    row=dict(dataset=dataset,method=method,seed=seed,outer_fold=fold,status='running',outer_train_rows=train.tolist(),outer_test_rows=test.tolist())
                    write_json(path,row)
                    c=dict(spaces[method][0])
                    if method=='UnoNMSE':c.update(hidden=[32,32],lr=.05,dropout=.1,batch=1024,alpha=.5,sigma=.01,epochs=200,_stage_epochs=200,_anchor_full=True)
                    try:
                        budget.context=dict(phase='anchor',dataset=dataset,method=method,seed=seed,outer_fold=fold)
                        budget.reserve(method+':anchor',[seed])
                        pred=fit_one(method,c,x.iloc[train],t[train],e[train],x.iloc[test],evaluation_grid(t[train],e[train]),seed,seed+271828,protocol['threads'])
                        point=np.asarray(pred.metadata['point_time_prediction'])
                        if not np.isfinite(point).all() or not np.isfinite(pred.risk).all():raise FloatingPointError('Nonfinite anchor point or risk')
                        metrics=dict(c=float(concordance_index(t[test],-pred.risk,e[test])),mae_uncensored=float(np.abs(point[e[test]==1]-t[test][e[test]==1]).mean()))
                        np.savez_compressed(out/(identity+'.npz'),risk=pred.risk,point_time=point,test_time=t[test],test_event=e[test],test_rows=test)
                        row.update(status='ok',metrics=metrics,metadata=pred.metadata,configuration=c)
                    except FitBudgetError:raise
                    except Exception:
                        row.update(status='failed',failure=traceback.format_exc());write_json(path,row)
                        # An anchor implementation/convergence failure is a gate,
                        # never silently replaced or followed by a changed main model.
                        raise RuntimeError('Anchor failure; inspect preserved record: '+identity)
                    write_json(path,row);print('anchor',identity,'ok',flush=True)
    records=[json.loads(path.read_text()) for path in out.glob('*.json')]
    if len(records)!=protocol['expected_anchor_tasks'] or any(r['status']!='ok' for r in records):
        raise RuntimeError('Anchor gate not complete; main is forbidden')
