"""Validate multi-root immutable paired evidence without relabeling old outputs."""
from pathlib import Path
import hashlib, importlib.metadata, json

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):h.update(part)
    return h.hexdigest()

def evidence_roots(p):
    mapping={}
    for source in p['cache_sources']:
        for method in source['methods']:
            if method in mapping:raise RuntimeError('Duplicate cached ownership')
            mapping[method]=Path(source['root'])
    if set(mapping)!=set(p['cached_methods']):raise RuntimeError('Cached ownership differs')
    return mapping

def result_folder(root,p,method):
    return (evidence_roots(p)[method] if method in p['cached_methods'] else Path(root))/'results'

def cached_records(root,p):
    return sorted(path for method,old in evidence_roots(p).items() for path in (old/'results').glob('*__'+method+'__*__*.json'))

def validate_cache(root,p):
    root=Path(root);evidence_roots(p)
    manifest=json.loads((root/'cache_manifest.json').read_text(encoding='utf-8'))
    if [{k:s[k] for k in ('root','methods')} for s in manifest['sources']]!=p['cache_sources']:raise RuntimeError('Cached descriptors changed')
    spaces=json.loads((root/'search_spaces.json').read_text(encoding='utf-8'))
    from .data import load_dataset,duplicate_groups,grouped_folds
    expected={(d,m,s,f) for d in p['datasets'] for m in p['cached_methods'] for s in p['seeds'] for f in range(3)}
    identities=[]
    for source in manifest['sources']:
        old=Path(source['root'])
        for name,sha in source['sha256'].items():
            if digest(old/name)!=sha:raise RuntimeError('Cached evidence changed: '+str(old/name))
        prior=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
        if source['methods']!=prior['methods']:raise RuntimeError('Cached membership differs')
        for key in ['datasets','seeds','outer_folds','inner_folds','n_configs','cindex_tolerance']:
            if p[key]!=prior[key]:raise RuntimeError('Unpaired protocol: '+key)
        previous_spaces=json.loads((old/'search_spaces.json').read_text(encoding='utf-8'))
        if any(spaces[m]!=previous_spaces[m] for m in source['methods']):raise RuntimeError('Cached grid differs')
        saved=json.loads((old/'results/run_manifest.json').read_text(encoding='utf-8'))
        if saved['protocol']!=prior or saved['search_spaces']!=previous_spaces:raise RuntimeError('Old manifest differs')
        if json.loads((old/'RUN_STATUS.json').read_text(encoding='utf-8'))['state']!='completed':raise RuntimeError('Old campaign not audited')
        if {f.name for f in (root/'data').glob('*.csv')}!={f.name for f in (old/'data').glob('*.csv')}:raise RuntimeError('CSV membership differs')
        for file in (root/'data').glob('*.csv'):
            if digest(file)!=digest(old/'data'/file.name):raise RuntimeError('Input data differs')
        for name in ['priors.json','engine/data.py','engine/metrics.py']:
            if digest(root/name)!=digest(old/name):raise RuntimeError('Scientific definition changed: '+name)
        for package,version in saved['packages'].items():
            if importlib.metadata.version(package)!=version:raise RuntimeError('Environment differs: '+package)
        frozen=json.loads((old/'reviews/frozen_manifest.json').read_text(encoding='utf-8'))
        for table in [frozen,saved['sha256']]:
            for name,sha in table.items():
                if digest(old/name)!=sha:raise RuntimeError('Old source differs: '+name)
        for method in source['methods']:
            for path in (old/'results').glob('*__'+method+'__*__*.json'):
                parts=path.stem.split('__');identities.append((parts[0],parts[1],int(parts[2]),int(parts[3])))
    if len(identities)!=len(set(identities)) or set(identities)!=expected:raise RuntimeError('Cached identities incomplete or duplicated')
    # All per-method row identities and predictions are replayed in final summary.
    # Independently reconstruct every outer/inner split using each source here.
    for d in p['datasets']:
        x,t,e=load_dataset(d);groups=duplicate_groups(x,t,e)
        for seed in p['seeds']:
            for fold,(train,test) in enumerate(grouped_folds(e,groups,3,seed)):
                for source in p['cache_sources']:
                    m=source['methods'][0]
                    row=json.loads((Path(source['root'])/'results'/f'{d}__{m}__{seed}__{fold}.json').read_text(encoding='utf-8'))
                    if row['outer_train_rows']!=train.tolist() or row['outer_test_rows']!=test.tolist():raise RuntimeError('Cached outer split differs')
                    inner=grouped_folds(e[train],groups[train],2,seed+314159)
                    for trial in row['inner_trials']:
                        for saved,(fit,val) in zip(trial['folds'],inner):
                            if saved['fit_rows']!=fit.tolist() or saved['validation_rows']!=val.tolist():raise RuntimeError('Cached inner split differs')
    return manifest
