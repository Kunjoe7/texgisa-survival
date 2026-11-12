# Copyright (c) 2026 DHAI Lab. Licensed under the MIT License.
"""Explicit isolated invocation of optional, unchanged third-party implementations."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

AUTHOR_ENTRIES={'DRO':'run_dro_cox_split.py','Kernet':'demo_tuna_kernet.py','SODEN':'main.py'}


def available_methods():
    """Return dispatch capabilities, not a claim that optional dependencies are installed."""
    from .native_models import METHODS
    return {**{m:'native-package' for m in METHODS},
            **{m:'isolated-author-cli' for m in AUTHOR_ENTRIES}}


def fit_predict(method,X,time,event,query,config=None,seed=271,threads=1,
                python_executable=None,author_source=None,timeout=None):
    """Fit an unchanged native model once in a fresh process and return native outputs.

    No expert priors, Fourier expansion, private losses, output calibration or
    model repair is applied. Native failures raise with their original log.
    ``python_executable`` selects an already-installed optional/legacy runtime.
    Four ICDM2021 point methods require their original source and NumPy1.x.
    """
    from .native_models import METHODS,PYCOX_METHODS
    if method not in METHODS:
        raise ValueError('Use run_author_cli for DRO, Kernet or SODEN')
    x=np.asarray(X,dtype=float);q=np.asarray(query,dtype=float)
    t=np.asarray(time,dtype=float);e=np.asarray(event)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or t.shape!=(len(x),) or e.shape!=t.shape:
        raise ValueError('Misaligned native arrays')
    defaults=json.loads(Path(__file__).with_name('native_configs.json').read_text())
    cfg=dict(defaults.get(method,[{}])[0]);cfg.update(config or {})
    # The baseline receives at most its original adapter's train-only scaling.
    validation=None
    if method in PYCOX_METHODS:
        from sklearn.model_selection import train_test_split
        fitting,valid=train_test_split(np.arange(len(x)),test_size=.2,random_state=seed,stratify=e)
        validation=[fitting.tolist(),valid.tolist()]
    with tempfile.TemporaryDirectory(prefix='texgisa_native_') as folder:
        root=Path(folder);np.savez(root/'input.npz',x=x,t=t,e=e,q=q)
        job={'method':method,'config':cfg,'seed':seed,'threads':threads,'validation':validation}
        (root/'job.json').write_text(json.dumps(job))
        env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        if author_source:env['TEXGISA_ICDM2021_SOURCE']=str(Path(author_source).resolve())
        proc=subprocess.run([str(python_executable or sys.executable),'-B',str(Path(__file__).with_name('worker.py')),str(root)],capture_output=True,text=True,env=env,timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f'{method} native execution failed (code {proc.returncode})\n'+proc.stdout+'\n'+proc.stderr)
        with np.load(root/'output.npz',allow_pickle=False) as z:
            result={name:z[name].copy() if z[name].size else None for name in ['risk','curve','curve_times','point']}
        result['metadata']=json.loads((root/'metadata.json').read_text())
        return result


def run_author_cli(method,source_dir,args=(),python_executable=None,timeout=None,log_path=None):
    """Run the method's original CLI with explicitly prepared author-format inputs.

    Arguments follow the upstream README. Separate runtime and source checkout
    are explicit because legacy dependencies and licenses differ. No source
    rewriting, monkeypatching or automatic fallback is performed.
    """
    if method not in AUTHOR_ENTRIES:raise ValueError('Unknown author CLI')
    folder=Path(source_dir).resolve();entry=folder/AUTHOR_ENTRIES[method]
    if not entry.is_file():raise FileNotFoundError(entry)
    before=hashlib.sha256(entry.read_bytes()).hexdigest()
    env=dict(os.environ,PYTHONUTF8='1');env.pop('PYTHONPATH',None)
    command=[str(python_executable or sys.executable),str(entry),*map(str,args)]
    proc=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,timeout=timeout)
    if log_path:Path(log_path).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if hashlib.sha256(entry.read_bytes()).hexdigest()!=before:raise RuntimeError('Author entry changed during execution')
    if proc.returncode:raise RuntimeError(f'{method} original CLI exited {proc.returncode}\n'+proc.stdout+'\n'+proc.stderr)
    return {'method':method,'command':command,'source_sha256':before,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}

