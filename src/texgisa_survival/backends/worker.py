# Copyright (c) 2026 DHAI Lab. Licensed under the MIT License.
"""Subprocess boundary: loads native adapters without importing any own model."""
from pathlib import Path
import json
import sys
import numpy as np

if __name__=='__main__':
    root=Path(sys.argv[1]);job=json.loads((root/'job.json').read_text())
    # Import by file directory, avoiding texgisa_survival.__init__ in legacy runtimes.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from native_models import fit_native
    with np.load(root/'input.npz',allow_pickle=False) as z:
        validation=tuple(np.asarray(v,dtype=int) for v in job['validation']) if job['validation'] else None
        result=fit_native(job['method'],job['config'],z['x'],z['t'],z['e'],z['q'],job['seed'],job['threads'],validation)
    curve=result['survival'];np.savez(root/'output.npz',
        risk=[] if result['risk'] is None else result['risk'],
        curve=[] if curve is None else np.asarray(curve),
        curve_times=[] if curve is None else np.asarray(curve.index),
        point=[] if result['point'] is None else result['point'])
    (root/'metadata.json').write_text(json.dumps(result['metadata'],default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
