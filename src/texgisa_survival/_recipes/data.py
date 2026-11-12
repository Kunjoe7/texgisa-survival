"""Raw data and split-local transformations. No loading occurs at import."""
import numpy as np


def load_dataset(name):
    import pandas as pd
    from pathlib import Path
    filenames=dict(metabric='metabric.csv',gbsg='gabs.csv',nwtco='nwtco.csv',
        flchain='flchain.csv',support='support.csv',telco_clt='Telco-CLT.csv',
        telco_clv='Telco-CLV.csv',employee='employee_attrition.csv')
    x=pd.read_csv(Path(__file__).resolve().parents[1]/'data'/filenames[name])
    time_name,event_name=x.columns[-2:]
    t,e=x.pop(time_name).to_numpy(float),x.pop(event_name).to_numpy(int)
    x=x.apply(pd.to_numeric,errors='raise')
    t, e = np.asarray(t), np.asarray(e)
    if not np.isfinite(t).all() or (t < 0).any() or not np.isin(e, [0, 1]).all():
        raise ValueError("Invalid outcome data")
    x=x.reset_index(drop=True)
    x.attrs['dataset']=name
    return x, t, e


def preprocess(x_train, *queries, drop_reference=False, dtype=np.float32,return_transformer=False):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    numeric = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in x_train.columns if column not in numeric]
    numeric_steps=[SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler()]
    if np.dtype(dtype)==np.dtype('float64'):
        # Promote before imputation/scaling, not after their float32 arithmetic.
        numeric_steps.insert(0,FunctionTransformer(lambda values:np.asarray(values,dtype=np.float64)))
    transformer = ColumnTransformer([
        ("numeric", make_pipeline(*numeric_steps), numeric),
        ("category", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                 OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               drop='first' if drop_reference else None,dtype=np.float64)), categorical)
    ])
    transformer.fit(x_train)
    arrays=[transformer.transform(frame).astype(dtype) for frame in (x_train, *queries)]
    return (arrays,transformer) if return_transformer else arrays


def classical_design(x_train, *queries):
    """Full-rank Cox design learned only from fitting rows, in float64."""
    from scipy.linalg import qr
    arrays=preprocess(x_train,*queries,drop_reference=True,dtype=np.float64)
    fit=np.asarray(arrays[0],dtype=float)
    centered=fit-fit.mean(axis=0)
    _, triangular, pivots=qr(centered,mode='economic',pivoting=True)
    diagonal=np.abs(np.diag(triangular))
    tolerance=max(float(diagonal.max(initial=0.))*1e-10,1e-12)
    rank=int((diagonal>tolerance).sum())
    if rank==0:
        raise ValueError('No informative Cox covariates remain')
    retained=np.sort(pivots[:rank])
    return ([np.asarray(a,dtype=float)[:,retained] for a in arrays],
            dict(reference_coding=True,features_before_rank_reduction=fit.shape[1],
                 features_after_rank_reduction=rank,retained_columns=retained.tolist(),
                 rank_tolerance=tolerance))


def duplicate_groups(x,t,e):
    import pandas as pd
    rows=x.copy()
    rows['__observed_time__']=np.asarray(t)
    rows['__event__']=np.asarray(e)
    return pd.util.hash_pandas_object(rows,index=False).to_numpy()


def grouped_folds(events,groups,n_splits,seed):
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed).split(
        np.zeros(len(events)),events,groups))


def neural_partitions(events, seed, calibration, groups=None):
    from sklearn.model_selection import train_test_split
    if groups is not None:
        unique,first=np.unique(groups,return_index=True)
        all_groups=np.arange(len(unique))
        labels=np.asarray(events)[first]
        remain,cal_groups=train_test_split(all_groups,test_size=.15,random_state=seed,stratify=labels)
        core_groups,stop_groups=train_test_split(remain,test_size=len(cal_groups),random_state=seed+1,
                                               stratify=labels[remain])
        core=np.flatnonzero(np.isin(groups,unique[core_groups]))
        stop=np.flatnonzero(np.isin(groups,unique[stop_groups]))
        cal=np.flatnonzero(np.isin(groups,unique[cal_groups]))
        assert not (set(groups[core])&set(groups[stop]) or set(groups[core])&set(groups[cal]) or set(groups[stop])&set(groups[cal]))
        return core,stop,cal
    all_rows = np.arange(len(events))
    if calibration:
        remaining, cal = train_test_split(all_rows, test_size=.15, random_state=seed,
                                         stratify=events)
        # Match the calibration count exactly; the floating ratio .15/.85 can
        # round just above an integer and silently allocate one extra stop row.
        core, stop = train_test_split(remaining, test_size=len(cal), random_state=seed+1,
                                      stratify=np.asarray(events)[remaining])
    else:
        core, stop = train_test_split(all_rows, test_size=.2, random_state=seed,
                                      stratify=events)
        cal = np.array([], dtype=int)
    assert not (set(core) & set(stop) or set(core) & set(cal) or set(stop) & set(cal))
    return core, stop, cal


def training_partitions(x,t,e,seed,config):
    core,stop,cal=neural_partitions(e,seed,True,groups=duplicate_groups(x,t,e))
    if config.get('_anchor_full'):
        return np.arange(len(x)),np.array([],dtype=int),np.array([],dtype=int)
    if '_stage_epochs' in config:
        core=np.sort(np.r_[core,stop]);stop=np.array([],dtype=int)
    return core,stop,cal
