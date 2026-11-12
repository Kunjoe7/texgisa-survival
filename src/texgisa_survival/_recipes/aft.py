"""Right-censored parametric location-scale regression, train-only transforms.

Gaussian is a right-censored Gaussian/Tobit-style comparator, not a claim of
left-censored Tobit reproduction. Raw Gaussian/logistic support is R; survival
is never silently conditioned on positive time. Positive families use T+delta.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, expit
from .adapters import Prediction, _standardize_risk
from .data import classical_design

METHODS={'Tobit','Weibull','Logistic','Loglogistic','Lognormal'}

def scores(z, family):
    if family=='normal':
        log_pdf=-.5*z*z-.5*np.log(2*np.pi)
        log_surv=log_ndtr(-z)
        mills=np.exp(np.minimum(log_pdf-log_surv,700))
        return log_pdf,log_surv,z,mills
    if family=='logistic':
        return -np.logaddexp(0,z)-np.logaddexp(0,-z),-np.logaddexp(0,z),np.tanh(z/2),expit(z)
    # Weibull's log-time error is minimum extreme value, not Gumbel(max).
    ez=np.exp(np.minimum(z,500))
    return z-ez,-ez,ez-1,ez

def objective(theta,design,y,event,family,penalty):
    scale=np.exp(theta[-1]); z=(y-design@theta[:-1])/scale
    lp,ls,event_score,censor_score=scores(z,family)
    score=event*event_score+(1-event)*censor_score
    value=-np.mean(event*(lp-theta[-1])+(1-event)*ls)+penalty*np.sum(theta[1:-1]**2)/2
    gradient=np.r_[-design.T@score/len(y)/scale,-np.mean(score*z-event)]
    gradient[1:-1]+=penalty*theta[1:-1]
    return value,gradient

def fit_aft(method,c,x,t,e,query,grid,seed):
    (xf,xq),design_meta=classical_design(x,query)
    positive=method in {'Weibull','Loglogistic','Lognormal'}
    # Translation chosen from fitting data alone; exact inverse includes -delta.
    delta=max(float(np.median(t[t>0]))*1e-6,1e-8) if positive else 0.
    raw=np.log(t+delta) if positive else t.copy()
    center=float(np.median(raw)); unit=max(float(np.subtract(*np.percentile(raw,[75,25]))),float(np.std(raw))*.1,1e-6)
    y=(raw-center)/unit
    design=np.c_[np.ones(len(xf)),xf]; query_design=np.c_[np.ones(len(xq)),xq]
    family={'Tobit':'normal','Lognormal':'normal','Logistic':'logistic','Loglogistic':'logistic','Weibull':'extreme'}[method]
    theta=np.r_[np.zeros(design.shape[1]),0.]
    result=minimize(objective,theta,args=(design,y,e,family,c['pen']),jac=True,method='L-BFGS-B',
        bounds=[(None,None)]*design.shape[1]+[(-4.,4.)],options={'maxiter':3000,'ftol':1e-11,'gtol':1e-6,'maxls':50})
    gradient_norm=float(np.max(np.abs(result.jac)))
    if not np.isfinite(result.fun) or (not result.success and gradient_norm>1e-4):
        raise FloatingPointError('AFT optimizer did not converge: '+str(result.message))
    mu=center+unit*(query_design@result.x[:-1]); ref=center+unit*(design@result.x[:-1]); scale=unit*np.exp(result.x[-1])
    transformed_grid=np.log(grid+delta) if positive else grid
    survival=np.exp(scores((transformed_grid[None,:]-mu[:,None])/scale,family)[1])
    error_median=np.log(np.log(2.)) if family=='extreme' else 0.
    point=np.exp(mu+scale*error_median)-delta if positive else mu
    if not np.isfinite(point).all():raise FloatingPointError('Nonfinite native AFT median')
    return Prediction(_standardize_risk(-ref,-mu),survival,dict(seed=seed,n_fit=len(x),n_stop=0,n_calibration=0,
        point_time_prediction=point.tolist(),point_readout='unconditional native median; positive time inverse subtracts fitting delta',
        family=family,positive_time_translation=delta,optimizer_success=bool(result.success),optimizer_message=str(result.message),
        optimizer_gradient_max=gradient_norm,design=design_meta,distribution_support=('T >= -delta' if positive else 'real line; no positive-time conditioning'),
        implementation='independent right-censored likelihood; Gaussian labelled Tobit-style, not exact original Tobit reproduction'))
