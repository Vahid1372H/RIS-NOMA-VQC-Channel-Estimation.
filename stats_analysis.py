"""Performance metrics and seed-level paired statistical analysis."""
import numpy as np
from scipy import stats


def regression_metrics(truth,prediction):
    y=np.asarray(truth); p=np.asarray(prediction)
    err=np.abs(y-p)**2; denom=np.sum(np.abs(y-y.mean(axis=0,keepdims=True))**2)
    nmse=float(np.sum(err)/(np.sum(np.abs(y)**2)+1e-15))
    r2=float(1.0-np.sum(err)/(denom+1e-15))
    return {'nmse':nmse,'nmse_db':float(10*np.log10(nmse+1e-15)),'r2':r2,
            'mse':float(np.mean(err))}


def paired_seed_db_difference(a_db,b_db):
    """Paired per-seed A-minus-B difference in dB (seed is independent unit)."""
    a=np.asarray(a_db,dtype=float); b=np.asarray(b_db,dtype=float)
    if a.shape!=b.shape: raise ValueError('paired result arrays must have identical shapes')
    return a-b


def paired_tests(differences,alpha=0.05):
    """Paired Wilcoxon and paired t test; reports p-values and confidence interval."""
    d=np.asarray(differences,dtype=float); d=d[np.isfinite(d)]
    if len(d)<2: raise ValueError('At least two paired seeds are required.')
    wil=stats.wilcoxon(d,zero_method='zsplit',alternative='two-sided',method='auto')
    tt=stats.ttest_1samp(d,0.0,nan_policy='omit')
    sd=d.std(ddof=1); se=sd/np.sqrt(len(d)); crit=stats.t.ppf(1-alpha/2,len(d)-1)
    return {'n':int(len(d)),'mean_db':float(d.mean()),'median_db':float(np.median(d)),
            'std_db':float(sd),'wilcoxon_statistic':float(wil.statistic),'wilcoxon_p':float(wil.pvalue),
            'paired_t_statistic':float(tt.statistic),'paired_t_p':float(tt.pvalue),
            'mean_ci':(float(d.mean()-crit*se),float(d.mean()+crit*se))}


def tost_equivalence(differences,lower=-0.5,upper=0.5,alpha=0.05):
    """Two one-sided paired t-tests for equivalence within dB bounds."""
    if lower>=upper: raise ValueError('lower bound must be smaller than upper bound')
    d=np.asarray(differences,dtype=float); d=d[np.isfinite(d)]
    if len(d)<2: raise ValueError('At least two paired seeds are required.')
    lo=stats.ttest_1samp(d,lower,alternative='greater')
    hi=stats.ttest_1samp(d,upper,alternative='less')
    return {'lower_bound_db':lower,'upper_bound_db':upper,'lower_test_p':float(lo.pvalue),
            'upper_test_p':float(hi.pvalue),'equivalent':bool(lo.pvalue<alpha and hi.pvalue<alpha),
            'alpha':alpha}


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, retaining input order."""
    p=np.asarray(p_values,dtype=float); n=len(p); order=np.argsort(p); out=np.empty(n); running=0.0
    for rank,i in enumerate(order):
        running=max(running,(n-rank)*p[i]); out[i]=min(running,1.0)
    return out
