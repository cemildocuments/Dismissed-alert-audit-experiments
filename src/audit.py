"""Exact finite-cohort bounds. MIT License."""
from functools import lru_cache
import numpy as np
from scipy.stats import hypergeom
from scipy.special import betaln, gammaln

@lru_cache(maxsize=500000)
def upper_count(N, m, k, alpha=0.05):
    """Largest K with P_HG(N,K,m)(X<=k)>alpha (nonrandomized inversion)."""
    N,m,k=int(N),int(m),int(k)
    if not 0<=k<=m<=N or not 0<alpha<1: raise ValueError('invalid bound arguments')
    if m==0 or k==m: return N
    lo,hi=k,N-(m-k)
    while lo<hi:
        mid=(lo+hi+1)//2
        if hypergeom.cdf(k,N,mid,m)>alpha: lo=mid
        else: hi=mid-1
    return lo

@lru_cache(maxsize=500000)
def ppr_upper(N,n,k,alpha=0.05,prior_a=.5,prior_b=.5):
    """Jeffreys beta-binomial prior/posterior-ratio confidence-sequence upper end."""
    N,n,k=int(N),int(n),int(k)
    if n==0: return N
    logpredict=(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
                +betaln(k+prior_a,n-k+prior_b)-betaln(prior_a,prior_b))
    cutoff=np.log(alpha)+logpredict
    lo=max(k,min(N-n+k,int(np.floor((N+1)*k/n))))
    hi=N-n+k
    while lo<hi:
        mid=(lo+hi+1)//2
        if hypergeom.logpmf(k,N,mid,n)>cutoff: lo=mid
        else: hi=mid-1
    return lo

def bounds(N,m,alpha=.05):
    return np.array([upper_count(N,m,k,alpha) for k in range(m+1)],dtype=np.int64)

def min_sample(N,Kcrit,alpha=.05):
    """Fewest all-benign uniform audits capable of excluding K >= Kcrit."""
    lo,hi=0,N
    while lo<hi:
        mid=(lo+hi)//2
        if hypergeom.cdf(0,N,Kcrit,mid)<=alpha: hi=mid
        else: lo=mid+1
    return lo

def sampled_labels(rng,N,K,m,reps):
    x=rng.hypergeometric(K,N-K,m,size=reps)
    # A uniformly permuted binary sample conditional on its hypergeometric total.
    ranks=np.argsort(np.argsort(rng.random((reps,m)),axis=1),axis=1)
    return ranks<x[:,None]
