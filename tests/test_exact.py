"""Exhaustive finite-population checks independent of Monte Carlo simulations."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit import upper_count,ppr_upper,min_sample
from scipy.stats import hypergeom
import numpy as np

def run():
 cases=0;maxrisk=0.;patternchecks=0
 for N in range(1,25):
  for m in range(N+1):
   u=np.array([upper_count(N,m,k,.05) for k in range(m+1)])
   assert np.all(np.diff(u)>=0)
   for K in range(N+1):
    risk=hypergeom.pmf(np.arange(m+1),N,K,m)[u<K].sum()
    assert risk<=.05+1e-12,(N,m,K,risk)
    maxrisk=max(maxrisk,float(risk));cases+=1
   for X in range(m+1):
    # Every feasible completion summary a,b with a<=X,b<=m-X.
    for a in range(X+1):
     for b in range(m-X+1):
      assert u[m-b]>=u[X];patternchecks+=1
   if m==N: assert np.array_equal(u,np.arange(N+1))
 for N in [10,100,4096]:
  for K in [1,min(10,N),N]:
   m=min_sample(N,K)
   assert hypergeom.cdf(0,N,K,m)<=.05
   if m: assert hypergeom.cdf(0,N,K,m-1)>.05
 # Enumerate all ordered finite-population paths for small N; PPR any-time failure.
 pprcases=0;pprmax=0.
 for N in range(1,11):
  for K in range(N+1):
   fail=0;total=0
   for ones in itertools.combinations(range(N),K):
    y=np.zeros(N,dtype=int);y[list(ones)]=1;k=0;bad=False
    for n,v in enumerate(y,1):
     k+=v;bad|=ppr_upper(N,n,k)<K
    fail+=bad;total+=1
   risk=fail/total;assert risk<=.05+1e-12
   pprmax=max(pprmax,risk);pprcases+=1
 # Exact upper-end acceptance, including samples close to a census.
 from scipy.special import betaln,gammaln
 for N in [10,1000]:
  for n in [N,N-1,N//2]:
   for k in {0,1,n//2,n}:
    for a,b in [(.5,.5),(.5,24.5)]:
     upper=ppr_upper(N,n,k,.05,a,b)
     cutoff=np.log(.05)+gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b)
     assert hypergeom.logpmf(k,N,upper,n)>cutoff
     if upper<N-n+k: assert hypergeom.logpmf(k,N,upper+1,n)<=cutoff
 # Independent vector/scalar agreement on lookup calculations and both PPR priors.
 import run_experiments as rexp
 lut=rexp.make_lookup(4096,256,.05);rng=np.random.default_rng(991)
 for _ in range(250):
  n=int(rng.integers(0,257));k=int(rng.integers(0,n+1))
  assert lut['fixed'][n,k]==upper_count(4096,n,k,.05)
  assert lut['seq'][n,k]==ppr_upper(4096,n,k,.05)
  assert lut['informed'][n,k]==ppr_upper(4096,n,k,.05,.5,24.5)
 result=dict(lookup_cross_checks=750,finite_population_cases=cases,completion_summaries=patternchecks,max_exact_failure=maxrisk,ppr_path_cases=pprcases,max_ppr_failure=pprmax,status='passed')
 (Path(__file__).resolve().parents[1]/'results/exact_checks.json').write_text(json.dumps(result,indent=2))
 print(result)
if __name__=='__main__':run()
