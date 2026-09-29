"""Exact certification probabilities and audit-budget sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom,binom
P=Path(__file__).resolve().parents[1]

def certificate_threshold(N,m,limit,alpha):
 k=np.arange(m+1);ok=k[hypergeom.cdf(k,N,limit+1,m)<=alpha]
 return m-int(ok.max()) if ok.size else m+1

def power(N,K,m,limit,alpha,t,nonresponse=0):
 bstar=certificate_threshold(N,m,limit,alpha)
 x=np.arange(max(0,m-(N-K)),min(m,K)+1)
 return float(np.sum(hypergeom.pmf(x,N,K,m)*binom.sf(bstar-1,m-x,(1-np.exp(-t))*(1-nonresponse))))

def main():
 cohorts=pd.read_csv(P/'results/cohorts.csv');rows=[]
 for budget in [64,128,256,512,1024]:
  for scope,alpha in [('cohort',.05),('all_202',.05/len(cohorts))]:
   for t in [4,8,16]:
    for rate in [.005,.01,.02]:
     values=[power(int(r.N),int(r.K),min(budget,int(r.N)),int(np.floor(rate*r.N)),alpha,t) for r in cohorts.itertuples()]
     rows.append(dict(audit_budget=budget,scope=scope,time=t,rate_target=rate,mean_certification=np.mean(values)))
 pd.DataFrame(rows).to_csv(P/'results/budget_power.csv',index=False)
 rows=[]
 for missing in [0,.005,.01,.02,.05,.1]:
  for t in [2,4,8,16,64,1000]:
   for K in [0,20,82]:
    rows.append(dict(nonresponse=missing,time=t,K=K,certification=power(4096,K,256,81,.05,t,missing)))
 pd.DataFrame(rows).to_csv(P/'results/nonresponse.csv',index=False)
 # A deterministic dilution experiment keeps the harmful count fixed while adding benign alerts.
 rows=[]
 for N in [512,1024,2048,4096,8192,16384]:
  for kind,limit in [('rate',int(.02*N)),('count',10)]:
   rows.append(dict(N=N,K=20,m=256,criterion=kind,limit=limit,actual_safe=20<=limit,certification=power(N,20,256,limit,.05,1000)))
 pd.DataFrame(rows).to_csv(P/'results/dilution.csv',index=False)
 print('Exact power and dilution experiments complete')
if __name__=='__main__':main()
