"""Deterministic CPU-parallel auditing, public-data replay, and service experiments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,platform,concurrent.futures
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.special import betaln,gammaln
from audit import bounds,sampled_labels,min_sample
P=Path(__file__).resolve().parents[1]
C=json.loads((P/'config.json').read_text())
METHODS=['Completed fixed','Completion-order CS','Random-prefix CS','Risk-informed prefix CS','Pending envelope']

def make_lookup(N,m,alpha):
 path=P/'data'/f'lookup_{N}_{m}_{alpha}.npz'
 if path.exists():
  cached=dict(np.load(path))
  if 'informed' in cached and np.array_equal(cached.get('priors',[]),C['ppr_priors']['diffuse']+C['ppr_priors']['risk_informed']):return cached
 ns,ks=np.tril_indices(m+1); valid=ns>0;n=ns[valid];k=ks[valid]
 fixed=np.full((m+1,m+1),N,dtype=np.int64);seq=fixed.copy()
 lo=k.copy();hi=N-n+k
 while np.any(lo<hi):
  mid=(lo+hi+1)//2;ok=hypergeom.cdf(k,N,mid,n)>alpha
  lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid-1)
 fixed[n,k]=lo
 u,v=C['ppr_priors']['diffuse']
 pred=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+u,n-k+v)-betaln(u,v)
 cut=np.log(alpha)+pred
 lo=np.maximum(k,np.minimum(N-n+k,np.floor((N+1)*k/n).astype(int)));hi=N-n+k
 while np.any(lo<hi):
  mid=(lo+hi+1)//2;ok=hypergeom.logpmf(k,N,mid,n)>cut
  lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid-1)
 seq[n,k]=lo
 informed=np.full((m+1,m+1),N,dtype=np.int64)
 u,v=C['ppr_priors']['risk_informed']
 pred2=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+u,n-k+v)-betaln(u,v)
 cut=np.log(alpha)+pred2
 lo=np.maximum(k,np.minimum(N-n+k,np.floor((N+1)*k/n).astype(int)));hi=N-n+k
 while np.any(lo<hi):
  mid=(lo+hi+1)//2;ok=hypergeom.logpmf(k,N,mid,n)>cut
  lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid-1)
 informed[n,k]=lo
 np.savez_compressed(path,fixed=fixed,seq=seq,informed=informed,priors=C['ppr_priors']['diffuse']+C['ppr_priors']['risk_informed'])
 return dict(fixed=fixed,seq=seq,informed=informed)

def evaluate(labels,delays,N,K,times,lookup,target=None):
 reps,m=labels.shape;fixed=lookup['fixed'];seq=lookup['seq'];informed=lookup['informed']
 kpref=np.cumsum(labels,axis=1)
 prefbounds=np.column_stack([np.full(reps,N),np.minimum.accumulate(seq[np.arange(1,m+1)[None,:],kpref],axis=1)])
 infobounds=np.column_stack([np.full(reps,N),np.minimum.accumulate(informed[np.arange(1,m+1)[None,:],kpref],axis=1)])
 ordering=np.argsort(delays,axis=1,kind='stable')
 sortedlabels=np.take_along_axis(labels,ordering,axis=1)
 kcomplete=np.cumsum(sortedlabels,axis=1)
 compbounds=np.column_stack([np.full(reps,N),np.minimum.accumulate(informed[np.arange(1,m+1)[None,:],kcomplete],axis=1)])
 failure=np.zeros((reps,len(METHODS)),bool); rows=[];last=None
 firstcert=np.full((reps,len(METHODS)),np.inf)
 for t in times:
  done=delays<=t;n=done.sum(axis=1);a=(done&labels).sum(axis=1);b=n-a
  prefix=np.where(done.all(axis=1),m,np.argmin(done,axis=1))
  vals=np.column_stack([fixed[n,a],compbounds[np.arange(reps),n],prefbounds[np.arange(reps),prefix],infobounds[np.arange(reps),prefix],fixed[m,m-b]])
  failure|=vals<K
  if target is not None:
   new=(vals<=target)&np.isinf(firstcert);firstcert[new]=t
  for j,name in enumerate(METHODS):
   rows.append(dict(time=t,method=name,failure=int(failure[:,j].sum()),repetitions=reps,
                    mean_upper=float(vals[:,j].mean()/N),mean_resolved=float(n.mean()),
                    certified=int((vals[:,j]<=target).sum()) if target is not None else 0))
  last=vals
 return rows,firstcert

def synthetic_task(task):
 index,K,ratio=task;c=C['synthetic'];rng=np.random.default_rng(C['seed']+index)
 N,m,R=c['N'],c['m'],c['repetitions'];labels=sampled_labels(rng,N,K,m,R)
 delays=rng.exponential(1,size=labels.shape)*np.where(labels,ratio,1)
 rows,_=evaluate(labels,delays,N,K,c['times'],dict(np.load(P/'data'/f'lookup_{N}_{m}_{C["alpha"]}.npz')))
 for row in rows:row.update(K=K,N=N,m=m,delay_ratio=ratio)
 return rows

def replay_task(task):
 index,row=task;c=C['replay'];rng=np.random.default_rng(C['seed']+10000+index)
 N,K,m=int(row['N']),int(row['K']),int(row['m']);R=c['repetitions']
 # Conditional design replay: draw uniform audits of this cohort's fixed event-label vector.
 labels=sampled_labels(rng,N,K,m,R)
 delays=rng.exponential(1,size=labels.shape)*np.where(labels,c['delay_ratio'],1)
 lookup=dict(np.load(P/'data'/f'lookup_{N}_{m}_{C["alpha"]}.npz'))
 rows,_=evaluate(labels,delays,N,K,c['times'],lookup,int(np.floor(c['rate_target']*N)))
 for item in rows:item.update(scenario=row['scenario'],cohort=row['cohort'],N=N,K=K,m=m)
 return rows

def queue_task(task):
 index,load=task;c=C['queue'];rng=np.random.default_rng(C['seed']+20000+index)
 R=c['repetitions'];T=c['horizon'];B=c['batches'];m=c['batch'];period=c['batch_period'];total=B*m
 jobs=rng.poisson(load*c['total_service'],size=(R,T))
 # All-benign audit workload isolates certificate availability from error detection.
 delay=rng.exponential(c['benign_delay'],size=(R,total))
 rows=[]
 for policy in ['Shared priority','Reserved audit']:
  backlog=np.zeros(R,int);served=np.zeros(R,int);starts=np.full((R,total),T+1,dtype=np.int32)
  for t in range(T):
   backlog+=jobs[:,t]
   arrived=min(B,t//period+1)*m
   if policy=='Reserved audit':
    a=np.minimum(c['reserved_service'],arrived-served)
    h=np.minimum(backlog,c['total_service']-a)
    # Unused high-priority capacity may serve additional audits (work-conserving).
    a+=np.minimum(c['total_service']-a-h,arrived-served-a)
   else:
    h=np.minimum(backlog,c['total_service']);a=np.minimum(c['total_service']-h,arrived-served)
   backlog-=h
   for offset in range(c['total_service']):
    ix=np.flatnonzero(a>offset)
    if ix.size:starts[ix,served[ix]+offset]=t
   served+=a
  verdict=starts+delay
  U=bounds(4096,m,C['alpha']);valid=np.flatnonzero(U<=int(c['rate_target']*4096));need=m-int(valid.max()) if valid.size else m+1
  for batch in range(B):
   ordered=np.sort(verdict[:,batch*m:(batch+1)*m],axis=1);release=batch*period
   cert=ordered[:,need-1]-release
   for r in range(R):rows.append(dict(load=load,policy=policy,replicate=r,batch=batch,certificate_latency=float(cert[r]) if cert[r]+release<=T else np.nan,certified_by_64=bool(cert[r]<=64),backlog_at_end=int(backlog[r])))
 return rows

def frontier():
 rows=[]
 for N in [512,1024,2048,4096,8192,16384,32768,65536]:
  for budget in [0,1,5,10,20,50,100]:
   rows.append(dict(N=N,attack_budget=budget,required_sample=min_sample(N,budget+1,C['alpha']),criterion='count'))
  for rate in [.001,.005,.01,.02,.05]:
   rows.append(dict(N=N,rate_budget=rate,required_sample=min_sample(N,int(np.floor(rate*N))+1,C['alpha']),criterion='rate'))
 pd.DataFrame(rows).to_csv(P/'results/frontier.csv',index=False)

def main():
 started=time.time();cohorts=pd.read_csv(P/'results/cohorts.csv')
 pairs={(C['synthetic']['N'],C['synthetic']['m'])}|{(int(r.N),int(r.m)) for r in cohorts.itertuples()}
 for N,m in sorted(pairs):make_lookup(N,m,C['alpha'])
 print('Lookup tables ready',flush=True)
 tasks=[(i,K,ratio) for i,(K,ratio) in enumerate((K,r) for K in C['synthetic']['K'] for r in C['synthetic']['delay_ratios'])]
 with concurrent.futures.ProcessPoolExecutor(max_workers=C['workers']) as ex:
  rows=[row for part in ex.map(synthetic_task,tasks) for row in part]
 pd.DataFrame(rows).to_csv(P/'results/synthetic.csv',index=False);print('Synthetic experiments complete',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=C['workers']) as ex:
  rows=[row for part in ex.map(replay_task,enumerate(cohorts.to_dict('records'))) for row in part]
 pd.DataFrame(rows).to_csv(P/'results/replay.csv',index=False);print('AIT replay complete',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=C['workers']) as ex:
  rows=[row for part in ex.map(queue_task,enumerate(C['queue']['loads'])) for row in part]
 pd.DataFrame(rows).to_csv(P/'results/queue.csv',index=False);frontier()
 metadata=dict(seed=C['seed'],workers=C['workers'],elapsed_seconds=time.time()-started,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),synthetic_repetitions=len(tasks)*C['synthetic']['repetitions'],replay_cohorts=len(cohorts),replay_repetitions=len(cohorts)*C['replay']['repetitions'])
 (P/'results/run_metadata.json').write_text(json.dumps(metadata,indent=2));print(metadata,flush=True)
if __name__=='__main__':main()
