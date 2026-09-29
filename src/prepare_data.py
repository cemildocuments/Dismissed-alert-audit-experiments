"""Fetch pinned AIT-ADS CSVs and derive held-out-scenario dismissal cohorts."""
from pathlib import Path
import hashlib,json,zipfile,urllib.request
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
def prepare():
 manifest=json.loads((P/'data/manifest.json').read_text()); item=manifest['files'][0]
 arc=P/'data'/item['name']
 if not arc.exists(): urllib.request.urlretrieve(item['url'],arc)
 if hashlib.sha256(arc.read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError('data checksum mismatch')
 dest=P/'data/alerts_csv'
 if not dest.exists():
  dest.mkdir(parents=True)
  with zipfile.ZipFile(arc) as z:
   for member in z.infolist():
    if member.filename.endswith('.txt'):
     (dest/Path(member.filename).name).write_bytes(z.read(member))
 config=json.loads((P/'config.json').read_text())['replay']
 datasets={f.stem.replace('_alerts',''):pd.read_csv(f,usecols=['time','short','event_label']).sort_values('time',kind='stable') for f in sorted(dest.glob('*.txt'))}
 rows=[]; arrays={}; cohorts=[]
 for name,d in datasets.items():
  tr=pd.concat([v for key,v in datasets.items() if key!=name]);tr['y']=tr.event_label.ne('-').astype(int)
  grouped=tr.groupby('short').y.agg(['sum','count'])
  score=(grouped['sum']+1)/(grouped['count']+2)
  p=d.short.map(score).fillna(.5).to_numpy(); y=d.event_label.ne('-').to_numpy()
  selected=p<=config['threshold']; yd=y[selected]; pdismiss=p[selected]
  arrays[name+'_y']=yd; arrays[name+'_score']=pdismiss
  rows.append(dict(scenario=name,alerts=len(y),attacks=int(y.sum()),dismissed=int(selected.sum()),missed=int(yd.sum()),rate=float(yd.mean())))
  N=config['cohort_size']
  for j,start in enumerate(range(0,len(yd),N)):
   part=yd[start:start+N];n=len(part)
   cohorts.append(dict(scenario=name,cohort=j,N=n,K=int(part.sum()),m=min(n,config['audit_size'])))
 np.savez_compressed(P/'data/cohorts.npz',**arrays)
 pd.DataFrame(rows).to_csv(P/'results/dataset_summary.csv',index=False)
 pd.DataFrame(cohorts).to_csv(P/'results/cohorts.csv',index=False)
 return rows
if __name__=='__main__':
 print(json.dumps(prepare(),indent=2))
