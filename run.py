#!/usr/bin/env python3
"""Reproduce all audit experiments and compare with reference numerical outputs."""
from pathlib import Path
import os,sys,subprocess,argparse,json
P=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true',help='compare reproduced numerical tables with reference outputs');args=parser.parse_args()
 (P/'results').mkdir(exist_ok=True);(P/'data').mkdir(exist_ok=True)
 for path in ['src/prepare_data.py','tests/test_exact.py','src/run_experiments.py','src/power.py']:
  subprocess.run([sys.executable,str(P/path)],cwd=P,check=True)
 if args.verify:
  import pandas as pd,numpy as np
  checked=[]
  for ref in sorted((P/'reference_results').glob('*.csv')):
   expected=pd.read_csv(ref);actual=pd.read_csv(P/'results'/ref.name)
   pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-12,atol=1e-12)
   checked.append(ref.name)
  observed=json.loads((P/'results/exact_checks.json').read_text());expected=json.loads((P/'reference_results/exact_checks.json').read_text());assert observed==expected
  print('Verified numerical outputs:',', '.join(checked))
if __name__=='__main__':main()
