"""Audit all completed adaptive runs, including fits and saved independent replays."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'New_Adaptive_Particle_Results';args=argparse.ArgumentParser();args.add_argument('--variant',default='continuation');args.add_argument('--expected-profiles',type=int,default=9);args=args.parse_args();OUT=BASE/args.variant
expected=json.loads((BASE/'design.json').read_text());fail=[];counts=dict(profiles=0,fits=0,replays=0)
for status in sorted((OUT/'progress').glob('*.json')):
 d=json.loads(status.read_text());name=d['profile']
 if d['phase']!='completed':continue
 counts['profiles']+=1;launch=json.loads((OUT/f'observations/{name}/launches.json').read_text())
 assert len(launch['launches'])==24
 for stage in d['evaluated']:
  n=stage['N'];p=OUT/f'fits/{name}/N{n}';r=json.loads((p/'report.json').read_text());counts['fits']+=1
  assert r['checkpoint_sha256']==expected['source_files_sha256']['models/candidates/pressure_long/checkpoint.npz']
  inlet=np.genfromtxt(p/'inlet.csv',delimiter=',',names=True)
  er=np.linalg.norm(inlet['estimated']-inlet['truth'])/np.linalg.norm(inlet['truth'])
  np.testing.assert_allclose(er,r['inlet_relative_l2'],rtol=1e-12)
  selected=r['candidates'][r['selected_start']]
  assert selected['objective']==min(c['objective'] for c in r['candidates'] if c['valid_inlet'])
  assert r['config']['max_iterations']==d['config']['max_iterations'] and r['config']['starts']==2
  if r.get('initialization_source_N') is not None:assert r['initialization_source_N']<n
  for k,tr in enumerate(r['trajectories'],1):
   obs=np.genfromtxt(OUT/f'observations/{name}/track_{k}.csv',delimiter=',',names=True)
   replay=np.genfromtxt(p/f'replay_{k}.csv',delimiter=',',names=True)
   np.testing.assert_allclose(replay['t'],obs['t'][:len(replay)],atol=1e-14)
   error=np.column_stack([replay['x']-obs['x'][:len(replay)],replay['y']-obs['y'][:len(replay)]])
   np.testing.assert_allclose(np.sqrt(np.mean(error**2)),tr['position_rmse_observed'],rtol=1e-10,atol=1e-14)
   counts['replays']+=1
  assert stage['complete']==all(t['completed'] for t in r['trajectories'])
report=dict(passed=counts['profiles']==args.expected_profiles,counts=counts,scope='Saved-fit/replay consistency; no uniqueness or global-optimum claim')
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if counts['profiles']!=args.expected_profiles:raise SystemExit('Study still incomplete')
