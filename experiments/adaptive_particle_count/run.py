"""Adaptive synthetic inverse-flow study; additive outputs, frozen forward model.
Run with the pinned FLOW checkout and a profile or all nine profiles.
Truth-guided count selection is evaluation-only and is not a real-data stopping rule.
"""
import os
os.environ['JAX_ENABLE_X64']='true'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/workspace/setup/mpl-cache'
import argparse,json,sys,hashlib,time,dataclasses
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--flow',type=Path,required=True);p.add_argument('--profile');p.add_argument('--cpu',type=int);p.add_argument('--variant',default='continuation');p.add_argument('--smoothness',type=float,default=1e-5);p.add_argument('--physics-weight',type=float,default=.02);p.add_argument('--launch-design',choices=['mixed','near-inlet'],default='mixed');p.add_argument('--max-iterations',type=int,default=200);a=p.parse_args()
if a.cpu is not None:os.sched_setaffinity(0,{a.cpu})
sys.dont_write_bytecode=True
sys.path.insert(0,str(a.flow))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from flow_reconstruction.particles import Track,simulate
from flow_operator.profiles import Inlet
import experiments.particle_count.fit_campaign as legacy
from experiments.particle_count.campaign import sha
OUT=Path(__file__).resolve().parents[2]/'New_Adaptive_Particle_Results'
COLD=OUT
OUT=OUT/a.variant
OUT.mkdir(exist_ok=True)
if not (OUT/'observations').exists():
 if a.launch_design=='mixed':(OUT/'observations').symlink_to('../observations')
 else:(OUT/'observations').mkdir()
legacy.OUT=OUT
legacy.CONFIG=dataclasses.replace(legacy.CONFIG,smoothness=a.smoothness,physics_weight=a.physics_weight,max_iterations=a.max_iterations)
CURRENT_CONFIG=json.loads(json.dumps(dataclasses.asdict(legacy.CONFIG)))
NAMES=['beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed','waves3_r30','lobes4_r30','nine30']
SEED=284913;MAX_N=24;TARGET=.05
assert sha(legacy.CP)==legacy.SHA

def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,h,d):p.parent.mkdir(parents=True,exist_ok=True);np.savetxt(p,d,delimiter=',',header=h,comments='')
def profile(name):
 d=np.load(a.flow/'results/particle_count/reference'/name/'reference.npz')
 return Inlet.from_samples(d['inlet_y'],d['inlet_values'])
legacy.profile=profile

def observations(name):
 path=OUT/'observations'/name;register=path/'launches.json'
 if register.exists():
  saved=json.loads(register.read_text());assert saved['seed']==SEED and len(saved['launches'])==MAX_N;return saved
 d=np.load(a.flow/'results/particle_count/reference'/name/'reference.npz');xy=d['xy'];x=np.unique(xy[:,0]);y=np.unique(xy[:,1])
 grid=RegularGridInterpolator((x,y),d['reference'][:,:2].reshape(len(x),len(y),2),bounds_error=True)
 velocity=lambda q:np.asarray(grid(np.asarray(q)[None]))[0]
 rng=np.random.default_rng(SEED);t=1.2*np.linspace(0,1,81)**(1.5 if a.launch_design=='near-inlet' else 1);launches=[];attempts=[]
 # First two match the archived design; all additions are seeded and truth-independent.
 while len(launches)<MAX_N:
  k=len(launches);trial=len(attempts)
  if trial<2:q0,v0=legacy.LAUNCHES[trial];kind='archived-base'
  elif a.launch_design=='near-inlet':
   yy=.035+.93*(((k-2)*.61803398875+.5)%1);q0=[float(rng.uniform(.005,.009)),float(yy)];v0=[float(rng.uniform(.05,.4)),float(rng.uniform(-.1,.1))];kind='near-inlet-stratified-random'
  elif trial%2==0:
   yy=(.08+.84*((k*.61803398875)%1));q0=[float(rng.uniform(.05,.4)),yy];v0=[float(rng.uniform(.5,3.5)),float(rng.uniform(-.45,.45))];kind='stratified-random'
  else:
   q0=[float(rng.uniform(.08,3.8)),float(rng.uniform(.08,.92))];speed=rng.uniform(.3,4);angle=rng.uniform(-np.pi,np.pi);v0=[float(speed*np.cos(angle)),float(speed*np.sin(angle))];kind='random-direction'
  try:
   sim=simulate(velocity,t,q0,v0,.42)
   valid=sim['completed'] and np.all(sim['position']> [x[0],y[0]]) and np.all(sim['position']<[x[-1],y[-1]])
  except ValueError:valid=False
  attempts.append(dict(position=q0,velocity=v0,kind=kind,accepted=bool(valid)))
  if not valid:
   if trial>3000:raise RuntimeError('Insufficient valid launches')
   continue
  refined=simulate(velocity,t,q0,v0,.42,rtol=1e-11,atol=1e-13)
  assert refined['completed']
  noise=np.random.default_rng(SEED+1000+k).normal(0,2e-4,sim['position'].shape);obs=sim['position']+noise;obs[0]=q0
  k+=1
  save(path/f'clean_{k}.csv','t,x,y,vx,vy',np.column_stack([t,sim['position'],sim['velocity']]))
  save(path/f'track_{k}.csv','t,x,y',np.column_stack([t,obs]))
  launches.append(dict(particle=k,position=q0,velocity=v0,kind=kind,reference_refinement_max=float(np.abs(sim['position']-refined['position']).max())))
 record=dict(seed=SEED,noise_std=2e-4,horizon=1.2,St=.42,launches=launches,attempts=attempts,reference_sha256=sha(a.flow/'results/particle_count/reference'/name/'reference.npz'))
 dump(register,record);return record

def run(name):
 register=observations(name);fitter=legacy.CampaignFitter();records=[];counts=[];n=2
 def fit(n):
  folder=OUT/'fits'/name/f'N{n}';path=folder/'report.json'
  # Reuse completed discovery fits except known regressions relative to an earlier fit.
  cold=COLD/'fits'/name/f'N{n}'
  if a.smoothness==1e-5 and a.physics_weight==.02 and a.launch_design=='mixed' and a.max_iterations==200 and not path.exists() and (cold/'report.json').exists():
   cr=json.loads((cold/'report.json').read_text())
   previous=[json.loads(p.read_text()) for p in (OUT/'fits'/name).glob('N*/report.json')]
   if not previous or cr['inlet_relative_l2']<=min(r['inlet_relative_l2'] for r in previous):
    import shutil;folder.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(cold,folder)
  if path.exists():
   r=json.loads(path.read_text())
   if r['config']!=CURRENT_CONFIG:raise ValueError('Existing fit uses another configuration; select a new --variant')
  else:
   tracks=[]
   for k,launch in enumerate(register['launches'][:n],1):
    d=np.genfromtxt(OUT/f'observations/{name}/track_{k}.csv',delimiter=',',names=True)
    tracks.append(Track(f'p{k}',d['t'],np.column_stack([d['x'],d['y']]),launch['velocity'],.42))
   previous=sorted([p for p in (OUT/'fits'/name).glob('N*/report.json') if int(p.parent.name[1:])<n],key=lambda p:int(p.parent.name[1:]))
   warm=None
   if previous:
    prior=json.loads(previous[-1].read_text());warm=np.asarray(prior['candidates'][prior['selected_start']]['theta'])
   original=legacy.minimize
   def continuation_minimize(fun,x,**kwargs):
    if warm is not None and np.count_nonzero(x)>0:x=warm.copy()
    return original(fun,x,**kwargs)
   legacy.minimize=continuation_minimize
   try:r=fitter.fit(name,tracks)
   finally:legacy.minimize=original
   r['initialization_policy']='zero and previous-count selected solution' if warm is not None else 'two original initializations'
   r['initialization_source_N']=int(previous[-1].parent.name[1:]) if warm is not None else None
   r['data_policy']='Only first N tracks; continuation uses fewer-track fits only; truth for evaluation only'
   dump(path,r)
  return dict(N=n,error=r['inlet_relative_l2'],complete=r['status']=='completed',optimizer_converged=r['candidates'][r['selected_start']]['optimizer_success'])
 reason='maximum count reached'
 while True:
  r=fit(n);records.append(r);counts.append(n)
  dump(OUT/'progress'/f'{name}.json',dict(profile=name,phase='adaptive',evaluated=records))
  if r['error']<=TARGET and r['complete']:reason='evaluation target reached';break
  if n>=MAX_N:break
  if (a.launch_design!='near-inlet' or n>=MAX_N) and len(records)>=4 and all(v['error']>=records[-4]['error']*.95 for v in records[-3:]):reason='error plateau under fixed fitting budget';break
  n=min(MAX_N,max(n+2,int(np.ceil(n*(1.8 if r['error']>.25 else 1.5)))))
 # Add intermediate counts for approximately six honest progression states.
 searched=counts.copy();final=counts[-1]
 available=sorted(set(counts)|{int(p.parent.name[1:]) for p in (OUT/'fits'/name).glob('N*/report.json') if int(p.parent.name[1:])<=final})
 if len(available)>=5:
  display=sorted(set(min(available,key=lambda n:abs(n-target)) for target in np.linspace(2,final,6)))
 elif len(counts)>=6:
  display=[counts[i] for i in np.rint(np.linspace(0,len(counts)-1,6)).astype(int)]
 elif len(counts)>=5:display=counts.copy()
 elif final<=7:display=list(range(2,final+1))
 else:
  display=counts.copy()
  while len(display)<5:
   display.sort();left,right=max(zip(display[:-1],display[1:]),key=lambda z:z[1]-z[0]);display.append((left+right)//2)
 display=sorted(set(display))
 for n in display:
  if n not in counts:records.append(fit(n));counts.append(n)
 dump(OUT/'progress'/f'{name}.json',dict(profile=name,phase='completed',evaluated=sorted(records,key=lambda r:r['N']),display_counts=display,adaptive_counts=searched,final_count=final,stopping_reason=reason,target_relative_l2=TARGET,oracle_selection=True,checkpoint_sha256=legacy.SHA,config=dataclasses.asdict(legacy.CONFIG)))
import fcntl
(OUT/'locks').mkdir(exist_ok=True)
for name in ([a.profile] if a.profile else NAMES):
 with (OUT/'locks'/f'{name}.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:
   print('ALREADY_RUNNING',name,flush=True);continue
  status=OUT/'progress'/f'{name}.json'
  if status.exists():
   saved=json.loads(status.read_text())
   if saved.get('config',CURRENT_CONFIG)!=CURRENT_CONFIG:raise ValueError('Existing run uses another configuration; select a new --variant')
   retry_focused=(a.launch_design=='near-inlet' and saved.get('final_count',MAX_N)<MAX_N and saved.get('phase')=='completed' and next(v['error'] for v in saved['evaluated'] if v['N']==saved['final_count'])>TARGET)
   if saved.get('phase')=='completed' and not retry_focused:continue
  run(name)
