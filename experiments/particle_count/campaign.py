"""Matched 2..6-particle inverse-flow campaign. Never trains forward weights."""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
from pathlib import Path
import sys,json,hashlib,time,argparse,dataclasses,tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from flow_operator.profiles import Inlet

OUT=ROOT/'results/particle_count'
NAMES=['parabolic','beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed','waves3_r30','lobes4_r30','nine30']
LABELS={'parabolic':'سهموی','beta5':'بتای باریک','skew_b3':'نامتقارن','flat30':'تقریباً تخت','twin_b3_r30':'دوقله‌ای','shoulder':'دارای شانه','mixed':'چندمقیاسی','waves3_r30':'سه‌نوسانی','lobes4_r30':'چهارنوسانی','nine30':'نه‌نوسانی'}
CP=ROOT/'models/candidates/pressure_long/checkpoint.npz'
SHA='ccf6fa2b3e179a22c443384083e2dec338fc497f9562a567ed7e0955d3c67bf8'
LAUNCHES=[([.03,.97],[6.,-1.2]),([4.9,.03],[-6.,1.2]),([.03,.15],[1.5,0.]),([.03,.85],[1.5,0.]),([.03,.35],[1.5,0.]),([.03,.65],[1.5,0.])]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,a):
 p=Path(p)
 with tempfile.NamedTemporaryFile(mode='w',dir=p.parent,prefix=p.name+'.',suffix='.tmp',delete=False) as f:
  json.dump(a,f,indent=2,allow_nan=False);f.write('\n');temp=f.name
 os.replace(temp,p)
def csvsave(p,header,a):np.savetxt(p,a,delimiter=',',header=header,comments='')
def fields_grid():
 x,y=np.meshgrid(np.r_[np.linspace(.002,.3,25),np.linspace(.4,4.99,48)],np.linspace(.002,.998,97),indexing='ij')
 return np.stack([x,y],-1).reshape(-1,2)
def profile(name):
 y=np.linspace(0,1,257)
 if name=='parabolic':u=6*y*(1-y)
 elif name=='beta5':u=630*y**4*(1-y)**4
 elif name=='flat30':u=(-np.expm1(-(y/.05)**2))*(-np.expm1(-((1-y)/.05)**2))
 elif name=='nine30':u=6*y*(1-y)*(1-.6*np.cos(18*np.pi*y))
 else:
  a=np.load(ROOT/f'results/diverse_profiles_2026_11_07/data/{name}.npz');y=a['inlet_y'];u=a['inlet_values']
 return Inlet.from_samples(y,u)
def prepare_one(name):
 from validation.fem import solve_reference
 folder=OUT/'reference'/name;folder.mkdir(parents=True,exist_ok=True)
 path=folder/'reference.npz'
 if path.exists() and (folder/'metadata.json').exists():return
 inlet=profile(name);xy=fields_grid();y=np.linspace(0,1,4097)
 rough=float(np.sqrt(np.trapezoid(inlet.spline(y,2)**2,y)))
 if name=='parabolic':
  values=np.column_stack([6*xy[:,1]*(1-xy[:,1]),np.zeros(len(xy)),12/30*(5-xy[:,0])]);meta=dict(kind='exact analytical Poiseuille',mesh_change=dict(velocity_relative_l2=0.,pressure_relative_l2=0.))
 elif name=='beta5':
  source=ROOT/'validation/references/beta.npz';data=np.load(source);np.testing.assert_array_equal(xy,data['xy']);values=data['reference'];meta=json.loads(str(data['metadata']));meta.update(source=str(source.relative_to(ROOT)),source_sha256=sha(source))
 elif name not in ['flat30','nine30']:
  source=ROOT/f'results/diverse_profiles_2026_11_07/data/{name}.npz';data=np.load(source);np.testing.assert_array_equal(xy,data['xy']);values=data['reference'];assert float(data['reynolds'])==30
  meta=json.loads(source.with_suffix('.json').read_text());meta.update(source=str(source.relative_to(ROOT)),source_sha256=sha(source))
 else:
  meshes=[(64,64),(80,96)] if name=='flat30' else [(80,108),(104,144)]
  v=[];records=[]
  for nx,ny in meshes:
   print('FEM',name,nx,ny,flush=True);fn,m=solve_reference(inlet,30,nx,ny)
   v.append(np.concatenate([fn(b) for b in np.array_split(xy,15)]));records.append(m)
  delta=v[0]-v[1];meta=dict(kind='independent Taylor-Hood FEM',meshes=records,mesh_change=dict(velocity_relative_l2=float(np.linalg.norm(delta[:,:2])/np.linalg.norm(v[1][:,:2])),pressure_relative_l2=float(np.linalg.norm(delta[:,2])/np.linalg.norm(v[1][:,2]))));values=v[1]
 assert meta['mesh_change']['velocity_relative_l2']<.01 and meta['mesh_change']['pressure_relative_l2']<.02
 np.savez_compressed(path,xy=xy,reference=values,inlet_y=inlet.y,inlet_values=inlet.values)
 dump(folder/'metadata.json',{**meta,'name':name,'label':LABELS[name],'roughness':rough,'reynolds':30})
 print('REFERENCE COMPLETE',name,'roughness',rough,flush=True)
def make_observations(name):
 from flow_reconstruction.particles import simulate
 folder=OUT/'observations'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'report.json').exists():return
 data=np.load(OUT/'reference'/name/'reference.npz');xy=data['xy'];x=np.unique(xy[:,0]);y=np.unique(xy[:,1])
 if name=='parabolic':velocity=lambda q:np.array([6*q[1]*(1-q[1]),0.])
 else:
  # Strict interior interpolation; no extrapolated or invented boundary nodes.
  grid=RegularGridInterpolator((x,y),data['reference'][:,:2].reshape(len(x),len(y),2),bounds_error=True)
  velocity=lambda q:np.asarray(grid(np.asarray(q)[None]))[0]
 t=np.linspace(0,1.2,81);rng=np.random.default_rng(76113);records=[]
 for i,(q0,v0) in enumerate(LAUNCHES,1):
  sim=simulate(velocity,t,q0,v0,.42);refined=simulate(velocity,t,q0,v0,.42,rtol=1e-11,atol=1e-13)
  assert sim['completed'] and refined['completed']
  obs=sim['position']+rng.normal(0,2e-4,sim['position'].shape);obs[0]=q0
  assert np.all(obs>0) and np.all(obs[:,0]<5) and np.all(obs[:,1]<1)
  csvsave(folder/f'track_{i}.csv','t,x,y',np.column_stack([t,obs]));csvsave(folder/f'clean_{i}.csv','t,x,y,vx,vy',np.column_stack([t,sim['position'],sim['velocity']]))
  records.append(dict(particle=i,initial_position=q0,initial_velocity=v0,St=.42,completed=True,solver_tolerance_maximum_difference=float(np.max(np.abs(sim['position']-refined['position'])))))
 dump(folder/'report.json',dict(generator='Independent analytical/FEM flow; no learned field used; interior linear interpolation of archived FEM samples',noise_std=2e-4,noise_seed=76113,initial_exact=True,tracks=records))
 print('OBSERVATIONS COMPLETE',name,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--observations');a=p.parse_args()
 if a.prepare:prepare_one(a.prepare)
 if a.observations:make_observations(a.observations)
if __name__=='__main__':main()
