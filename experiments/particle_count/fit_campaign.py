"""Reuse five JAX programs with dynamic observed paths; same archived objective.

The full 65-node inlet basis and priors are retained. Jacobian SVD is omitted:
this campaign measures reconstruction error, not identifiability rank.
"""
import os,sys,argparse,json,time,dataclasses,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--profile-only');p.add_argument('--parity',action='store_true');args=p.parse_args() if __name__=='__main__' else argparse.Namespace(worker=0,workers=1,profile_only=None,parity=False)
if __name__=='__main__':
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[allowed[args.worker]])
os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from experiments.particle_count.campaign import OUT,NAMES,LAUNCHES,CP,SHA,profile,dump,csvsave,sha
from flow_reconstruction.inverse import FlowReconstructor,InverseConfig
from flow_reconstruction.particles import Track,integral_observations

CONFIG=InverseConfig(reference_reynolds=30.,mean_speed_bounds=(.5,1.5),profile_points=65,starts=2,max_iterations=200,seed=71103)

class CampaignFitter:
 def __init__(self):
  self.m=FlowReconstructor(CP,CONFIG,allow_unvalidated=True);m=self.m;cfg=CONFIG;b=m.builder
  probe=np.random.default_rng(71105).uniform([.001,.001],[4.999,.999],(1,cfg.physics_points,2));probe[...,0]=5*(probe[...,0]/5)**2;probe=jnp.asarray(probe)
  def predictions(theta,d):
   z,speed,re=m.decode(theta);pr=b.arrays(z)
   fluid=m.functions['fields'](m.params,{k:v[None] for k,v in pr.items()},re[None],d['xy'][None])[0,:,:2]*speed
   return d['known']+d['weights']@fluid
  def loss(theta,d,n):
   z,speed,re=m.decode(theta);pr=b.arrays(z)
   residual=((predictions(theta,d)-d['observed'])*d['scales'][:,None]/cfg.position_scale).ravel()
   aa,bb=pr['coeff'][0],pr['coeff'][1]
   curvature=jnp.sum((12*aa*aa+12*aa*bb+4*bb*bb)/b.h**3)
   r=m.functions['residual'](m.params,{k:v[None] for k,v in pr.items()},re[None],probe)
   physical=cfg.physics_weight*jnp.mean(jnp.sum((speed**2*r[...,:2]/cfg.physics_scale)**2,axis=-1))
   regularization=cfg.smoothness*curvature+cfg.shape_prior*jnp.mean(z*z)+cfg.mean_prior*theta[-1]**2+physical
   dense=b.dense@(jnp.asarray(b.base)*jnp.exp(jnp.pad(z,(1,1))))
   penalty=1e5*jnp.mean(jnp.minimum(dense,0)**2)
   return jnp.sum(residual**2)/n+regularization+penalty
  self.loss=jax.jit(jax.value_and_grad(loss));self.predict=jax.jit(predictions)
 def data(self,tracks):
  a=integral_observations(tracks,5,CONFIG.quadrature_order)
  return {k:jnp.asarray(v) for k,v in a.items() if k!='counts'}
 def fit(self,name,tracks,parity=False):
  m=self.m;n=len(tracks);d=self.data(tracks);x=np.zeros(len(m.bounds));begin=time.perf_counter()
  value,grad=self.loss(jnp.asarray(x),d,jnp.array(float(n)));value.block_until_ready();compile_seconds=time.perf_counter()-begin
  if parity:
   m.prepare(tracks);old,oldg=m.loss_and_gradient(jnp.asarray(x))
   np.testing.assert_allclose(value,old,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(grad,oldg,rtol=1e-8,atol=1e-8)
   test=np.random.default_rng(29).normal(0,.02,len(x));newv,newg=self.loss(jnp.asarray(test),d,jnp.array(float(n)));ov,og=m.loss_and_gradient(jnp.asarray(test))
   np.testing.assert_allclose(newv,ov,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(newg,og,rtol=1e-8,atol=1e-8)
   dump(OUT/'objective_parity.json',dict(passed=True,particle_count=n,zero_cost=float(value),nonzero_cost=float(newv),maximum_gradient_difference=float(np.max(np.abs(np.asarray(newg)-np.asarray(og)))),checks='Full original objective and gradient at zero and nonzero coefficients; priors and physical penalty included'))
  start=time.perf_counter();v,g=self.loss(jnp.asarray(x),d,jnp.array(float(n)));v.block_until_ready();gradient_seconds=time.perf_counter()-start
  last={}
  def objective(theta):
   v,g=self.loss(jnp.asarray(theta),d,jnp.array(float(n)))
   if not np.isfinite(float(v)) or not np.isfinite(np.asarray(g)).all():raise RuntimeError('Nonfinite objective/gradient')
   last.update(theta=np.asarray(theta).copy(),value=float(v))
   return float(v),np.asarray(g,float)
  rng=np.random.default_rng(CONFIG.seed);candidates=[]
  for i in range(CONFIG.starts):
   x=np.zeros(len(m.bounds));
   if i:x[:-1]=rng.normal(0,.06,len(x)-1)
   history=[];started=time.perf_counter()
   result=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=m.bounds,callback=lambda z:history.append(last['value'] if np.array_equal(last['theta'],z) else objective(z)[0]),options=dict(maxiter=CONFIG.max_iterations,ftol=1e-12,gtol=1e-6,maxls=40))
   z,speed,re=m.decode(jnp.asarray(result.x));minimum=m.builder.minimum(z)
   candidates.append(dict(start=i,theta=result.x.tolist(),objective=float(result.fun),iterations=int(result.nit),evaluations=int(result.nfev),optimizer_success=bool(result.success),reason=str(result.message),valid_inlet=minimum>=-1e-7,spline_minimum=minimum,mean_speed=float(speed),seconds=time.perf_counter()-started,objective_history=history))
  valid=[i for i,c in enumerate(candidates) if c['valid_inlet']]
  if not valid:raise RuntimeError('No nonnegative final spline')
  selected=min(valid,key=lambda i:candidates[i]['objective']);theta=np.asarray(candidates[selected]['theta']);z,speed,re=m.decode(jnp.asarray(theta));inlet=m.builder.inlet(z)
  y=np.linspace(0,1,4097);truth=profile(name).spline(y);estimate=inlet.spline(y)*float(speed);shape=inlet.spline(y)
  scores=dict(inlet_relative_l2=float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth)),shape_relative_l2=float(np.linalg.norm(shape-truth)/np.linalg.norm(truth)),mean_speed_error=float(abs(speed-1)),inlet_maximum_error=float(np.max(np.abs(estimate-truth))))
  folder=OUT/'fits'/name/f'N{n}';folder.mkdir(parents=True,exist_ok=True)
  csvsave(folder/'inlet.csv','y,truth,estimated,estimated_own_mean',np.column_stack([y,truth,estimate,shape]))
  for candidate in candidates:
   cz,cs,_=m.decode(jnp.asarray(candidate['theta']));est=m.builder.inlet(cz).spline(y)*float(cs)
   candidate['inlet_relative_l2_evaluation_only']=float(np.linalg.norm(est-truth)/np.linalg.norm(truth))
  sims=[]
  for i,track in enumerate(tracks,1):
   sim=m.simulate(theta,track);q=sim['position'];count=len(q);error=q-track.position[:count];clean=np.genfromtxt(OUT/'observations'/name/f'clean_{i}.csv',delimiter=',',names=True);truthq=np.column_stack([clean['x'],clean['y']])
   sims.append(dict(particle=i,completed=sim['completed'],position_rmse_observed=float(np.sqrt(np.mean(error**2))),position_rmse_clean=float(np.sqrt(np.mean((q-truthq[:count])**2))),maximum_observed_euclidean_error=float(np.linalg.norm(error,axis=1).max()),boundary_exit_time=sim['boundary_exit_time']))
   csvsave(folder/f'replay_{i}.csv','t,x,y,vx,vy',np.column_stack([sim['time'],q,sim['velocity']]))
  assert sha(CP)==SHA
  report=dict(profile=name,N=n,status='completed' if all(s['completed'] for s in sims) else 'partial_replay',**scores,selected_start=selected,candidates=candidates,config=dataclasses.asdict(CONFIG),checkpoint_sha256=SHA,checkpoint_unchanged=True,trajectories=sims,seconds_including_compile=time.perf_counter()-begin,initial_call_seconds=compile_seconds,warm_gradient_seconds=gradient_seconds,
   data_policy='Only the first N observed tracks; no truth inlet or FEM field in fit; identical two initializations and priors for every fit',omitted_diagnostic='Data-Jacobian SVD omitted; no parameter rank or uniqueness claim')
  dump(folder/'report.json',report);print('FIT',name,n,'inlet_percent',100*scores['inlet_relative_l2'],'seconds',report['seconds_including_compile'],flush=True)
  return report

def load_tracks(name,n):
 tracks=[]
 for i,(q0,v0) in enumerate(LAUNCHES[:n],1):
  a=np.genfromtxt(OUT/'observations'/name/f'track_{i}.csv',delimiter=',',names=True)
  tracks.append(Track(f'particle_{i}',a['t'],np.column_stack([a['x'],a['y']]),v0,.42))
 return tracks

def main():
 assert sha(CP)==SHA
 fitter=CampaignFitter()
 names=[args.profile_only] if args.profile_only else [name for i,name in enumerate(NAMES) if i%args.workers==args.worker]
 for name in names:
  for n in range(2,7):
   folder=OUT/'fits'/name/f'N{n}';path=folder/'report.json'
   if path.exists():
    previous=json.loads(path.read_text())
    if previous.get('status')!='failed' and previous.get('config')!=json.loads(json.dumps(dataclasses.asdict(CONFIG))):raise ValueError('Existing configuration differs; use separate results for a new design')
    continue
   try:fitter.fit(name,load_tracks(name,n),parity=args.parity and name==names[0] and n==2)
   except Exception as exc:
    import traceback
    folder.mkdir(parents=True,exist_ok=True);dump(path,dict(profile=name,N=n,status='failed',error=str(exc),traceback=traceback.format_exc()));print('FAILED',name,n,exc,flush=True)
if __name__=='__main__':main()
