"""Unknown clocks + frozen flow operator, using only ordered curve geometry."""
from dataclasses import dataclass,asdict
import hashlib,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from flow_reconstruction.inverse import FlowReconstructor,InverseConfig
from flow_reconstruction.particles import simulate
from flow_operator.model import context,point_fields
from .geometry import resample_curve,curve_metrics

@dataclass(frozen=True)
class GeometryConfig:
    points: int = 129
    clock_knots: int = 33
    clock_smoothness: float = .001
    clock_log_bound: float = 5.
    starts: int = 2
    max_iterations: int = 700
    seed: int = 81206

    def __post_init__(self):
        if any(not isinstance(v,int) for v in [self.points,self.clock_knots,self.starts,self.max_iterations]):raise ValueError('Counts must be integers')
        if not 4<=self.clock_knots<=self.points or min(self.starts,self.max_iterations)<1:raise ValueError('Invalid counts')
        if not np.isfinite([self.clock_smoothness,self.clock_log_bound]).all() or self.clock_smoothness<0 or self.clock_log_bound<=0:raise ValueError('Invalid clock regularization or range')

def positive_clock(parameters,basis,ds,launch_speed,log_bound=5.):
    # log(dt/ds) is anchored by the known launch speed; no measured duration.
    log_ratio=log_bound*jnp.tanh((basis@jnp.r_[jnp.zeros(1),parameters])/log_bound)
    slowness=jnp.exp(log_ratio)/launch_speed
    dt=ds*(slowness[:-1]+slowness[1:])/2
    return jnp.r_[jnp.zeros(1),jnp.cumsum(dt)]

def integral_positions(fluid,time,position0,velocity0,tau,z,w):
    dt=jnp.diff(time)
    sample_time=(time[:-1,None]+dt[:,None]*z).ravel()
    lag=time[1:,None]-sample_time[None,:]
    weights=jnp.where(lag>0,-jnp.expm1(-jnp.maximum(lag,0)/tau),0)*(dt[:,None]*w).ravel()[None,:]
    return position0+tau*(-jnp.expm1(-time[1:]/tau))[:,None]*velocity0+weights@fluid

class GeometryReconstructor:
    def __init__(self,checkpoint,config=GeometryConfig(),forward_config=None):
        self.config=config
        self.forward=FlowReconstructor(checkpoint,forward_config or InverseConfig(reference_reynolds=30.),allow_unvalidated=True)
        self.flow_dim=len(self.forward.bounds)

    def prepare(self,curves):
        cfg=self.config;f=self.forward;self.curves=curves
        if len(curves)!=2:raise ValueError('Supply exactly two ordered curves')
        z,w=np.polynomial.legendre.leggauss(3);z=(z+1)/2;w=w/2
        positions=[];points=[];lengths=[]
        for curve in curves:
            q,s=resample_curve(curve.position,cfg.points);positions.append(q);lengths.append(s[-1])
            points.append((q[:-1,None,:]*(1-z)[None,:,None]+q[1:,None,:]*z[None,:,None]).reshape(-1,2))
        self.resampled=np.asarray(positions);self.lengths=lengths
        q=jnp.asarray(positions);xy=jnp.asarray(np.concatenate(points))[None]
        fraction=np.linspace(0,1,cfg.points)
        # A smooth unknown slowness function, not a reuse of timed trajectories.
        basis=jnp.asarray(CubicSpline(np.linspace(0,1,cfg.clock_knots),np.eye(cfg.clock_knots),axis=0)(fraction))
        v0=jnp.asarray([c.initial_velocity for c in curves]);tau=jnp.asarray([c.stokes_time for c in curves])
        ds=jnp.asarray(lengths)/(cfg.points-1);nz=(cfg.points-1)*3
        def clocks(theta):
            controls=theta[self.flow_dim:].reshape(2,cfg.clock_knots-1)
            return jax.vmap(lambda a,b,c:positive_clock(a,basis,b,c,cfg.clock_log_bound))(controls,ds,jnp.linalg.norm(v0,axis=1))
        def prediction(theta):
            shape,speed,re=f.decode(theta[:self.flow_dim]);profile=f.builder.arrays(shape)
            batch={k:v[None] for k,v in profile.items()}
            fluid=f.functions['fields'](f.params,batch,re[None],xy)[0,:,:2].reshape(2,nz,2)*speed
            tt=clocks(theta)
            return jax.vmap(lambda a,b,c,d,e:integral_positions(a,b,c,d,e,jnp.asarray(z),jnp.asarray(w)))(fluid,tt,q[:,0],v0,tau)
        probe=np.random.default_rng(71105).uniform([.001,.001],[5-.001,1-.001],(1,f.config.physics_points,2));probe[...,0]=5*(probe[...,0]/5)**2;probe=jnp.asarray(probe)
        def components(theta):
            shape,speed,re=f.decode(theta[:self.flow_dim]);profile=f.builder.arrays(shape)
            residual=(prediction(theta)-q[:,1:])/f.config.position_scale
            path=jnp.mean(jnp.sum(residual**2,axis=-1))
            a,b=profile['coeff'][0],profile['coeff'][1]
            curvature=jnp.sum((12*a*a+12*a*b+4*b*b)/f.builder.h**3)
            physics=f.functions['residual'](f.params,{k:v[None] for k,v in profile.items()},re[None],probe)
            physical=f.config.physics_weight*jnp.mean(jnp.sum((speed**2*physics[...,:2]/f.config.physics_scale)**2,axis=-1))
            prior=f.config.smoothness*curvature+f.config.shape_prior*jnp.mean(shape*shape)+f.config.mean_prior*theta[self.flow_dim-1]**2
            controls=jnp.pad(theta[self.flow_dim:].reshape(2,-1),((0,0),(1,0)))
            clock=cfg.clock_smoothness*jnp.mean(jnp.diff(controls,n=2,axis=1)**2)
            negative=jnp.minimum(f.builder.dense@(jnp.asarray(f.builder.base)*jnp.exp(jnp.pad(shape,(1,1)))),0)
            positive=1e5*jnp.mean(negative*negative)
            return jnp.stack([path,prior,physical,clock,positive])
        self.clocks=jax.jit(clocks);self.prediction=jax.jit(prediction);self.components=jax.jit(components)
        self.value_gradient=jax.jit(jax.value_and_grad(lambda theta:jnp.sum(components(theta))))
        self.bounds=f.bounds+[(None,None)]*(2*(cfg.clock_knots-1))
        return self

    def replay(self,theta,curve,index,samples=1001):
        f=self.forward;shape,speed,re=f.decode(jnp.asarray(theta[:self.flow_dim]));profile=f.builder.arrays(shape)
        branch=context(f.params,profile,re,f.operator.config)
        velocity=jax.jit(lambda xy:speed*point_fields(f.params,branch,profile,re,xy,f.operator.config)[:2])
        duration=float(self.clocks(jnp.asarray(theta))[index,-1])
        return simulate(lambda xy:np.asarray(velocity(jnp.asarray(xy))),np.linspace(0,duration,samples),curve.position[0],curve.initial_velocity,curve.stokes_time)

    def fit(self,curves,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True);self.prepare(curves)
        rng=np.random.default_rng(self.config.seed);start_time=time.perf_counter();candidates=[];histories=[]
        def objective(theta):
            value,gradient=self.value_gradient(jnp.asarray(theta))
            if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():raise RuntimeError('Nonfinite objective/gradient')
            return float(value),np.asarray(gradient,float)
        # A generic clock seed depends on geometry and known launch speed only.
        fraction=np.linspace(0,1,self.config.clock_knots)[1:]
        for k in range(self.config.starts):
            theta=np.zeros(len(self.bounds));theta[self.flow_dim:]=np.tile(2*fraction,2)
            if k:
                theta[:self.flow_dim-1]=rng.normal(0,.04,self.flow_dim-1)
                theta[self.flow_dim:]+=rng.normal(0,.05,len(theta)-self.flow_dim)
            history=[]
            def callback(t):
                record=dict(start=k,iteration=len(history)+1,objective=objective(t)[0]);history.append(record)
                if len(history)%50==0:print(json.dumps(record),flush=True)
            opt=minimize(objective,theta,jac=True,method='L-BFGS-B',bounds=self.bounds,callback=callback,
                         options=dict(maxiter=self.config.max_iterations,maxls=50,maxcor=30,ftol=1e-12,gtol=1e-6))
            shape,speed,re=self.forward.decode(jnp.asarray(opt.x[:self.flow_dim]));minimum=self.forward.builder.minimum(shape)
            row=dict(start=k,objective=float(opt.fun),iterations=int(opt.nit),success=bool(opt.success),message=str(opt.message),
                     spline_minimum=minimum,valid=minimum>=-1e-7,mean_speed=float(speed),reynolds=float(re),
                     components=np.asarray(self.components(jnp.asarray(opt.x))).tolist())
            candidates.append(dict(theta=opt.x,record=row));histories.append(history)
            np.savez_compressed(output/f'candidate_{k}.npz',theta=opt.x)
            (output/f'candidate_{k}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        valid=[i for i,c in enumerate(candidates) if c['record']['valid']]
        if not valid:raise RuntimeError('No nonnegative inlet candidate')
        index=min(valid,key=lambda i:candidates[i]['record']['objective']);best=candidates[index];theta=best['theta'];f=self.forward
        shape,speed,re=f.decode(jnp.asarray(theta[:self.flow_dim]));profile=f.builder.inlet(shape)
        clocks=np.asarray(self.clocks(jnp.asarray(theta)));pred=np.asarray(self.prediction(jnp.asarray(theta)))
        simulations=[self.replay(theta,c,i) for i,c in enumerate(curves)]
        metrics=[dict(completed=sim['completed'],**curve_metrics(c.position,sim['position'])) for c,sim in zip(curves,simulations)]
        report=dict(status='experimental_geometry_only_inverse',measured_timestamps_used=False,timed_fit_initialization_used=False,
                    config=asdict(self.config),flow_config=asdict(f.config),selected_start=index,candidates=[c['record'] for c in candidates],
                    mean_speed=float(speed),reynolds=float(re),paths=metrics,inferred_durations=clocks[:,-1].tolist(),
                    checkpoint_sha256=f.operator.sha256,forward_weights_unchanged=hashlib.sha256(Path(f.operator.path).read_bytes()).hexdigest()==f.operator.sha256,
                    component_order=['integral_path','inlet_prior','momentum_residual','clock_smoothness','negative_inlet'],
                    elapsed_seconds=time.perf_counter()-start_time,unique_flow_guaranteed=False)
        for i,(curve,sim) in enumerate(zip(curves,simulations)):
            np.savetxt(output/f'track_{i+1}_forward.csv',np.column_stack([sim['time'],sim['position'],sim['velocity']]),delimiter=',',header='inferred_t,x,y,vx,vy',comments='')
            np.savetxt(output/f'track_{i+1}_latent_clock.csv',np.column_stack([clocks[i],self.resampled[i]]),delimiter=',',header='inferred_t,x,y',comments='')
        np.savez_compressed(output/'fit_arrays.npz',theta=theta,clocks=clocks,integral_prediction=pred,resampled_curves=self.resampled,
                            candidate_theta=np.stack([c['theta'] for c in candidates]))
        np.savetxt(output/'inlet_reference_units.csv',np.column_stack([profile.y,profile.values*float(speed)]),delimiter=',',header='y,u',comments='')
        np.savetxt(output/'inlet_normalized.csv',np.column_stack([profile.y,profile.values]),delimiter=',',header='y,u',comments='')
        x,y=np.linspace(0,5,101),np.linspace(0,1,65);xx,yy=np.meshgrid(x,y,indexing='ij')
        fields=f.fields(theta[:self.flow_dim],np.column_stack([xx.ravel(),yy.ravel()])).reshape(101,65,3)
        np.savez_compressed(output/'reconstructed_flow.npz',x=x,y=y,u=fields[...,0],v=fields[...,1],p=fields[...,2])
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        fitted=dict(format='geometry_only_inverse_v1',theta=theta.tolist(),config=asdict(self.config),flow_config=asdict(f.config),checkpoint_sha256=f.operator.sha256)
        (output/'fitted_model.json').write_text(json.dumps(fitted,indent=2)+'\n')
        with (output/'optimization.jsonl').open('w') as out:
            for history in histories:
                for row in history:out.write(json.dumps(row)+'\n')
        return report
