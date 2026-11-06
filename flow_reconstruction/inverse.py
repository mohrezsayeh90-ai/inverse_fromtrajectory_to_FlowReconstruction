"""Regularized inverse inference. The trained flow weights are never updated."""
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from flow_operator.api import FlowOperator
from flow_operator.model import make_functions,context,point_fields
from .particles import integral_observations,simulate
from .profile import ProfileBuilder


@dataclass(frozen=True)
class InverseConfig:
    reference_reynolds: float = 55.
    mean_speed_bounds: tuple = (.5,1.5)
    profile_points: int = 65
    position_scale: float = .001
    smoothness: float = 1e-5
    regularization_kind: str = 'inlet_curvature'
    physics_weight: float = .02
    physics_scale: float = .1
    physics_points: int = 192
    shape_prior: float = 1e-4
    mean_prior: float = 1e-4
    log_shape_bound: float = 3.
    precondition_shape: bool = True
    starts: int = 3
    max_iterations: int = 400
    quadrature_order: int = 3
    seed: int = 71103

    def __post_init__(self):
        low,high=self.mean_speed_bounds
        values=[self.reference_reynolds,low,high,self.position_scale,self.smoothness,self.shape_prior,self.mean_prior,self.log_shape_bound,self.physics_weight,self.physics_scale]
        if not np.isfinite(values).all() or min(low,self.position_scale,self.reference_reynolds,self.log_shape_bound)<=0 or high<low:
            raise ValueError('Invalid physical scales or bounds')
        if min(self.smoothness,self.shape_prior,self.mean_prior,self.physics_weight)<0 or min(self.starts,self.max_iterations,self.physics_points,self.physics_scale)<1e-12:
            raise ValueError('Regularization must be nonnegative; counts positive')
        if self.regularization_kind not in ['inlet_curvature','log_shape_curvature']:raise ValueError('Unknown regularization kind')
        if not (10<=low*self.reference_reynolds<=high*self.reference_reynolds<=100):
            raise ValueError('Mean-speed bounds must keep Re=mean_speed*reference_reynolds inside [10,100]')


class FlowReconstructor:
    def __init__(self,checkpoint,config=InverseConfig(),allow_unvalidated=False):
        jax.config.update('jax_enable_x64',True)
        self.operator=FlowOperator(checkpoint,allow_unvalidated=allow_unvalidated)
        self.params=jax.tree_util.tree_map(lambda p:p.astype(jnp.float64),self.operator.params)
        self.config=config;self.builder=ProfileBuilder(config.profile_points)
        self.functions=make_functions(self.operator.config)
        self.anchor=(config.profile_points-2)//2
        n=config.profile_points-3
        insert=np.delete(np.eye(config.profile_points-2),self.anchor,axis=1)
        rough=self.builder.second_difference@insert
        eigen,vectors=np.linalg.eigh(np.eye(n)+1e-4*(rough.T@rough))
        self.shape_transform=jnp.asarray((vectors/np.sqrt(eigen))@vectors.T)
        self.bounds=([(None,None)]*n if config.precondition_shape else [(-config.log_shape_bound,config.log_shape_bound)]*n)+[
            tuple(np.log(config.mean_speed_bounds))]

    def decode(self,theta):
        # Fix one log-shape coefficient: additive constants otherwise cancel
        # under normalization and create an artificial unidentifiable parameter.
        free=theta[:-1]
        if self.config.precondition_shape:
            # Invertible Sobolev coordinate change, retaining EVERY nodal mode.
            # The tanh applies the same log-shape bounds smoothly.
            free=self.config.log_shape_bound*jnp.tanh((self.shape_transform@free)/self.config.log_shape_bound)
        z=jnp.concatenate([free[:self.anchor],jnp.zeros(1),free[self.anchor:]])
        speed=jnp.exp(theta[-1]);re=speed*self.config.reference_reynolds
        return z,speed,re

    def prepare(self,tracks):
        self.tracks=tracks
        self.observations=integral_observations(tracks,self.operator.config.length,self.config.quadrature_order)
        d={k:jnp.asarray(v) for k,v in self.observations.items() if k!='counts'}
        cfg=self.config;builder=self.builder;second=jnp.asarray(builder.second_difference)
        probe=np.random.default_rng(71105).uniform([.001,.001],[self.operator.config.length-.001,.999],(1,cfg.physics_points,2))
        probe[...,0]=self.operator.config.length*(probe[...,0]/self.operator.config.length)**2
        probe=jnp.asarray(probe)
        def predictions(theta):
            z,speed,re=self.decode(theta);profile=builder.arrays(z)
            batch={k:v[None] for k,v in profile.items()}
            fluid=self.functions['fields'](self.params,batch,re[None],d['xy'][None])[0,:,:2]*speed
            return d['known']+d['weights']@fluid
        def residual(theta):return ((predictions(theta)-d['observed'])*d['scales'][:,None]/cfg.position_scale).ravel()
        def regularization(theta):
            z,speed,re=self.decode(theta)
            if cfg.regularization_kind=='inlet_curvature':
                profile=builder.arrays(z);a,b=profile['coeff'][0],profile['coeff'][1]
                curvature=jnp.sum((12*a*a+12*a*b+4*b*b)/builder.h**3)
            else:curvature=jnp.mean((second@z)**2)
            physical=0.
            if cfg.physics_weight:
                profile=builder.arrays(z)
                r=self.functions['residual'](self.params,{k:v[None] for k,v in profile.items()},re[None],probe)
                physical=cfg.physics_weight*jnp.mean(jnp.sum((speed**2*r[...,:2]/cfg.physics_scale)**2,axis=-1))
            return cfg.smoothness*curvature+cfg.shape_prior*jnp.mean(z*z)+cfg.mean_prior*theta[-1]**2+physical
        def loss(theta):
            z,_,_=self.decode(theta);profile=builder.arrays(z)
            # Positive nodal values alone do not guarantee a positive cubic.
            dense=builder.dense@(jnp.asarray(builder.base)*jnp.exp(jnp.pad(z,(1,1))))
            negative=jnp.minimum(dense,0)
            return jnp.sum(residual(theta)**2)/len(tracks)+regularization(theta)+1e5*jnp.mean(negative**2)
        self.residual=jax.jit(residual);self.loss_and_gradient=jax.jit(jax.value_and_grad(loss))
        self.predictions=jax.jit(predictions);self.regularization=jax.jit(regularization)
        return self

    def fields(self,theta,xy):
        z,speed,re=self.decode(jnp.asarray(theta));profile=self.builder.arrays(z)
        data={k:v[None] for k,v in profile.items()};xy=np.asarray(xy,float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('xy must be a finite (N,2) array')
        if np.any(xy<0) or np.any(xy[:,0]>self.operator.config.length) or np.any(xy[:,1]>1):raise ValueError('Coordinates outside the channel')
        if not len(xy):return np.empty((0,3))
        pieces=[]
        for block in np.array_split(xy,max(1,int(np.ceil(len(xy)/512)))):
            pieces.append(np.asarray(self.functions['fields'](self.params,data,re[None],block[None]))[0])
        result=np.concatenate(pieces);result[:,:2]*=float(speed);result[:,2]*=float(speed)**2
        return result

    def simulate(self,theta,track):
        z,speed,re=self.decode(jnp.asarray(theta));profile=self.builder.arrays(z)
        branch=context(self.params,profile,re,self.operator.config)
        fn=jax.jit(lambda xy:speed*point_fields(self.params,branch,profile,re,xy,self.operator.config)[:2])
        return simulate(lambda xy:np.asarray(fn(jnp.asarray(xy))),track.time,track.position[0],
                        track.initial_velocity,track.stokes_time,self.operator.config.length)

    def fit(self,tracks):
        start=time.perf_counter();self.prepare(tracks);rng=np.random.default_rng(self.config.seed)
        def objective(x):
            value,gradient=self.loss_and_gradient(jnp.asarray(x))
            if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
                raise RuntimeError('Nonfinite inverse objective or gradient')
            return float(value),np.asarray(gradient,float)
        candidates=[];histories=[]
        initial_speed=np.clip(1.,*self.config.mean_speed_bounds)
        for index in range(self.config.starts):
            x=np.zeros(len(self.bounds));x[-1]=np.log(initial_speed)
            if index:x[:-1]=rng.normal(0,.06,len(x)-1)
            history=[]
            def callback(theta):history.append(dict(iteration=len(history)+1,objective=objective(theta)[0]))
            result=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=self.bounds,callback=callback,
                            options=dict(maxiter=self.config.max_iterations,ftol=1e-12,gtol=1e-6,maxls=40))
            z,speed,re=self.decode(jnp.asarray(result.x));minimum=self.builder.minimum(z)
            record=dict(start=index,objective=float(result.fun),iterations=int(result.nit),
                        optimizer_success=bool(result.success),optimizer_message=str(result.message),
                        mean_speed=float(speed),reynolds=float(re),spline_minimum=minimum,valid_inlet=minimum>=-1e-7)
            candidates.append(dict(theta=result.x,record=record));histories.append(history)
        valid=[i for i,c in enumerate(candidates) if c['record']['valid_inlet']]
        if not valid:raise RuntimeError('No nonnegative spline candidate; increase regularization or input resolution')
        selected=min(valid,key=lambda i:candidates[i]['record']['objective']);theta=candidates[selected]['theta']
        sims=[self.simulate(theta,t) for t in tracks];track_records=[]
        for track,sim in zip(tracks,sims):
            n=len(sim['time']);error=sim['position']-track.position[:n]
            track_records.append(dict(name=track.name,completed=sim['completed'],boundary_exit_time=sim['boundary_exit_time'],
                                      observed_points=len(track.time),simulated_points=n,
                                      position_rmse=float(np.sqrt(np.mean(error**2))),max_position_error=float(np.linalg.norm(error,axis=1).max())))
        # Observability of data only; exclude priors and parameters fixed by bounds.
        # Column-wise JVP avoids materializing a batched reverse-mode tape for
        # every observed coordinate at once on small CPU machines.
        tangent=jax.jit(lambda x,v:jax.jvp(self.residual,(x,),(v,))[1])
        jac=np.column_stack([np.asarray(tangent(jnp.asarray(theta),jnp.asarray(v))) for v in np.eye(len(theta))])
        free=[i for i,(lo,hi) in enumerate(self.bounds) if lo is None or hi is None or hi>lo];singular=np.linalg.svd(jac[:,free],compute_uv=False)
        rank=int(np.sum(singular>max(float(singular[0])*1e-5,1e-12))) if len(singular) else 0
        z,speed,re=self.decode(jnp.asarray(theta));profile=self.builder.inlet(z)
        xy=rng.uniform([.001,.001],[self.operator.config.length-.001,.999],(1,512,2));xy[...,0]=self.operator.config.length*(xy[...,0]/self.operator.config.length)**2
        residual=np.asarray(self.functions['residual'](self.params,{k:v[None] for k,v in self.builder.arrays(z).items()},re[None],xy))[0]
        unchanged=hashlib.sha256(Path(self.operator.path).read_bytes()).hexdigest()==self.operator.sha256
        if not unchanged:raise RuntimeError('Forward checkpoint changed during reconstruction')
        report=dict(status='experimental_regularized_inverse_fit',forward_checkpoint_sha256=self.operator.sha256,
                    forward_training_step=self.operator.metadata['step'],forward_operator_benchmark_accepted=self.operator.benchmark_accepted,
                    forward_weights_unchanged=unchanged,config=asdict(self.config),selected_start=selected,
                    candidates=[c['record'] for c in candidates],mean_speed=float(speed),reynolds=float(re),
                    trajectories=track_records,trajectory_consistency_passed=all(t['completed'] and t['position_rmse']<max(5*self.config.position_scale,.005) for t in track_records),
                    inlet=profile.diagnostics(),physical_momentum_rms=float(speed**2*np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1)))),
                    divergence_rms=float(speed*np.sqrt(np.mean(residual[:,2]**2))),
                    data_jacobian_singular_values=singular.tolist(),data_jacobian_effective_rank=rank,free_parameters=len(free),
                    rank_threshold_relative=1e-5,unique_flow_guaranteed=False,
                    uncertainty_notice='Candidate spread is empirical optimizer sensitivity, not a calibrated confidence interval. Local rank is parameterization- and threshold-dependent.',
                    pressure_notice='Pressure is supplied conditionally by the forward operator; Stokes trajectories do not observe it directly.',
                    seconds_including_compilation=time.perf_counter()-start)
        return dict(theta=theta,candidates=candidates,valid_candidates=valid,simulations=sims,report=report,histories=histories,
                    inlet=profile,integral_prediction=np.asarray(self.predictions(jnp.asarray(theta))))
