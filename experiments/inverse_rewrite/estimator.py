"""Fit a positive spectral spline from tracks, chosen by disjoint validation tracks."""
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cached_operator import CachedObjective
from flow_operator.model import context
from flow_reconstruction.inverse import FlowReconstructor, InverseConfig
from flow_reconstruction.particles import integral_observations, simulate


@dataclass(frozen=True)
class Candidate:
    label: str
    modes: int
    smoothness: float
    early_weight: float


class Estimator:
    def __init__(self, checkpoint):
        config = InverseConfig(reference_reynolds=30.,physics_weight=0.,profile_points=65)
        self.model = model = FlowReconstructor(checkpoint,config,allow_unvalidated=True)
        self.cached = CachedObjective(SimpleNamespace(m=model),config)
        self.modes = 30
        heights = model.builder.y[1:-1]
        cosine = jnp.asarray(np.cos(np.pi*heights[:,None]*np.arange(1,self.modes+1)))
        builder = model.builder
        operator = model.operator.config

        def values(theta):
            theta = jnp.asarray(theta)
            z = 4*jnp.tanh((cosine@theta[:-1])/4)
            speed = jnp.exp(theta[-1])
            return z,speed,30*speed
        self.values = jax.jit(values)

        def delta(theta,data):
            z,speed,re = values(theta)
            profile = builder.arrays(z)
            branch = context(model.params,profile,re,operator)
            return self.cached.positions(profile,speed,re,data,branch)-data['observed']

        def loss(theta,data,mask,smoothness,early_weight):
            theta = jnp.r_[theta[:-1]*mask,theta[-1]]
            z,_,_ = values(theta)
            profile = builder.arrays(z)
            error = delta(theta,data)
            emphasis = 1+early_weight*data['entrance']
            emphasis /= jnp.mean(emphasis)
            a,b = profile['coeff'][:2]
            curvature = jnp.sum((12*a*a+12*a*b+4*b*b)/builder.h**3)
            raw = builder.dense@(jnp.asarray(builder.base)*jnp.exp(jnp.pad(z,(1,1))))
            return (jnp.mean(emphasis[...,None]*error**2)/1e-6+smoothness*curvature
                    +2e-5*jnp.sum(theta[:-1]**2)+1e5*jnp.mean(jnp.minimum(raw,0)**2))
        self.loss = jax.jit(jax.value_and_grad(loss))
        # Independent tracks are scored identically for every candidate and launch design.
        self.validation = jax.jit(lambda theta,data:jnp.sqrt(jnp.mean(
            ((.4+.6*data['entrance'][...,None])*delta(theta,data)**2))))
        self.residual = jax.jit(delta)

    def data(self,tracks):
        data = self.cached.data(tracks,integral_observations)
        xy = data['observed']
        times = jnp.asarray(np.stack([t.time[1:] for t in tracks]))
        # Avoid overweighting the first points with nearly zero particle response.
        data['entrance'] = jnp.exp(-xy[...,0]/.085)*(-jnp.expm1(-times/.055))**2
        return data

    def solve(self, observations, validation, previous):
        specs = [
            Candidate('coarse',8,2e-7,0.),
            Candidate('fine',30,1e-8,0.),
            Candidate('fine_entrance',30,1e-8,10.),
            Candidate('fine_regularized',30,4e-7,10.),
        ]
        starts = np.zeros(self.modes+1)
        starts[1] = .025
        candidates = []
        if previous is not None:
            candidates.append(self.score('previous',previous,observations,validation,0,False,'candidate carried forward without re-optimizing'))
        for spec in specs:
            mask = np.arange(self.modes)<spec.modes
            bounds = [(-1.5,1.5) if keep else (0.,0.) for keep in mask]+[(np.log(.5),np.log(1.5))]
            inputs = (observations,jnp.asarray(mask),spec.smoothness,spec.early_weight)
            def objective(theta):
                value,gradient = self.loss(theta,*inputs)
                return float(value),np.asarray(gradient,dtype=np.float64)
            initial = starts if spec.label in ('coarse','fine') or previous is None else previous
            result = minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                              options=dict(maxiter=250,ftol=1e-11,gtol=2e-6,maxls=35))
            if spec.label=='coarse':
                starts = result.x
            candidates.append(self.score(spec.label,result.x,observations,validation,result.nit,
                                         result.success,str(result.message)))
        eligible = [c for c in candidates if c['positive']]
        if not eligible:
            raise RuntimeError('Every fitted spline violated positivity')
        selected = min(eligible,key=lambda c:c['validation_score'])
        return selected,candidates

    def score(self,label,theta,observations,validation,iterations,converged,message):
        z,_,_ = self.values(theta)
        return dict(label=label,theta=np.asarray(theta).tolist(),iterations=int(iterations),converged=bool(converged),
                    message=message,positive=self.model.builder.minimum(z)>=-1e-8,
                    training_rmse=float(jnp.sqrt(jnp.mean(self.residual(theta,observations)**2))),
                    validation_score=float(self.validation(theta,validation)))

    def inlet(self,theta,heights):
        z,speed,_ = self.values(theta)
        return self.model.builder.inlet(z).spline(heights)*float(speed)

    def replays(self,theta,tracks):
        z,speed,re = self.values(theta)
        profile = self.model.builder.arrays(z)
        branch = context(self.model.params,profile,re,self.model.operator.config)
        return [simulate(lambda q:np.asarray(self.cached.point(jnp.asarray(q),branch,profile,re,speed)),
                         track.time,track.position[0],track.initial_velocity,track.stokes_time) for track in tracks]
