"""Input functions only. No PDE responses, reference solver, or old imports."""
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cholesky
from .profiles import Inlet, stack

FAMILIES=('grf','oscillatory','beta','flat','jets','random_spline','parabolic')


@lru_cache(maxsize=6)
def grf_factor(scale):
    y=np.linspace(0,1,257)
    return cholesky(np.exp(-.5*((y[:,None]-y[None,:])/scale)**2)+1e-10*np.eye(len(y)),lower=True)


def draw(rng,family,points=257):
    y=np.linspace(0,1,points);base=6*y*(1-y)
    if family=='grf':
        g=grf_factor(float(rng.choice([.015,.03,.06,.12,.25,.4])))@rng.normal(size=257)
        g=np.interp(y,np.linspace(0,1,257),g)
        u=(y*(1-y))**rng.choice([1,2,4])*np.logaddexp(0,rng.uniform(.3,1.3)*g)
    elif family=='oscillatory':
        cycles=rng.integers(1,21,size=rng.integers(1,5))
        a=rng.normal(size=len(cycles));a*=rng.uniform(.05,.75)/np.abs(a).sum()
        u=base*(1+np.sum(a[:,None]*np.cos(2*np.pi*cycles[:,None]*y+rng.uniform(0,2*np.pi,len(a))[:,None]),axis=0))
    elif family=='beta':
        a,b=rng.integers(1,7,size=2)
        if (a,b)==(4,4):a=5  # Reserve Beta(5,5) as an unseen profile.
        u=y**a*(1-y)**b
    elif family=='flat':
        d=rng.uniform(.025,.2);power=rng.choice([1,2])
        u=-np.expm1(-(y/d)**power)*-np.expm1(-((1-y)/d)**power)
    elif family=='jets':
        centers=rng.uniform(.1,.9,rng.integers(1,5));width=rng.uniform(.07,.25,len(centers))
        u=np.sum(rng.uniform(.2,1.2,len(centers))[:,None]*np.exp(-((y-centers[:,None])/width[:,None])**2),axis=0)
        u*=(-np.expm1(-y/.04))*(-np.expm1(-(1-y)/.04))
    elif family=='random_spline':
        knots=np.linspace(0,1,rng.integers(5,18))
        g=CubicSpline(knots,rng.normal(size=len(knots)))(y)
        u=(y*(1-y))**rng.choice([1,2,3])*np.logaddexp(0,g)
    elif family=='parabolic':
        u=base
    else:raise ValueError(f'Unknown family {family}')
    u[[0,-1]]=0
    return Inlet.from_samples(y,u)


def make_pool(count=10000,seed=9703107,points=257,encoder_points=257):
    rng=np.random.default_rng(seed);profiles=[];families=[]
    # Retain the no-slip/unit-mean convention, with no four-mode restriction.
    while len(profiles)<count:
        # Include exact Poiseuille as an input condition, never as a response label.
        family=str(rng.choice(FAMILIES,p=[.22,.25,.13,.12,.12,.12,.04]))
        p=draw(rng,family,points)
        check=p.spline(np.linspace(0,1,1025))
        if check.min() < -1e-7 or check.max()>3.0:continue
        p.encoder_points=encoder_points;profiles.append(p);families.append(family)
    data=stack(profiles)
    return data,dict(seed=seed,count=count,input_points=points,encoder_points=encoder_points,
        families={f:families.count(f) for f in FAMILIES},response_labels_present=False,
        reserved_profiles=['exact Beta(5,5); other diagnostic profiles are checked against the assembled pool'],
        training_speed_cap=3.,profile_source='seven input-only families; no CFD',
        note='Training coverage is finite; the inference interface does not restrict profiles to these generators.')


def diagnostic_profiles(points=257,encoder_points=257):
    y=np.linspace(0,1,points)
    raw=[('poiseuille',6*y*(1-y),40.),
         ('nine_oscillations',6*y*(1-y)*(1-.6*np.cos(18*np.pi*y)),55.),
         ('beta_5_5',630*y**4*(1-y)**4,30.),
         ('flat',(-np.expm1(-(y/.05)**2))*(-np.expm1(-((1-y)/.05)**2)),70.)]
    profiles=[Inlet.from_samples(y,u,encoder_points) for _,u,_ in raw]
    return stack(profiles),np.array([r for _,_,r in raw],np.float32),[name for name,_,_ in raw]
