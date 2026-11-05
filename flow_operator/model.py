"""Variable-quadrature function encoder + full-profile streamfunction DeepONet.

The input shape is not truncated to a Fourier basis. Positional features enrich
the neural encoder/trunk; they never replace the function in the hard boundary.
"""
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Config:
    width: int = 96
    latent: int = 48
    depth: int = 3
    encoder_width: int = 48
    length: float = 5.
    seed: int = 91837

    def json(self): return asdict(self)


FREQUENCIES = tuple(float(k) for k in range(1,49))


def initialize(config=Config()):
    rng = np.random.default_rng(config.seed)
    def layer(n,m,scale=None):
        return dict(w=jnp.array(rng.normal(0,np.sqrt(2/(n+m)) if scale is None else scale,(n,m)),dtype=jnp.float32),
                    b=jnp.zeros(m,dtype=jnp.float32))
    def mlp(n,m,width,depth):
        return dict(u=layer(n,width),v=layer(n,width),
                    gates=[layer(n if i==0 else width,width) for i in range(depth)], out=layer(width,m,1e-3))
    # Pointwise encoder weights do not depend on the number/locations of samples.
    return dict(encoder=[layer(2+2*len(FREQUENCIES),config.encoder_width),layer(config.encoder_width,config.encoder_width)],
                branch=mlp(2*config.encoder_width+2,2*config.latent,config.width,config.depth),
                trunk=mlp(10+2*len(FREQUENCIES),2*config.latent,config.width,config.depth))


def dense(p,x): return x@p['w']+p['b']


def mlp(p,x):
    u,v=jnp.tanh(dense(p['u'],x)),jnp.tanh(dense(p['v'],x))
    h=x
    for gate in p['gates']:
        z=jnp.tanh(dense(gate,h));h=z*u+(1-z)*v
    return dense(p['out'],h)


def spline(profile,y):
    index=jnp.clip(jnp.searchsorted(profile['knots'],y,side='right')-1,0,len(profile['knots'])-2)
    left=profile['knots'][index];h=profile['knots'][index+1]-left
    t=(y-left)/h
    a,b,c,d=profile['coeff'][:,index]
    f=((a*t+b)*t+c)*t+d
    integral=profile['prefix'][index]+h*(((a*t/4+b/3)*t+c/2)*t+d)*t
    return f,integral


def context(params,profile,re,config):
    y,f,w=profile['encoder_y'],profile['encoder_f'],profile['encoder_w']
    phase=jnp.pi*y[:,None]*jnp.array(FREQUENCIES)
    encoded=jnp.concatenate([y[:,None],(f-6*y*(1-y))[:,None],jnp.sin(phase),jnp.cos(phase)],axis=1)
    for p in params['encoder']: encoded=jnp.tanh(dense(p,encoded))
    mean=jnp.sum(w[:,None]*encoded,axis=0)
    second=jnp.sum(w[:,None]*encoded**2,axis=0)
    inputs=jnp.concatenate([mean,second,jnp.array([jnp.log(re/10)/jnp.log(10.),jnp.log1p(profile['effective_mode'])/5])])
    return mlp(params['branch'],inputs).reshape(2,config.latent)


def potentials(params,branch,profile,re,xy,config):
    x,y=xy
    modes=jnp.array(FREQUENCIES)
    phase=jnp.pi*y*modes
    # These are coordinate embeddings, NOT a projection of the input function.
    # Amplitude scaling conditions high derivatives.
    features=jnp.concatenate([jnp.array([2*x/config.length-1,2*y-1]),
        jnp.sin(phase)/modes,jnp.cos(phase)/modes,
        jnp.array([jnp.exp(-x/s) for s in [.02,.05,.1,.3,1.,3.]]),
        jnp.array([jnp.log1p(20*x)/jnp.log1p(20*config.length),x/config.length])])
    raw=jnp.sum(branch*mlp(params['trunk'],features).reshape(2,config.latent),axis=1)/jnp.sqrt(float(config.latent))
    _,integral=spline(profile,y)
    base=3*y*y-2*y**3
    ell=.05+.05*re/profile['effective_mode']**2
    decay=jnp.exp(-(x/ell)**2)
    # Entire spline antiderivative, including ALL supplied oscillations.
    lift=base+(integral-base)*decay
    mask=16*y*y*(1-y)**2*(1-jnp.exp(-x/.1))**2
    psi=lift+mask*raw[0]
    pressure=12/re*(config.length-x)+raw[1]
    return psi,pressure


def point_fields(params,branch,profile,re,xy,config):
    grad=jax.grad(lambda z:potentials(params,branch,profile,re,z,config)[0])(xy)
    return jnp.array([grad[1],-grad[0],potentials(params,branch,profile,re,xy,config)[1]])


def make_functions(config=Config()):
    jac=jax.jacfwd(point_fields,argnums=4)
    hess=jax.jacfwd(jac,argnums=4)
    def point_residual(params,branch,profile,re,xy):
        u,v,p=point_fields(params,branch,profile,re,xy,config)
        g=jac(params,branch,profile,re,xy,config);h=hess(params,branch,profile,re,xy,config)
        return jnp.array([u*g[0,0]+v*g[0,1]+g[2,0]-(h[0,0,0]+h[0,1,1])/re,
            u*g[1,0]+v*g[1,1]+g[2,1]-(h[1,0,0]+h[1,1,1])/re,g[0,0]+g[1,1]])
    def outlet_one(params,branch,profile,re,y):
        xy=jnp.array([config.length,y]);f=point_fields(params,branch,profile,re,xy,config)
        g=jac(params,branch,profile,re,xy,config)
        return jnp.array([g[0,0]/re-f[2],g[1,0]/re])
    def one_fields(params,profile,re,xy):
        branch=context(params,profile,re,config)
        return jax.vmap(lambda z:point_fields(params,branch,profile,re,z,config))(xy)
    def one_residual(params,profile,re,xy):
        branch=context(params,profile,re,config)
        return jax.vmap(lambda z:point_residual(params,branch,profile,re,z))(xy)
    def one_outlet(params,profile,re,y):
        branch=context(params,profile,re,config)
        return jax.vmap(lambda z:outlet_one(params,branch,profile,re,z))(y)
    fields=jax.vmap(one_fields,in_axes=(None,0,0,0))
    residual=jax.vmap(one_residual,in_axes=(None,0,0,0))
    outlet=jax.vmap(one_outlet,in_axes=(None,0,0,0))
    def terms(params,batch):
        r=residual(params,batch['profiles'],batch['reynolds'],batch['xy'])
        o=outlet(params,batch['profiles'],batch['reynolds'],batch['outlet_y'])
        return jnp.array([jnp.mean(jnp.sum(r[...,:2]**2,axis=-1)),jnp.mean(jnp.sum(o**2,axis=-1)),jnp.mean(r[...,2]**2)])
    return dict(fields=jax.jit(fields),residual=jax.jit(residual),outlet=jax.jit(outlet),terms=jax.jit(terms),raw_terms=terms)
