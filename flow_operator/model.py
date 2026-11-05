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
    width: int = 128
    latent: int = 64
    depth: int = 4
    encoder_width: int = 48
    length: float = 5.
    seed: int = 91837
    lift: str = 'oseen'
    functional_moments: bool = False
    encoder_kind: str = 'sensor'
    sensor_count: int = 257
    separate_pressure_trunk: bool = False

    def __post_init__(self):
        if min(self.width,self.latent,self.depth,self.encoder_width)<1 or self.sensor_count<16:
            raise ValueError('Network dimensions must be positive and sensor_count >=16')
        if self.lift not in ['gaussian','oseen'] or self.encoder_kind not in ['pooled','sensor']:
            raise ValueError('Unknown model configuration')
        if self.encoder_kind=='sensor' and self.functional_moments:
            raise ValueError('The direct sensor branch already receives the full sensor vector; disable functional_moments')

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
    branch_inputs=config.sensor_count+2 if config.encoder_kind=='sensor' else 2*config.encoder_width+2+(2*len(FREQUENCIES)+1 if config.functional_moments else 0)
    outputs=config.latent if config.separate_pressure_trunk else 2*config.latent
    params=dict(branch=mlp(branch_inputs,outputs,config.width,config.depth),
                trunk=mlp(10+2*len(FREQUENCIES),outputs,config.width,config.depth))
    if config.separate_pressure_trunk:
        params['pressure_branch']=mlp(branch_inputs,config.latent,config.width,config.depth)
        params['pressure_trunk']=mlp(10+2*len(FREQUENCIES),config.latent,config.width,config.depth)
    if config.encoder_kind=='pooled':
        params['encoder']=[layer(2+2*len(FREQUENCIES),config.encoder_width),layer(config.encoder_width,config.encoder_width)]
    return params


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


def branch_outputs(params,inputs,config):
    if config.separate_pressure_trunk:
        return jnp.stack([mlp(params['branch'],inputs),mlp(params['pressure_branch'],inputs)])
    return mlp(params['branch'],inputs).reshape(2,config.latent)


def context(params,profile,re,config):
    if config.encoder_kind=='sensor':
        # The paper's direct-sensor branch: ALL samples, not pooled moments.
        # The supplied spline itself (possibly with a different knot grid) still
        # enters the hard boundary unchanged. This grid is only the branch input.
        sensors=jnp.linspace(0.,1.,config.sensor_count)
        sampled=jax.vmap(lambda y:spline(profile,y)[0])(sensors)
        inputs=jnp.concatenate([sampled-6*sensors*(1-sensors),
            jnp.array([jnp.log(re/10)/jnp.log(10.),jnp.log1p(profile['effective_mode'])/5])])
        return branch_outputs(params,inputs,config)
    y,f,w=profile['encoder_y'],profile['encoder_f'],profile['encoder_w']
    phase=jnp.pi*y[:,None]*jnp.array(FREQUENCIES)
    encoded=jnp.concatenate([y[:,None],(f-6*y*(1-y))[:,None],jnp.sin(phase),jnp.cos(phase)],axis=1)
    for p in params['encoder']: encoded=jnp.tanh(dense(p,encoded))
    mean=jnp.sum(w[:,None]*encoded,axis=0)
    second=jnp.sum(w[:,None]*encoded**2,axis=0)
    inputs=jnp.concatenate([mean,second,jnp.array([jnp.log(re/10)/jnp.log(10.),jnp.log1p(profile['effective_mode'])/5])])
    if config.functional_moments:
        # Strong condition features, alongside the learned pointwise encoder.
        # They do NOT reconstruct or replace the inlet spline in the boundary.
        delta=f-6*y*(1-y)
        moments=jnp.concatenate([jnp.sum((w*delta)[:,None]*jnp.sin(phase),axis=0),
                                 jnp.sum((w*delta)[:,None]*jnp.cos(phase),axis=0),
                                 jnp.array([jnp.sum(w*delta*delta)])])
        inputs=jnp.concatenate([inputs,moments])
    return branch_outputs(params,inputs,config)


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
    if config.separate_pressure_trunk:
        # psi enters NS through third derivatives, p through first derivatives.
        # Separate scalar DeepONets prevent their spatial bases from competing.
        pressure_features=jnp.concatenate([features[:2],jnp.sin(phase),jnp.cos(phase),features[2+2*len(FREQUENCIES):]])
        raw=jnp.array([jnp.sum(branch[0]*mlp(params['trunk'],features)),
                       jnp.sum(branch[1]*mlp(params['pressure_trunk'],pressure_features))])/jnp.sqrt(float(config.latent))
    else:
        raw=jnp.sum(branch*mlp(params['trunk'],features).reshape(2,config.latent),axis=1)/jnp.sqrt(float(config.latent))
    inlet,integral=spline(profile,y)
    base=3*y*y-2*y**3
    pressure_lift=0.
    if config.lift=='oseen':
        # Linearized half-plane NS lift. This is an analytic prior, not a CFD
        # label or a replacement for the full nonlinear NS objective.
        k=jnp.pi*profile['effective_mode']
        alpha=2*k*k/(jnp.sqrt(re*re+4*k*k)+re)
        delta=re*alpha/(k+alpha)
        decay=jnp.exp(-alpha*x)*(1-alpha*jnp.expm1(-delta*x)/delta)
        pressure_lift=(k+alpha)/re*(inlet-6*y*(1-y))*jnp.exp(-k*x)
    elif config.lift=='gaussian':
        ell=.05+.05*re/profile['effective_mode']**2
        decay=jnp.exp(-(x/ell)**2)
    else:
        raise ValueError('Unknown lift configuration')
    # Entire spline antiderivative, including ALL supplied oscillations.
    lift=base+(integral-base)*decay
    mask=16*y*y*(1-y)**2*(1-jnp.exp(-x/.1))**2
    psi=lift+mask*raw[0]
    pressure=12/re*(config.length-x)+pressure_lift+raw[1]
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
