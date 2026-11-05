"""JIT-compiled implementation of the SAME streamfunction model and NS equations.

Parameter keys deliberately match v2_model.py for numerical parity and export.
"""
import numpy as np
import jax
import jax.numpy as jnp
from .v2_profiles import PROJECT,INTEGRALS
from .profiles import Y

jax.config.update('jax_default_matmul_precision','highest')


def initialize(width=64,latent=32,depth=3,seed=5197,pressure_mode='viscous'):
    rng=np.random.default_rng(seed);params={}
    for prefix,inputs in [('branch',34),('trunk',14)]:
        shapes=[('u',inputs,width),('v',inputs,width)]
        shapes += [(f'gates.{k}',inputs if k==0 else width,width) for k in range(depth)]
        shapes += [('output',width,2*latent)]
        for name,di,do in shapes:
            std=1e-3 if prefix=='trunk' and name=='output' else np.sqrt(2/(di+do))
            params[f'{prefix}.{name}.weight']=jnp.asarray(rng.normal(0,std,(do,di)),dtype=jnp.float32)
            params[f'{prefix}.{name}.bias']=jnp.zeros(do,dtype=jnp.float32)
    return params


def make_functions(latent=32,depth=3,pressure_mode='viscous'):
    projection=jnp.asarray(PROJECT,dtype=jnp.float32)
    integrals=jnp.asarray(INTEGRALS,dtype=jnp.float32)
    baseline=jnp.asarray(6*Y*(1-Y),dtype=jnp.float32)
    alpha=jnp.pi*jnp.arange(1,5,dtype=jnp.float32)
    modes=jnp.arange(1,5,dtype=jnp.float32)
    def dense(params,prefix,x):
        return x@params[prefix+'.weight'].T+params[prefix+'.bias']
    def mlp(params,prefix,x):
        u=jnp.tanh(dense(params,prefix+'.u',x));v=jnp.tanh(dense(params,prefix+'.v',x));h=x
        for i in range(depth):
            z=jnp.tanh(dense(params,prefix+f'.gates.{i}',h));h=z*u+(1-z)*v
        return dense(params,prefix+'.output',h)
    def context(params,profile,re):
        condition=jnp.concatenate([profile-baseline,jnp.array([jnp.log(re/10)/jnp.log(10.)])])
        b=profile@projection.T;b=b/(b[0]+jnp.sum(b[1:]*integrals))
        return mlp(params,'branch',condition).reshape(2,latent),b,re
    contexts=jax.vmap(context,in_axes=(None,0,0))
    def potentials(params,ctx,xy):
        branch,b,re=ctx;x,y=xy
        phase=y*alpha
        features=jnp.concatenate([jnp.array([2*x/5-1,2*y-1]),jnp.sin(phase),jnp.cos(phase),
                    jnp.array([jnp.log1p(20*x)/jnp.log(101.),jnp.exp(-x),jnp.exp(-4*x),jnp.exp(-16*x)])])
        raw=jnp.sum(branch*mlp(params,'trunk',features).reshape(2,latent),-1)/jnp.sqrt(float(latent))
        base=3*y*y-2*y**3
        ik=6*((y-y*y)*jnp.sin(phase)/alpha+(1-2*y)*jnp.cos(phase)/alpha**2+2*jnp.sin(phase)/alpha**3-1/alpha**2)
        ell=.1+.05*re/modes**2
        psi=base+jnp.sum(b[1:]*(ik-base*integrals)*jnp.exp(-(x/ell)**2))
        psi=psi+16*y*y*(1-y)**2*(1-jnp.exp(-x/.2))**2*raw[0]
        pressure=12/re*(5-x)+(12/re if pressure_mode=='viscous' else 1.)*raw[1]
        return psi,pressure
    def point_fields(params,ctx,xy):
        g=jax.grad(lambda z:potentials(params,ctx,z)[0])(xy)
        return jnp.array([g[1],-g[0],potentials(params,ctx,xy)[1]])
    jac=jax.jacfwd(point_fields,argnums=2)
    hessian=jax.jacfwd(jac,argnums=2)
    def point_physics(params,ctx,xy):
        u,v,p=point_fields(params,ctx,xy);g=jac(params,ctx,xy);h=hessian(params,ctx,xy)
        mx=u*g[0,0]+v*g[0,1]+g[2,0]-(h[0,0,0]+h[0,1,1])/ctx[2]
        my=u*g[1,0]+v*g[1,1]+g[2,1]-(h[1,0,0]+h[1,1,1])/ctx[2]
        return jnp.array([mx,my,g[0,0]+g[1,1]])
    def point_outlet(params,ctx,y):
        xy=jnp.array([5.,y]);f=point_fields(params,ctx,xy);g=jac(params,ctx,xy)
        return jnp.array([g[0,0]-ctx[2]*f[2],g[1,0]])
    field_map=jax.vmap(jax.vmap(point_fields,in_axes=(None,None,0)),in_axes=(None,0,0))
    physics_map=jax.vmap(jax.vmap(point_physics,in_axes=(None,None,0)),in_axes=(None,0,0))
    outlet_map=jax.vmap(jax.vmap(point_outlet,in_axes=(None,None,0)),in_axes=(None,0,0))
    def fields(params,profiles,re,xy):return field_map(params,contexts(params,profiles,re),xy)
    def physics(params,profiles,re,xy):return physics_map(params,contexts(params,profiles,re),xy)
    def out(params,profiles,re,y):return outlet_map(params,contexts(params,profiles,re),y)
    def terms(params,batch):
        ctx=contexts(params,batch['profiles'],batch['reynolds'])
        residual=physics_map(params,ctx,batch['xy'])
        momentum=jnp.mean(jnp.sum(residual[...,:2]**2,-1))
        boundary=jnp.mean(jnp.sum(outlet_map(params,ctx,batch['out_y'])**2,-1))
        pred=fields(params,batch['data_profiles'],batch['data_reynolds'],batch['data_xy'])
        scale=jnp.stack([jnp.ones_like(batch['data_reynolds']),jnp.full_like(batch['data_reynolds'],10.),batch['data_reynolds']/60],-1)[:,None,:]
        data=jnp.mean(((pred-batch['target'])*scale)**2)
        return momentum,boundary,data,jnp.sqrt(jnp.mean(residual[...,2]**2))
    def loss(params,batch,weight):
        momentum,boundary,data,mass=terms(params,batch)
        return momentum+boundary+weight*data,jnp.array([momentum,boundary,data,mass])
    def norm(tree):return jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree_util.tree_leaves(tree)))
    def step(params,m,v,count,weight,batch):
        (value,aux),grad=jax.value_and_grad(loss,has_aux=True)(params,batch,weight)
        gn=norm(grad);factor=jnp.minimum(1.,10/(gn+1e-12))
        grad=jax.tree_util.tree_map(lambda g:g*factor,grad)
        m=jax.tree_util.tree_map(lambda old,g:.9*old+.1*g,m,grad)
        v=jax.tree_util.tree_map(lambda old,g:.999*old+.001*g*g,v,grad)
        lr=jnp.maximum(1e-5,1e-3*.9**(count/1000))
        params=jax.tree_util.tree_map(lambda w,a,b:w-lr*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
        return params,m,v,value,aux,gn
    def balance(params,batch):
        gp=jax.grad(lambda theta:sum(terms(theta,batch)[:2]))(params)
        gd=jax.grad(lambda theta:terms(theta,batch)[2])(params)
        return norm(gp),norm(gd)
    return {'fields':jax.jit(fields),'physics':jax.jit(physics),'outlet':jax.jit(out),
            'step':jax.jit(step),'balance':jax.jit(balance),'terms':jax.jit(terms)}
