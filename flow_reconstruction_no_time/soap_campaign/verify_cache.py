"""Compare cached SOAP fields and physics against the original evaluator."""
import jax
import jax.numpy as jnp
import numpy as np

from flow_operator.model import context
from .fit import CHECKPOINT, FLOW, OUT, SHA, Estimator, digest, write_json


def main():
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    model=Estimator()
    forward=model.m
    rng=np.random.default_rng(23)
    points=jnp.asarray(rng.uniform([.01,.01],[4.99,.99],(43,2)))
    basis=model.cached.basis_velocity(points)
    probe=np.random.default_rng(71105).uniform([.001,.001],[4.999,.999],(FLOW.physics_points,2))
    probe[:,0]=5*(probe[:,0]/5)**2
    residual=[]
    for theta in (np.zeros(model.size), rng.normal(0,.1,model.size)):
        z,_,re=forward.decode(jnp.asarray(theta))
        data=forward.builder.arrays(z)
        branch=context(forward.params,data,re,forward.operator.config)
        batch={key:value[None] for key,value in data.items()}
        slow=np.asarray(forward.functions['fields'](forward.params,batch,re[None],points[None]))[0,:,:2]
        fast=np.asarray(model.cached.velocity(data,re,branch,points,basis))
        physics=np.asarray(forward.functions['residual'](forward.params,batch,re[None],probe[None]))[0,:,:2]
        cached=np.asarray(model.cached.momentum(data,re,branch))
        np.testing.assert_allclose(fast,slow,rtol=1e-10,atol=1e-11)
        np.testing.assert_allclose(cached,physics,rtol=1e-9,atol=1e-10)
        residual.append(dict(max_velocity_difference=float(np.max(abs(fast-slow))),
                             max_momentum_difference=float(np.max(abs(cached-physics)))))
    write_json(OUT/'cache_parity.json',dict(checkpoint_sha256=SHA,jax=jax.__version__,
               integrator_difference=model.parity,checks=residual))


if __name__=='__main__':
    main()
