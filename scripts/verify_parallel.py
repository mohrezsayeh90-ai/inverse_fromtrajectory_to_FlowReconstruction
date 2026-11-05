"""Check that distributed gradients implement the same global physics objective."""
import jax
import jax.numpy as jnp
import numpy as np
from flow_operator.model import Config,initialize,make_functions
from flow_operator.sampling import make_pool
from flow_operator.train import make_step,batch,replicate,shard_batch


def main():
    if len(jax.local_devices())<2:raise RuntimeError('Need two devices (virtual CPU devices also work)')
    cfg=Config(width=8,latent=4,depth=1,separate_pressure_trunk=True)
    params=initialize(cfg);m=jax.tree_util.tree_map(jnp.zeros_like,params)
    data,_=make_pool(4,1837,encoder_points=33)
    b=batch(np.random.default_rng(1840),data,4,4)
    functions=make_functions(cfg)
    single=make_step(functions,'input_curvature_sqrt')
    parallel=make_step(functions,'input_curvature_sqrt','training_devices')
    expected=single(params,m,m,jnp.array(1),b)
    actual=parallel(replicate(params,2),replicate(m,2),replicate(m,2),jnp.array([1,1]),shard_batch(b,2))
    for i in [0,1,2]:
        for a,e in zip(jax.tree_util.tree_leaves(actual[i]),jax.tree_util.tree_leaves(expected[i])):
            np.testing.assert_allclose(np.array(a[0]),np.array(e),rtol=3e-4,atol=3e-6)
            np.testing.assert_allclose(np.array(a[0]),np.array(a[1]),rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(actual[3][0],expected[3],rtol=1e-5)
    print('PASS: two-device and single-device physics update agree; replicas remain synchronized.')


if __name__=='__main__':main()
