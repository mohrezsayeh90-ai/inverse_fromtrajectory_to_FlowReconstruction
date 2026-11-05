"""Non-pickle versioned checkpoint with config, optimizer, and RNG provenance."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from .model import Config, initialize

FORMAT='full_profile_physics_operator_v1'


def save(path,params,config,step,rng_state,optimizer=None,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    flat=jax.tree_util.tree_leaves(params)
    arrays={f'parameter_{i}':np.array(v) for i,v in enumerate(flat)}
    if optimizer is not None:
        for name,tree in zip(['adam_m','adam_v'],optimizer):
            arrays.update({f'{name}_{i}':np.array(v) for i,v in enumerate(jax.tree_util.tree_leaves(tree))})
    arrays['metadata']=np.array(json.dumps(dict(format=FORMAT,config=config.json(),step=int(step),
        rng_state=rng_state,optimizer_present=optimizer is not None,**(metadata or {}))))
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(path)


def load(path,optimizer=False):
    with np.load(path,allow_pickle=False) as d:
        metadata=json.loads(str(d['metadata']))
        if metadata['format']!=FORMAT:raise ValueError('Not a full-profile checkpoint; old V2 weights are incompatible')
        config=Config(**metadata['config']);template=initialize(config)
        flat,structure=jax.tree_util.tree_flatten(template)
        def tree(prefix):
            values=[jnp.asarray(d[f'{prefix}_{i}']) for i in range(len(flat))]
            if any(a.shape!=b.shape for a,b in zip(values,flat)):raise ValueError('Checkpoint parameter shape mismatch')
            return jax.tree_util.tree_unflatten(structure,values)
        params=tree('parameter');state=None
        if optimizer:
            if not metadata['optimizer_present']:raise ValueError('Optimizer is missing')
            state=(tree('adam_m'),tree('adam_v'))
    return params,config,metadata,state
