"""Compare analytic initializations with ZERO neural corrections; no training."""
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from flow_operator.model import Config,initialize,make_functions
from flow_operator.sampling import diagnostic_profiles
from flow_operator.validation import read_cases,compare

def main():
    output=Path('reports/initial_lifts');output.mkdir(parents=True,exist_ok=True)
    held,re,names=diagnostic_profiles(encoder_points=65)
    held={k:v.astype(np.float64) for k,v in held.items()};re=re.astype(np.float64)
    rng=np.random.default_rng(18371);xy=rng.uniform(size=(4,512,2));xy[...,0]=5*xy[...,0]**2
    records={}
    for lift in ['gaussian','oseen']:
        cfg=Config(width=12,latent=6,depth=2,encoder_width=12,lift=lift)
        params=jax.tree_util.tree_map(lambda x:jnp.zeros(x.shape,dtype=jnp.float64),initialize(cfg));fn=make_functions(cfg)
        residual=np.asarray(fn['residual'](params,held,re,xy))
        record=dict(momentum_rms={name:float(np.sqrt(np.mean(np.sum(r[:,:2]**2,axis=1)))) for name,r in zip(names,residual)},
            reference=compare(params,fn,read_cases('validation/references',65)),neural_correction='identically zero',training_steps=0)
        records[lift]=record;print(lift,json.dumps(record),flush=True)
    (output/'metrics.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
