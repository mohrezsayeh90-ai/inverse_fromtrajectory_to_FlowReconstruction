"""N48 sensitivity to an alternative geometry-only clock seed and larger budget."""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from .fit import BASE, CHECKPOINT, OUT, SEEDS, SHA, Estimator, digest, pool, write_json

PROFILES = ('beta5','flat30','nine30')
BUDGET = 700


def run(model, seed, profile):
    for design in ('random','hybrid'):
        base = OUT/f'seed_{seed}'/profile/design/'N48'
        fallback = json.loads((base/'report.json').read_text())
        out = OUT/'sensitivity'/f'seed_{seed}'/profile/design
        if (out/'report.json').exists():
            continue
        q,v,source = pool(BASE/f'seed_{seed}'/profile/'observations',design,48)
        assert source==fallback['sources']
        data = model.data(q,v)
        selected,candidates = model.fit(data,budget=BUDGET,use_release_clock=True)
        theta = jnp.asarray(selected['theta'])
        z,speed,_ = model.m.decode(theta[:model.size])
        clock = np.asarray(model.clocks(theta,data))
        assert np.all(np.diff(clock,axis=1)>0)
        with np.load(base/'arrays.npz') as rows:
            y,truth=rows['y'],rows['truth']
        inlet = model.m.builder.inlet(z).spline(y)*float(speed)
        error = float(np.linalg.norm(inlet-truth)/np.linalg.norm(truth))
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'arrays.npz', y=y, inlet=inlet, truth=truth, clock=clock, theta=theta)
        write_json(out/'report.json',dict(profile=profile,seed=seed,design=design,baseline_error=fallback['error'],
                   baseline_cost=fallback['selected']['value'],error=error,
                   start='Stokes relaxation in a parabolic reference guess using arclength and nominal speed',
                   budget=BUDGET,selected=selected,candidates=candidates,
                   source_fit_sha256=digest(base/'arrays.npz'),checkpoint_sha256=SHA))
        print('SENSITIVITY',seed,profile,design,round(100*error,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    args=p.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    model=Estimator()
    for name in PROFILES:
        run(model,args.seed,name)
