"""Independent Stokes replay and post-fit comparison with synthetic truth."""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from flow_operator.model import context
from flow_reconstruction.particles import simulate
from flow_reconstruction_no_time.geometry import curve_metrics

from .fit import BASE, CHECKPOINT, OUT, PROFILES, ROOT, SEEDS, SHA, STOKES, Estimator, digest, pool, write_json


def held_out(folder):
    for group in ('test_random','test_near'):
        launches = json.loads((folder/group/'launches.json').read_text())['launches']
        for i, launch in enumerate(launches,1):
            path = folder/group/f'track_{i}.csv'
            data = np.genfromtxt(path, names=True, delimiter=',')
            yield np.column_stack([data['x'],data['y']]), launch['velocity'], float(data['t'][-1]-data['t'][0]), path


def run(model, seed, profile, design):
    destination = OUT/f'seed_{seed}'/profile/design/'N48'
    output = destination/'replay.json'
    if output.exists():
        return
    folder = BASE/f'seed_{seed}'/profile/'observations'
    truth, velocities, sources = pool(folder,design,48)
    with np.load(destination/'arrays.npz',allow_pickle=False) as fit:
        inlet, speed, re = model.m.decode(jnp.asarray(fit['theta'][:model.size]))
        durations = fit['clock'][:, -1]
    profile_data = model.m.builder.arrays(inlet)
    branch = context(model.m.params,profile_data,re,model.m.operator.config)

    def field(xy):
        return np.asarray(model.cached.point(jnp.asarray(xy), branch, profile_data, re, speed))

    train = [(q,v,t,ROOT/source['path']) for q,v,t,source in zip(truth,velocities,durations,sources,strict=True)]
    summaries, arrays = [], {}
    for i,(q,v,duration,path) in enumerate(train+list(held_out(folder)),1):
        result = simulate(field,np.linspace(0,duration,129),q[0],v,STOKES)
        np.testing.assert_array_equal(result['position'][0],q[0])
        archived = np.loadtxt(path,delimiter=',',skiprows=1,usecols=(0,))
        observed_duration = float(archived[-1]-archived[0])
        complete = bool(result['completed'])
        entry = dict(index=i, subset='fit' if i<=48 else 'test', completed=complete,
                     horizon=float(duration), observed_duration=observed_duration,
                     source=str(path.relative_to(ROOT)), source_sha256=digest(path),
                     boundary_exit=result['boundary_exit_time'])
        if len(result['position'])>=4:
            entry.update(curve_metrics(q,result['position'],count=129))
        summaries.append(entry)
        arrays[f'replay_{i}'] = np.column_stack([result['time'],result['position']])
        arrays[f'observed_{i}'] = q
    np.savez_compressed(destination/'replays.npz',**arrays)
    write_json(output,dict(geometry_fit_sha256=digest(destination/'arrays.npz'),tracks=summaries,
                          held_out_policy='Archived horizon supplied only for evaluation from an unseen launch. No clock inferred for held-out paths.'))
    print('REPLAY',seed,profile,design,sum(r['completed'] for r in summaries),len(summaries),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    p.add_argument('profiles',nargs='+',choices=PROFILES)
    args = p.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    model = Estimator()
    for profile in args.profiles:
        for design in ('random','hybrid'):
            run(model,args.seed,profile,design)
