"""Time-free SOAP comparison for the nine-cycle inlet study on printed page 130."""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from flow_operator.model import context
from flow_operator.profiles import Inlet
from flow_reconstruction.particles import simulate
from ..geometry import curve_metrics
from .fit import BASE, CHECKPOINT, OUT, ROOT, SEEDS, SHA, STOKES, Estimator, digest, write_json
from .replay import held_out

COUNTS = (2,4,10,20,31,48,64,96)
EDGE = .04
REF = ROOT/'results/particle_count/reference/nine30/reference.npz'


def inputs(archive):
    curves, velocity, origins = [], [], []
    for group,count in (('near',32),('extra_near',64)):
        folder=archive/group
        meta=folder/'launches.json'
        launches=json.loads(meta.read_text())['launches']
        assert len(launches)==count
        for i,launch in enumerate(launches,1):
            path=folder/f'track_{i}.csv'
            raw=np.loadtxt(path,delimiter=',',skiprows=1,usecols=(1,2))
            accepted=np.maximum.accumulate(raw[:,0])<=EDGE
            prefix=raw[accepted]
            assert len(prefix)>=4 and np.isfinite(prefix).all()
            curves.append(prefix)
            velocity.append(launch['velocity'])
            origins.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
                                launch_sha256=digest(meta),samples=len(prefix)))
    return curves,np.asarray(velocity),origins


def run(model,seed):
    archive=BASE/f'seed_{seed}'/'nine30'
    fits=OUT/'thesis_samples/spatial_prefix'/f'seed_{seed}'
    q,v,sources=inputs(archive/'observations')
    with np.load(REF,allow_pickle=False) as ref:
        y=np.linspace(0,1,4097)
        truth=Inlet.from_samples(ref['inlet_y'],ref['inlet_values']).spline(y)
    for n in COUNTS:
        folder=fits/f'N{n}'
        if (folder/'report.json').exists():
            continue
        data=model.data(q[:n],v[:n])
        selected,candidates=model.fit(data,use_release_clock=True)
        theta=jnp.asarray(selected['theta'])
        z,speed,re=model.m.decode(theta[:model.size])
        profile=model.m.builder.inlet(z).spline(y)*float(speed)
        clock=np.asarray(model.clocks(theta,data))
        assert np.all(np.diff(clock,axis=1)>0)
        error=float(np.linalg.norm(profile-truth)/np.linalg.norm(truth))
        former=archive/'local_extended_fine'/f'N{n}'/'report.json'
        previous=json.loads(former.read_text())
        assert previous['N']==n
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'arrays.npz',y=y,truth=truth,inlet=profile,clock=clock,
                            curves=np.asarray(data['q']),launch_velocity=v[:n],theta=theta)
        write_json(folder/'report.json',dict(seed=seed,N=n,error=error,checkpoint_sha256=SHA,
                   edge=EDGE,timed_error=previous['error'],timed_sha256=digest(former),selected=selected,
                   candidates=candidates,sources=sources[:n],reference_sha256=digest(REF)))
        print('SPATIAL',seed,n,round(100*error,3),flush=True)

    final=fits/'N96'
    with np.load(final/'arrays.npz',allow_pickle=False) as fit:
        z,speed,re=model.m.decode(jnp.asarray(fit['theta'][:model.size]))
    profile=model.m.builder.arrays(z)
    branch=context(model.m.params,profile,re,model.m.operator.config)
    rows,arrays=[],{}
    for i,(obs,velocity,horizon,path) in enumerate(held_out(archive/'observations'),1):
        result=simulate(lambda point:np.asarray(model.cached.point(point,branch,profile,re,speed)),
                        np.linspace(0,horizon,129),obs[0],velocity,STOKES)
        row=dict(completed=result['completed'],source=str(path.relative_to(ROOT)),
                 sha256=digest(path),horizon=horizon)
        if len(result['position'])>=4:
            row.update(curve_metrics(obs,result['position'],count=129))
        rows.append(row)
        arrays[f'observed_{i}']=obs
        arrays[f'replay_{i}']=np.column_stack([result['time'],result['position']])
    np.savez_compressed(final/'replays.npz',**arrays)
    write_json(final/'replay.json',dict(timing_used_only_after_fit=True,tracks=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True,choices=SEEDS)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    run(Estimator(),args.seed)
