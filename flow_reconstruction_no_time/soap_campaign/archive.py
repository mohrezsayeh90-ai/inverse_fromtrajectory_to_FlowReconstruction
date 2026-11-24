"""Geometry-only refits of the exact archived sample counts displayed in the thesis."""
import argparse
import csv
import json

import jax
import jax.numpy as jnp
import numpy as np

from flow_operator.profiles import Inlet
from .fit import CHECKPOINT, OUT, PROFILES, ROOT, SHA, STOKES, Estimator, digest, write_json

ADAPTIVE = ROOT/'New_Adaptive_Particle_Results'
FIXED = ROOT/'results/particle_count'


def load_rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def cases(study):
    variant = 'soap_314k' if study=='adaptive' else 'soap_314k_near_extended'
    folder = ADAPTIVE/variant
    manifest = {r['profile']:r for r in json.loads((folder/'figures_manifest.json').read_text())}
    for row in load_rows(folder/'summary.csv'):
        name=row['profile']
        counts={int(value) for value in row['display_N'].split(',')}
        if study=='near':
            historical=ADAPTIVE/'near_inlet/fits'/name
            counts.update(int(path.name[1:]) for path in historical.glob('N*') if (path/'report.json').is_file())
        for n in sorted(counts):
            path=folder/'fits'/name/f'N{n}'/'report.json'
            yield name,n,folder/'observations'/name,path,manifest[name]['observation_sha256']


def observations(folder, n, expected_hash):
    metadata=folder/'launches.json'
    record=json.loads(metadata.read_text())
    assert record['St']==STOKES
    velocities=[item['velocity'] for item in record['launches'][:n]]
    xy,sources=[],[]
    for i in range(1,n+1):
        path=folder/f'track_{i}.csv'
        sha=digest(path)
        if expected_hash is not None:
            assert sha==expected_hash[path.name],path
        xy.append(np.loadtxt(path,delimiter=',',skiprows=1,usecols=(1,2)))
        sources.append(dict(track=str(path.relative_to(ROOT)),sha256=sha))
    return xy,np.asarray(velocities),sources,metadata


def fit(model, study, selected_profiles):
    for name,n,folder,timed,expected in cases(study):
        if selected_profiles and name not in selected_profiles:
            continue
        output=OUT/'thesis_samples'/study/name/f'N{n}'
        if (output/'report.json').exists():
            continue
        q,v,sources,launch=observations(folder,n,expected)
        assert len(q)==len(v)==n
        data=model.data(q,v)
        selected,candidates=model.fit(data)
        theta=jnp.asarray(selected['theta'])
        z,speed,_=model.m.decode(theta[:model.size])
        clocks=np.asarray(model.clocks(theta,data))
        assert np.all(np.diff(clocks,axis=1)>0)
        reference=FIXED/'reference'/name/'reference.npz'
        with np.load(reference,allow_pickle=False) as raw:
            y=np.linspace(0,1,4097)
            truth=Inlet.from_samples(raw['inlet_y'],raw['inlet_values']).spline(y)
        profile=model.m.builder.inlet(z).spline(y)*float(speed)
        error=float(np.linalg.norm(profile-truth)/np.linalg.norm(truth))
        previous=json.loads(timed.read_text())
        old=previous['inlet_relative_l2']
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'arrays.npz',y=y,truth=truth,inlet=profile,clock=clocks,
                            curves=np.asarray(data['q']),launch_velocity=v)
        write_json(output/'report.json',dict(study=study,profile=name,count=n,error=error,timed_error=old,
                   selected=selected,candidates=candidates,checkpoint_sha256=SHA,
                   timed_report=str(timed.relative_to(ROOT)),timed_report_sha256=digest(timed),
                   observations=sources,launch_sha256=digest(launch) if launch else None,
                   reference_sha256=digest(reference)))
        print('THESIS',study,name,n,round(100*error,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',choices=('adaptive','near'))
    p.add_argument('profiles',nargs='*',choices=PROFILES)
    args=p.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    fit(Estimator(),args.study,args.profiles)
