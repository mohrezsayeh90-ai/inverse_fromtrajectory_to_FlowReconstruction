"""Generate nested synthetic trajectories from archived FEM, independently of SOAP."""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from flow_reconstruction.particles import Track, simulate

TIME = 1.2*np.linspace(0.,1.,81)**1.5
STOKES = .42
NOISE = 2e-4


def read_pool(directory):
    info = json.loads((directory/'launches.json').read_text())
    return [Track(f'p{k}', data['t'], np.column_stack([data['x'],data['y']]), launch['velocity'], STOKES)
            for k,launch in enumerate(info['launches'],1)
            for data in [np.genfromtxt(directory/f'track_{k}.csv',delimiter=',',names=True)]]


def generate_pool(directory, field_path, kind, count, seed):
    if (directory/'launches.json').exists():
        tracks = read_pool(directory)
        if len(tracks)!=count:
            raise ValueError(f'{directory}: cached count differs')
        return tracks
    directory.mkdir(parents=True,exist_ok=True)
    with np.load(field_path,allow_pickle=False) as saved:
        x,y = [np.unique(saved['xy'][:,i]) for i in (0,1)]
        field = RegularGridInterpolator((x,y),saved['reference'][:,:2].reshape(len(x),len(y),2),bounds_error=True)
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed+10_000)
    phase = rng.uniform()
    launches, attempts = [], []
    tracks = []
    for attempt in range(1500):
        if len(tracks)==count:
            break
        if kind=='near':
            height = .025+.95*((phase+len(tracks)*.61803398875)%1)
            q = [float(rng.uniform(.004,.008)),float(height)]
            v = [float(rng.uniform(.025,.25)),float(rng.uniform(-.08,.08))]
        else:
            q = rng.uniform([.04,.045],[4.75,.955]).tolist()
            speed,angle = rng.uniform(.15,3.5),rng.uniform(-np.pi,np.pi)
            v = (speed*np.array([np.cos(angle),np.sin(angle)])).tolist()
        candidate = dict(position=q,velocity=v,design=kind,attempt=attempt)
        try:
            path = simulate(lambda point:field(point[None])[0],TIME,q,v,STOKES)
            accepted = path['completed']
        except ValueError:
            accepted = False
        attempts.append({**candidate,'accepted':accepted})
        if not accepted:
            continue
        position = path['position']+noise_rng.normal(0,NOISE,path['position'].shape)
        position[0] = q
        if not (np.all(position>0) and np.all(position<[5,1])):
            attempts[-1]['accepted'] = False
            continue
        k = len(tracks)+1
        np.savetxt(directory/f'track_{k}.csv',np.column_stack([TIME,position]),delimiter=',',header='t,x,y',comments='')
        np.savetxt(directory/f'clean_{k}.csv',np.column_stack([TIME,path['position'],path['velocity']]),
                   delimiter=',',header='t,x,y,vx,vy',comments='')
        tracks.append(Track(f'{kind}{k}',TIME,position,v,STOKES))
        launches.append(candidate)
    if len(tracks)!=count:
        raise RuntimeError(f'{directory}: insufficient trajectories')
    tmp = directory/'launches.tmp'
    tmp.write_text(json.dumps(dict(kind=kind,seed=seed,noise_std=NOISE,launches=launches,attempts=attempts),indent=2)+'\n')
    tmp.replace(directory/'launches.json')
    return tracks


def hybrid(random, near):
    # Both designs start with the same two random tracks. After that,
    # two near-inlet tracks are added per additional random track.
    result = list(random[:2])
    random_index = 2
    near_index = 0
    for i in range(2,len(random)):
        if (i-2)%3==2:
            result.append(random[random_index])
            random_index+=1
        else:
            result.append(near[near_index])
            near_index+=1
    return result
