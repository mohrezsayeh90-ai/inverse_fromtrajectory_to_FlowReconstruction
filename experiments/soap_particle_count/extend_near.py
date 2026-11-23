"""Extend the nine-cycle focused design without changing its first 24 tracks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from run import BASE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--flow',type=Path,required=True)
args = parser.parse_args()
sys.path.insert(0,str(args.flow.resolve()))
from flow_reconstruction.particles import simulate

baseline = BASE/'near_inlet/observations/nine30'
out = BASE/'soap_314k_near_extended/observations/nine30'
registration = out/'launches.json'
if registration.exists():
    sys.exit(0)
out.mkdir(parents=True,exist_ok=True)
record = json.loads((baseline/'launches.json').read_text())
assert len(record['launches'])==24
ref = args.flow/'results/particle_count/reference/nine30/reference.npz'
assert hashlib.sha256(ref.read_bytes()).hexdigest()==record['reference_sha256']
with np.load(ref,allow_pickle=False) as data:
    x,y = [np.unique(data['xy'][:,i]) for i in range(2)]
    field = RegularGridInterpolator((x,y),data['reference'][:,:2].reshape(len(x),len(y),2),bounds_error=True)
time = np.genfromtxt(baseline/'track_1.csv',delimiter=',',names=True)['t']
for k in range(1,25):
    for kind in ('track','clean'):
        shutil.copyfile(baseline/f'{kind}_{k}.csv',out/f'{kind}_{k}.csv')
seed = 284914
rng = np.random.default_rng(seed)
record['additional_launch_seed'] = seed
for attempt in range(128):
    k = len(record['launches'])
    if k==48:
        break
    height = .035+.93*(((k-2)*.61803398875+.5)%1)
    position = [float(rng.uniform(.005,.009)),height]
    velocity = [float(rng.uniform(.05,.4)),float(rng.uniform(-.1,.1))]
    kind = 'near-inlet-stratified-random-extension'
    try:
        path = simulate(lambda point:field(point[None])[0],time,position,velocity,record['St'])
        valid = path['completed']
    except ValueError:
        valid = False
    record['attempts'].append(dict(position=position,velocity=velocity,kind=kind,accepted=valid))
    if not valid:
        continue
    refined = simulate(lambda point:field(point[None])[0],time,position,velocity,record['St'],rtol=1e-11,atol=1e-13)
    assert refined['completed']
    noisy = path['position'] + np.random.default_rng(seed+1000+k).normal(0,record['noise_std'],path['position'].shape)
    noisy[0] = position
    assert np.all(noisy>0) and np.all(noisy<[5,1])
    record['launches'].append(dict(particle=k+1,position=position,velocity=velocity,kind=kind,
        reference_refinement_max=float(np.abs(path['position']-refined['position']).max())))
    np.savetxt(out/f'track_{k+1}.csv',np.column_stack([time,noisy]),delimiter=',',header='t,x,y',comments='')
    np.savetxt(out/f'clean_{k+1}.csv',np.column_stack([time,path['position'],path['velocity']]),delimiter=',',
               header='t,x,y,vx,vy',comments='')
assert len(record['launches'])==48
temporary = registration.with_suffix('.tmp')
temporary.write_text(json.dumps(record,indent=2)+'\n')
temporary.replace(registration)
print('Extended nine-cycle launch data to 48, first 24 preserved.',flush=True)
