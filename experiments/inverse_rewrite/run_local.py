"""Run the inlet-specific inverse on the same near-inlet prefixes as the mixed design."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
from run import CHECKPOINT_SHA, NAMES, OUT, save_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--flow',type=Path,required=True)
parser.add_argument('--seed',type=int,default=39421)
parser.add_argument('--extend',action='store_true')
parser.add_argument('--fine-grid',action='store_true')
parser.add_argument('profiles',nargs='+',choices=NAMES)
args = parser.parse_args()
sys.path.insert(0,str(args.flow.resolve()))
import jax
import numpy as np
from launches import read_pool, generate_pool
from local_inlet import estimate, predict_test
from flow_operator.api import FlowOperator
from flow_operator.model import context, point_fields
from flow_operator.profiles import Inlet, stack
from flow_reconstruction.particles import simulate
import jax.numpy as jnp

checkpoint = args.flow/'reports/experiments/soap_cosine_200k/final_checkpoint.npz'
operator = FlowOperator(checkpoint,allow_unvalidated=True)
assert operator.sha256==CHECKPOINT_SHA
velocity = jax.jit(lambda point,branch,profile,re,mean:
                   mean*point_fields(operator.params,branch,profile,re,point,operator.config)[:2])
for name in args.profiles:
    root = OUT/f'seed_{args.seed}'/name
    obs = root/'observations'
    ref = args.flow/f'results/particle_count/reference/{name}/reference.npz'
    near = read_pool(obs/'near')
    validation = read_pool(obs/'validation_near')
    test = read_pool(obs/'test_near')+read_pool(obs/'test_random')
    heights = np.linspace(0.,1.,4097)
    with np.load(ref,allow_pickle=False) as d:
        truth = Inlet.from_samples(d['inlet_y'],d['inlet_values']).spline(heights)
    # Same exact prefixes used in the mixed campaign. The last count is the
    # number of near-inlet trajectories, not the mixed pool's total count.
    counts = [2,4,10,20,31]
    if args.extend:
        extension = generate_pool(obs/'extra_near',ref,'near',64,args.seed+1279)
        near += extension
        counts += [48,64,96]
    label = 'local_extended' if args.extend else 'local'
    resolutions = (11,21,29,37)
    if args.fine_grid:
        label += '_fine'
        resolutions += (45,53)
    result = root/label
    result.mkdir(exist_ok=True)
    if (result/'summary.json').exists():
        continue
    rows = []
    for n in counts:
        folder = result/f'N{n}'
        folder.mkdir(exist_ok=True)
        p = folder/'report.json'
        if p.exists():
            import json
            rows.append(json.loads(p.read_text()))
            continue
        curve,selected,candidates = estimate(near[:n],validation,heights,resolutions)
        row = dict(N=n,error=float(np.linalg.norm(curve-truth)/np.linalg.norm(truth)),
                   selected=selected,candidates=candidates,unseen_early=predict_test(selected,test[:4]))
        np.savetxt(folder/'inlet.csv',np.column_stack([heights,truth,curve]),delimiter=',',
                   header='y,truth,estimated',comments='')
        save_json(p,row)
        rows.append(row)
        print('LOCAL',args.seed,name,n,round(100*row['error'],3),selected['knots'],selected['region'],flush=True)
    sampled = np.genfromtxt(result/f'N{counts[-1]}'/'inlet.csv',delimiter=',',names=True)
    mean = float(np.trapezoid(sampled['estimated'],sampled['y']))
    inlet = Inlet.from_samples(sampled['y'][::16],sampled['estimated'][::16])
    profiles = {k:jnp.asarray(v[0]) for k,v in stack([inlet]).items()}
    re = jnp.asarray(mean*30)
    branch = context(operator.params,profiles,re,operator.config)
    replays = []
    for i,track in enumerate(test,1):
        replay = simulate(lambda q:np.asarray(velocity(q,branch,profiles,re,mean)),track.time,track.position[0],
                          track.initial_velocity,track.stokes_time)
        obs_position = track.position[:len(replay['time'])]
        residual = replay['position']-obs_position
        replays.append(dict(test_track=i,complete=replay['completed'],samples=len(residual),
                            coordinate_rmse=float(np.sqrt(np.mean(residual**2)))))
        np.savetxt(result/f'replay_{i:02d}.csv',np.column_stack([replay['time'],replay['position'],obs_position]),
                   delimiter=',',header='t,x,y,x_observed,y_observed',comments='')
    save_json(result/'summary.json',dict(profile=name,label=label,seed=args.seed,stages=rows,
              reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),resolutions=resolutions,full_domain_replay=replays))
