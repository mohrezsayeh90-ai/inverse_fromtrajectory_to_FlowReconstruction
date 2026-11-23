"""Repeat the SOAP inverse study with randomized and mixed near-inlet release pools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['JAX_ENABLE_X64']='true'
sys.dont_write_bytecode=True

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'New_Adaptive_Particle_Results/inverse_rewrite'
ARCHIVE = OUT.parent
NAMES = ['beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed','waves3_r30','lobes4_r30','nine30']
COUNTS = {'recorded':(2,4,6,9,14,24), 'random':(2,4,8,16,32,48), 'hybrid':(2,4,8,16,32,48)}
CHECKPOINT_SHA = 'e46687f8d50818329072eca1af7c70136fb9e3d314207f89e9427ccc140dfbf9'


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def run(args):
    if args.cpu is not None:
        os.sched_setaffinity(0,{args.cpu})
    sys.path.insert(0,str(args.flow.resolve()))
    sys.path.insert(0,str(ROOT/'experiments/soap_particle_count'))
    import jax
    import numpy as np
    from flow_operator.profiles import Inlet
    from launches import generate_pool, read_pool, hybrid
    from estimator import Estimator

    checkpoint = args.flow/'reports/experiments/soap_cosine_200k/final_checkpoint.npz'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA
    model = Estimator(checkpoint)
    seed = args.seed
    for name in args.profiles:
        destination = OUT/f'seed_{seed}'/name
        destination.mkdir(parents=True,exist_ok=True)
        reference = args.flow/f'results/particle_count/reference/{name}/reference.npz'
        with np.load(reference,allow_pickle=False) as data:
            target = Inlet.from_samples(data['inlet_y'],data['inlet_values'])
        yy = np.linspace(0.,1.,4097)
        truth = target.spline(yy)
        observations = destination/'observations'
        random = generate_pool(observations/'random',reference,'random',48,seed)
        near = generate_pool(observations/'near',reference,'near',32,seed+119)
        validation = (generate_pool(observations/'validation_random',reference,'random',8,seed+257)
                      +generate_pool(observations/'validation_near',reference,'near',8,seed+389))
        validation_data = model.data(validation)
        test = (generate_pool(observations/'test_random',reference,'random',4,seed+681)
                +generate_pool(observations/'test_near',reference,'near',4,seed+863))
        pools = {'recorded':lambda:read_pool(ARCHIVE/'observations'/name),
                 'random':lambda:random,'hybrid':lambda:hybrid(random,near)}
        for design in args.designs:
            folder = destination/design
            folder.mkdir(exist_ok=True)
            if (folder/'summary.json').exists():
                continue
            tracks = pools[design]()
            all_data = model.data(tracks)
            previous = None
            rows = []
            for n in COUNTS[design]:
                fit = folder/f'N{n}'
                fit.mkdir(exist_ok=True)
                report = fit/'report.json'
                if report.exists():
                    row = json.loads(report.read_text())
                else:
                    data = {key:value[:n] for key,value in all_data.items()}
                    selected,candidates = model.solve(data,validation_data,previous)
                    predicted = model.inlet(selected['theta'],yy)
                    error = float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth))
                    np.savetxt(fit/'inlet.csv',np.column_stack([yy,truth,predicted]),delimiter=',',
                               header='y,truth,estimated',comments='')
                    row = dict(N=n,error=error,selected=selected,candidates=candidates)
                    save_json(report,row)
                    print('FIT',seed,name,design,n,round(100*error,3),selected['label'],flush=True)
                previous = np.array(row['selected']['theta'])
                rows.append(row)
            replays = model.replays(previous, tracks+test)
            replay_rows = []
            for k,(track,replay) in enumerate(zip(tracks+test,replays),1):
                expected = track.position[:len(replay['time'])]
                subset = 'train' if k<=len(tracks) else 'test'
                residual = replay['position']-expected
                metrics = dict(index=k,subset=subset,complete=bool(replay['completed']),
                               sample_count=len(residual),coordinate_rmse=float(np.sqrt(np.mean(residual**2))))
                replay_rows.append(metrics)
                np.savetxt(folder/f'replay_{k:02d}.csv',
                           np.column_stack([replay['time'],replay['position'],expected]),
                           delimiter=',',header='t,x,y,x_observed,y_observed',comments='')
            save_json(folder/'summary.json',dict(profile=name,design=design,seed=seed,jax=jax.__version__,
                checkpoint=CHECKPOINT_SHA,reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                stages=rows,final_replay=replay_rows))
            print('COMPLETE',seed,name,design,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=39421)
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--designs',nargs='+',choices=list(COUNTS),default=list(COUNTS))
    parser.add_argument('profiles',nargs='+',choices=NAMES)
    run(parser.parse_args())
