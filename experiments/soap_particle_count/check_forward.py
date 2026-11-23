"""Compare frozen forward weights to the same saved FEM fields used for the tracks."""
import argparse
import csv
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('JAX_ENABLE_X64','true')
sys.dont_write_bytecode=True

from run import BASE, CHECKPOINT, NAMES, SHA

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--flow',type=Path,required=True)
parser.add_argument('--cpu',type=int)
args=parser.parse_args()
if args.cpu is not None:
    os.sched_setaffinity(0,{args.cpu})
sys.path.insert(0,str(args.flow.resolve()))

import numpy as np
from flow_operator.api import FlowOperator
from flow_operator.profiles import Inlet

old=FlowOperator(args.flow/'models/candidates/pressure_long/checkpoint.npz',allow_unvalidated=True)
new=FlowOperator(args.flow/CHECKPOINT,allow_unvalidated=True)
assert new.sha256==SHA
rows=[]
for name in NAMES:
    path=args.flow/f'results/particle_count/reference/{name}/reference.npz'
    with np.load(path,allow_pickle=False) as data:
        inlet=Inlet.from_samples(data['inlet_y'],data['inlet_values'])
        truth=data['reference']
        errors=[]
        for operator in [old,new]:
            prediction=operator.evaluate(inlet,30.,data['xy'])
            diff=prediction-truth
            errors.append([100*np.linalg.norm(diff[:,sl])/np.linalg.norm(truth[:,sl]) for sl in [slice(0,2),2]])
    rows.append(dict(profile=name,points=len(truth),old_velocity_l2_percent=errors[0][0],
                     soap_velocity_l2_percent=errors[1][0],old_pressure_l2_percent=errors[0][1],
                     soap_pressure_l2_percent=errors[1][1]))
with (BASE/'soap_314k/forward_comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
