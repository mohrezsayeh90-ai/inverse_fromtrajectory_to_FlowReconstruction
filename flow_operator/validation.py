"""Read-only reference comparison. Never imported into the optimizer objective."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from .profiles import Inlet,stack


def read_cases(directory,encoder_points=257):
    cases=[]
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            inlet=Inlet.from_samples(d['inlet_y'],d['inlet_values'],encoder_points)
            cases.append(dict(name=path.stem,profile=inlet,re=float(d['reynolds']),xy=d['xy'].copy(),
                              reference=d['reference'].copy(),metadata=json.loads(str(d['metadata']))))
    if not cases:raise ValueError('No reference cases found')
    return cases


def compare(params,functions,cases):
    records=[]
    dtype=np.float64 if jax.config.x64_enabled else np.float32
    for c in cases:
        profile=stack([c['profile']],dtype);predictions=[]
        for xy in np.array_split(c['xy'],int(np.ceil(len(c['xy'])/512))):
            predictions.append(np.asarray(functions['fields'](params,profile,np.array([c['re']],dtype),xy[None].astype(dtype)))[0])
        pred=np.concatenate(predictions);truth=c['reference'];error=pred-truth
        velocity=float(np.linalg.norm(error[:,:2])/np.linalg.norm(truth[:,:2]))
        pressure=float(np.linalg.norm(error[:,2])/max(np.linalg.norm(truth[:,2]),1e-12))
        records.append(dict(name=c['name'],reynolds=c['re'],velocity_relative_l2=velocity,
            u_relative_l2=float(np.linalg.norm(error[:,0])/np.linalg.norm(truth[:,0])),
            v_relative_l2=float(np.linalg.norm(error[:,1])/max(np.linalg.norm(truth[:,1]),1e-12)),
            v_rmse=float(np.sqrt(np.mean(error[:,1]**2))),pressure_relative_l2=pressure,
            role='repeated_reference_validation_excluded_from_objective'))
    return dict(role='read_only_validation_not_backpropagated',cases=records,
                max_velocity_relative_l2=max(r['velocity_relative_l2'] for r in records),
                max_pressure_relative_l2=max(r['pressure_relative_l2'] for r in records))
