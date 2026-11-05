import json
from pathlib import Path
import numpy as np
from .v2_profiles import sample,monitoring_conditions,inlet_values
from .reference import solve_channel


def prepare(output,train_cases=128):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata={'train_seed':97031,'monitor_is_training':False,'monitor_role':'validation, repeatedly inspected; not a blind test','splits':{}}
    for name,conditions in [('train',sample(np.random.default_rng(97031),train_cases)),('monitor',monitoring_conditions())]:
        if (output/f'{name}.npz').exists():
            raise ValueError(f'Refusing to overwrite existing {name} data')
        profiles,reynolds,coeff=conditions
        x,y=np.meshgrid(np.linspace(.01,4.99,49),np.linspace(.005,.995,25),indexing='ij')
        xy=np.stack([x,y],-1).reshape(-1,2)
        fields,records=[],[]
        for i,(p,re,a) in enumerate(zip(profiles,reynolds,coeff)):
            inlet=lambda y,a=a:inlet_values(a,y)
            nx,ny=(40,20) if name=='train' else (80,32)
            fn,info=solve_channel(p,re,nx=nx,ny=ny,inlet_function=inlet)
            values=fn(xy);fields.append(values)
            if name=='monitor':
                coarse,_=solve_channel(p,re,nx=56,ny=24,inlet_function=inlet)
                diff=coarse(xy)-values
                info['mesh_velocity_relative_l2']=float(np.linalg.norm(diff[:,:2])/np.linalg.norm(values[:,:2]))
                info['mesh_pressure_relative_l2']=float(np.linalg.norm(diff[:,2])/np.linalg.norm(values[:,2]))
                if info['mesh_velocity_relative_l2']>.01 or info['mesh_pressure_relative_l2']>.02:
                    raise RuntimeError(f'Monitor reference not mesh converged: {info}')
            records.append(info)
            print(f'REFERENCE {name} {i+1}/{len(profiles)} Re={re:.1f}',flush=True)
        np.savez_compressed(output/f'{name}.npz',profiles=profiles,reynolds=reynolds,coefficients=coeff,xy=xy,fields=np.array(fields))
        metadata['splits'][name]=records
    train=np.load(output/'train.npz');monitor=np.load(output/'monitor.npz')
    for a in monitor['coefficients']:
        if np.any(np.max(np.abs(train['coefficients']-a),axis=1)<1e-8):
            raise RuntimeError('A monitored inlet leaked into training')
    (output/'reference_metadata.json').write_text(json.dumps(metadata,indent=2))
