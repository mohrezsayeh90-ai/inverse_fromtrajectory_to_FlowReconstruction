"""Cover conditional corners systematically, without training at monitor inlets."""
import argparse,json,sys,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from channel_operator.v2_profiles import inlet_values
from channel_operator.reference import solve_channel


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=Path(a.data);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'train.npz').exists():raise ValueError('Refusing to overwrite training data')
    with np.load(data/'train.npz') as d:original={k:d[k] for k in d.files}
    with np.load(data/'monitor.npz') as d:monitor=d['coefficients'].copy()
    coeff,res=[],[]
    for mode in range(4):
        for amplitude in [-.8,-.7,-.5,-.3,.3,.5,.7,.8]:
            for re in [10.,30.,60.,100.]:
                c=np.zeros(4);c[mode]=amplitude;coeff.append(c);res.append(re)
    rng=np.random.default_rng(701113)
    mixed=rng.normal(size=(32,4));mixed*=rng.uniform(.55,.8,(32,1))/np.abs(mixed).sum(1,keepdims=True)
    coeff.extend(mixed);res.extend(np.tile([10.,100.],16))
    coeff=np.array(coeff);res=np.array(res)
    if np.min(np.abs(coeff[:,None]-monitor[None]).max(-1))<1e-8:raise ValueError('Monitor leakage')
    profiles=inlet_values(coeff);fields=[];records=[]
    for i,(c,re) in enumerate(zip(coeff,res)):
        inlet=lambda y,c=c:inlet_values(c,y)
        fn,info=solve_channel(profiles[i],re,nx=40,ny=20,inlet_function=inlet)
        fields.append(fn(original['xy']));records.append(info)
        print(f'DOE TRAIN reference {i+1}/{len(coeff)} Re={re:.0f}',flush=True)
    updated={'xy':original['xy']}
    for key,value in [('profiles',profiles),('coefficients',coeff),('reynolds',res),('fields',np.array(fields))]:
        updated[key]=np.concatenate([original[key],value]).astype(np.float32)
    np.savez_compressed(output/'train.npz',**updated)
    for name in ['monitor.npz','reference_metadata.json','packaging.json']:
        shutil.copy2(data/name,output/name)
    (output/'doe_metadata.json').write_text(json.dumps({'added_cases':len(coeff),'total_cases':len(updated['profiles']),
        'axis_amplitudes':[-.8,-.7,-.5,-.3,.3,.5,.7,.8],'axis_reynolds':[10,30,60,100],
        'monitor_amplitudes_excluded':[-.6,.6],'monitor_unchanged_sha256':hashlib.sha256((output/'monitor.npz').read_bytes()).hexdigest(),
        'reference':records},indent=2))


if __name__=='__main__':main()
