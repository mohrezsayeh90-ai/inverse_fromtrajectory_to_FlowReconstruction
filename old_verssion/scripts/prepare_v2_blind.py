"""Predetermined stronger cases, never mounted into the training notebooks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from channel_operator.v2_profiles import inlet_values
from channel_operator.reference import solve_channel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'blind.npz').exists():raise ValueError('Blind data already exists; refusing replacement')
    seed=7115903;rng=np.random.default_rng(seed)
    coeff=np.zeros((8,4));coeff[np.arange(4),np.arange(4)]=[.73,-.76,.74,.78]
    mixed=rng.normal(size=(4,4));mixed*=rng.uniform(.72,.79,(4,1))/np.abs(mixed).sum(1,keepdims=True)
    coeff[4:]=mixed;res=np.array([100.,100.,10.,100.,10.,100.,30.,90.])
    profiles=inlet_values(coeff)
    x,y=np.meshgrid(np.linspace(.01,4.99,49),np.linspace(.005,.995,25),indexing='ij')
    xy=np.stack([x,y],-1).reshape(-1,2);fields=[];records=[]
    for i,(c,re) in enumerate(zip(coeff,res)):
        inlet=lambda y,c=c:inlet_values(c,y)
        fine,info=solve_channel(profiles[i],re,nx=96,ny=36,inlet_function=inlet)
        coarse,_=solve_channel(profiles[i],re,nx=64,ny=28,inlet_function=inlet)
        truth=fine(xy);difference=coarse(xy)-truth;fields.append(truth)
        info['mesh_velocity_relative_l2']=float(np.linalg.norm(difference[:,:2])/np.linalg.norm(truth[:,:2]))
        info['mesh_pressure_relative_l2']=float(np.linalg.norm(difference[:,2])/np.linalg.norm(truth[:,2]))
        if info['mesh_velocity_relative_l2']>.01 or info['mesh_pressure_relative_l2']>.02:raise RuntimeError(f'Blind reference mesh not converged: {info}')
        records.append(info);print(f'BLIND CFD {i+1}/8 Re={re}',flush=True)
    np.savez_compressed(output/'blind.npz',profiles=profiles,coefficients=coeff,reynolds=res,xy=xy,fields=np.array(fields))
    (output/'blind_metadata.json').write_text(json.dumps({'seed':seed,'cases':records,'role':'final blind evaluation after checkpoint selection','uploaded_to_training_notebooks':False},indent=2))


if __name__=='__main__':main()
