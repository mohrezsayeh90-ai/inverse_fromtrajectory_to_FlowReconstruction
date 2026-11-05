"""Inference with fixed v2 weights, never per-inlet retraining or CFD solving."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .v2_model import load,physics
from .v2_profiles import inlet_values,PROJECT,A,INTEGRALS
from .profiles import load_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--re',required=True,type=float)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--coefficients',type=float,nargs=4);g.add_argument('--inlet-csv')
    p.add_argument('--output',default='prediction_v2.npz');p.add_argument('--nx',type=int,default=201);p.add_argument('--ny',type=int,default=81)
    a=p.parse_args();torch.set_num_threads(4)
    if not 10<=a.re<=100:p.error('Reynolds must be within [10,100]')
    if a.coefficients:
        if sum(abs(v) for v in a.coefficients)>.800001:p.error('Coefficient L1 radius exceeds trained interval 0.8')
        profile=inlet_values(np.array(a.coefficients))
    else:
        supplied=load_csv(a.inlet_csv);b=PROJECT@supplied;b/=b[0]+b[1:]@INTEGRALS
        profile=A@b
        mismatch=np.linalg.norm(profile-supplied)/np.linalg.norm(supplied)
        if mismatch>.001:p.error(f'Input is outside the four-mode inlet family (fit error {mismatch:.2%}); boundary will not be silently changed')
        if np.abs(b[1:]/b[0]).sum()>.800001:p.error('CSV lies outside the trained amplitude domain')
    model,state=load(a.checkpoint);model.double()
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    x,y=np.meshgrid(np.linspace(0,5,a.nx),np.linspace(0,1,a.ny),indexing='ij');xy=np.stack([x,y],-1)
    with torch.no_grad():f=model(t(profile[None]),t([a.re]),t(xy.reshape(1,-1,2))).numpy().reshape(a.nx,a.ny,3)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,x=x,y=y,u=f[...,0],v=f[...,1],p=f[...,2],inlet=profile,reynolds=a.re)
    coords=np.random.default_rng(74681).uniform([.005,.005],[4.995,.995],(1,512,2))
    mx,my,mass=physics(model,t(profile[None]),t([a.re]),t(coords))
    flux=np.trapezoid(f[...,0],y,axis=-1)
    report={'step':state['step'],'retraining':False,'reynolds':a.re,'output':str(output),
        'divergence_rms':float(mass.square().mean().sqrt().detach()),'momentum_rms':float((mx.square().mean()+my.square().mean()).sqrt().detach()),
        'flux_max_error':float(np.abs(flux-1).max()),'scope':'four smooth cosine modes, L1<=0.8, Re in [10,100]; fixed trained operator'}
    validation=Path(a.checkpoint).parent/'validation_summary.json'
    if validation.exists():
        checks=json.loads(validation.read_text())
        report['published_validation']=checks
        if not checks.get('all_gates_passed'):
            print('WARNING: field tests passed but the strict momentum-residual target remains unmet; see the model report')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
