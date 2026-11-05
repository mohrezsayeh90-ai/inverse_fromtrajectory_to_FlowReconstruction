"""Prepare CFD ONLY for repeated validation; not for the physics-only trainer."""
import argparse,json
from pathlib import Path
import numpy as np
from flow_operator.profiles import Inlet
from .fem import solve_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--suite',choices=['validation','heldout'],default='validation');a=p.parse_args()
    out=Path(a.output or ('validation/references' if a.suite=='validation' else 'validation/heldout_references'));out.mkdir(parents=True,exist_ok=True)
    y=np.linspace(0,1,257 if a.suite=='validation' else 513)
    cases=[('nine',6*y*(1-y)*(1-.6*np.cos(18*np.pi*y)),55.,(80,108),(104,144)),
           ('beta',630*y**4*(1-y)**4,30.,(48,32),(64,48)),
           ('flat',(-np.expm1(-(y/.05)**2))*(-np.expm1(-((1-y)/.05)**2)),70.,(64,64),(80,96)),
           ('asymmetric',6*y*(1-y)*(1+.6*np.cos(np.pi*y)),100.,(48,32),(64,48))]
    if a.suite=='heldout':
        cases=[('mixed_scales_513',6*y*(1-y)*(1+.35*np.cos(18*np.pi*y)+.2*np.cos(6*np.pi*y)+.1*np.sin(10*np.pi*y)),37.,(80,108),(104,144)),
               ('skew_flat_513',(-np.expm1(-(y/.06)**2))*(-np.expm1(-((1-y)/.09)**2))*(1+.4*np.exp(-((y-.27)/.12)**2)),83.,(64,64),(80,96))]
    for name,u,re,coarse,fine in cases:
        path=out/f'{name}.npz'
        if path.exists():continue
        inlet=Inlet.from_samples(y,u)
        xx,yy=np.meshgrid(np.r_[np.linspace(.002,.3,25),np.linspace(.4,4.99,48)],np.linspace(.002,.998,97),indexing='ij')
        xy=np.stack([xx,yy],-1).reshape(-1,2)
        values=[];records=[]
        for nx,ny in [coarse,fine]:
            print(f'REFERENCE {name} {nx}x{ny}',flush=True)
            fn,metadata=solve_reference(inlet,re,nx,ny)
            values.append(np.concatenate([fn(v) for v in np.array_split(xy,int(np.ceil(len(xy)/512)))]));records.append(metadata)
        error=values[0]-values[1]
        check=dict(velocity_relative_l2=float(np.linalg.norm(error[:,:2])/np.linalg.norm(values[1][:,:2])),
                   pressure_relative_l2=float(np.linalg.norm(error[:,2])/np.linalg.norm(values[1][:,2])))
        if check['velocity_relative_l2']>.01 or check['pressure_relative_l2']>.02:raise RuntimeError(f'Reference mesh refinement failed: {name} {check}')
        metadata=dict(meshes=records,mesh_change=check,role='repeated_validation_only_not_a_blind_test' if a.suite=='validation' else 'reserved_final_heldout_test',
                      input_representation='same normalized complete spline as neural operator',pressure_alignment=False)
        np.savez_compressed(path,inlet_y=y,inlet_values=inlet.values,reynolds=re,xy=xy,reference=values[1],metadata=json.dumps(metadata))
        print(json.dumps(dict(name=name,**metadata)),flush=True)


if __name__=='__main__':main()
