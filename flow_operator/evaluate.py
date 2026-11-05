"""Evaluate frozen weights against independent references; never train here."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .checkpoint import load
from .model import make_functions
from .profiles import stack
from .validation import read_cases,compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--references',required=True)
    p.add_argument('--output',required=True);p.add_argument('--acceptance',default='configs/acceptance.json')
    p.add_argument('--role',choices=['repeated_validation','final_heldout_test'],default='repeated_validation')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(a.checkpoint);sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    params,cfg,metadata,_=load(checkpoint)
    # Diagnostics in double precision from the actual trained float32 parameters.
    jax.config.update('jax_enable_x64',True)
    params=jax.tree_util.tree_map(lambda x:x.astype(np.float64),params)
    functions=make_functions(cfg);cases=read_cases(a.references)
    field_report=compare(params,functions,cases);rng=np.random.default_rng(918397)
    limits=json.loads(Path(a.acceptance).read_text())
    for case,record in zip(cases,field_report['cases']):
        data=stack([case['profile']],np.float64);re=np.array([case['re']])
        xy=rng.uniform([.001,.001],[cfg.length-.001,.999],size=(1,1024,2));xy[...,:1]=cfg.length*(xy[...,:1]/cfg.length)**2
        residual=np.asarray(functions['residual'](params,data,re,xy))[0]
        y=np.linspace(0,1,1025)
        inlet_xy=np.column_stack([np.zeros_like(y),y])[None]
        fields=np.asarray(functions['fields'](params,data,re,inlet_xy))[0]
        record.update(momentum_rms=float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1)))),
            divergence_rms=float(np.sqrt(np.mean(residual[:,2]**2))),
            boundary_max_error=float(max(np.max(abs(fields[:,0]-case['profile'].spline(y))),np.max(abs(fields[:,1])))))
        boundary=np.asarray(functions['outlet'](params,data,re,np.linspace(.001,.999,257)[None]))[0]
        record['outlet_traction_rms']=float(np.sqrt(np.mean(np.sum(boundary**2,axis=1))))
        xx,yy=np.meshgrid(np.linspace(0,cfg.length,31),np.linspace(0,1,513),indexing='ij')
        grid_xy=np.stack([xx,yy],-1).reshape(-1,2)
        parts=[np.asarray(functions['fields'](params,data,re,z[None]))[0]
               for z in np.array_split(grid_xy,int(np.ceil(len(grid_xy)/1024)))]
        grid=np.concatenate(parts).reshape(31,513,3)
        record['flux_max_error']=float(np.max(abs(np.trapezoid(grid[...,0],yy,axis=1)-1)))
        # Save reference + actual model outputs, without gauge fitting.
        pred=np.concatenate([np.asarray(functions['fields'](params,data,re,z[None]))[0]
             for z in np.array_split(case['xy'],int(np.ceil(len(case['xy'])/1024)))])
        np.savez_compressed(out/(case['name']+'.npz'),xy=case['xy'],reference=case['reference'],prediction=pred,
                            inlet_y=y,inlet_requested=case['profile'].spline(y),inlet_applied=fields[:,0])
        nx,ny=len(np.unique(case['xy'][:,0])),len(np.unique(case['xy'][:,1]))
        fig,axes=plt.subplots(3,3,figsize=(15,9),layout='constrained')
        for row,label in enumerate(['u','v','p']):
            reference=case['reference'][:,row].reshape(nx,ny).T;model=pred[:,row].reshape(nx,ny).T
            low,high=min(reference.min(),model.min()),max(reference.max(),model.max())
            for col,(title,value) in enumerate([('FEM',reference),('Frozen full-profile operator',model),('model - FEM',model-reference)]):
                kwargs=dict(vmin=low,vmax=high) if col<2 else dict(vmin=-np.max(abs(value))-1e-15,vmax=np.max(abs(value))+1e-15)
                im=axes[row,col].pcolormesh(np.unique(case['xy'][:,0]),np.unique(case['xy'][:,1]),value,
                     shading='nearest',cmap='viridis' if col<2 else 'RdBu_r',**kwargs)
                axes[row,col].set(title=f'{label}: {title}',xlabel='x/H',ylabel='y/H')
                fig.colorbar(im,ax=axes[row,col],format='%.3g')
        fig.suptitle(f"{case['name']} | Re={case['re']:g} | velocity error {100*record['velocity_relative_l2']:.2f}% | pressure error {100*record['pressure_relative_l2']:.2f}%")
        fig.savefig(out/(case['name']+'.png'),dpi=140);plt.close(fig)
    thresholds={k:v for k,v in limits.items() if isinstance(v,(int,float))}
    gates={k:all(r[k]<v for r in field_report['cases']) for k,v in thresholds.items()}
    report=dict(checkpoint_sha256=sha,step=metadata['step'],role=a.role,cases=field_report['cases'],
        thresholds=thresholds,gates=gates,all_gates_passed=all(gates.values()),
        universal_accuracy_guaranteed=False,training_labels_used=metadata['CFD_response_labels_used'],
        note='Accuracy measured on the listed benchmark only; no pressure alignment or per-input training.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    # The predictor accepts only a matching, passing report next to the checkpoint.
    checkpoint.with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
