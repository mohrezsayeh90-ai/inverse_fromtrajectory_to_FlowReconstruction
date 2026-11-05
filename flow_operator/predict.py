"""Fixed-weight evaluation; never optimize weights or call a reference solver."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
from .profiles import Inlet,stack
from .checkpoint import load
from .model import make_functions,context
from .provenance import runtime_signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--inlet-csv',required=True)
    p.add_argument('--re',type=float,required=True);p.add_argument('--output',default='prediction.npz')
    p.add_argument('--nx',type=int,default=151);p.add_argument('--ny',type=int,default=129)
    p.add_argument('--encoder-points',type=int,default=257);p.add_argument('--allow-unvalidated',action='store_true')
    a=p.parse_args()
    if a.re<=0 or min(a.nx,a.ny)<2:p.error('Require Re>0 and at least two points per axis')
    checkpoint=Path(a.checkpoint);sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report_path=checkpoint.with_suffix('.validation.json')
    accepted=False
    if report_path.exists():
        validation=json.loads(report_path.read_text())
        accepted=(validation.get('checkpoint_sha256')==sha and validation.get('all_gates_passed') is True
                  and validation.get('runtime_signature')==runtime_signature())
    if not accepted and not a.allow_unvalidated:
        p.error('No matching accepted validation report. Use --allow-unvalidated only for an explicitly provisional diagnostic.')
    if not accepted:
        print('PROVISIONAL: this checkpoint has not passed the registered independent benchmark.')
    profile=Inlet.from_csv(a.inlet_csv,a.encoder_points)
    params,config,metadata,_=load(checkpoint)
    functions=make_functions(config);data=stack([profile])
    x,y=np.linspace(0,config.length,a.nx),np.linspace(0,1,a.ny)
    xx,yy=np.meshgrid(x,y,indexing='ij');xy=np.stack([xx,yy],-1).reshape(-1,2)
    start=time.perf_counter();blocks=[]
    for z in np.array_split(xy,int(np.ceil(len(xy)/1024))):
        blocks.append(np.asarray(functions['fields'](params,data,np.array([a.re],np.float32),z[None].astype(np.float32)))[0])
    result=np.concatenate(blocks).reshape(len(x),len(y),3)
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite prediction')
    elapsed=time.perf_counter()-start
    check_y=np.linspace(0,1,1025);check_xy=np.stack([np.zeros_like(check_y),check_y],-1)
    inlet=np.asarray(functions['fields'](params,data,np.array([a.re],np.float32),check_xy[None].astype(np.float32)))[0]
    diag=profile.diagnostics()
    diag.update(checkpoint_sha256=sha,training_step=metadata['step'],retraining=False,
        benchmark_accepted=accepted,universal_accuracy_guaranteed=False,
        seconds_including_first_JIT=elapsed,reynolds=a.re,
        outside_trained_re_interval=not 10<=a.re<=100,
        boundary_max_error=float(np.max(abs(inlet[:,0]-profile.spline(check_y)))),
        boundary_v_max=float(np.max(abs(inlet[:,1]))),
        inference='Function evaluation with frozen weights; no optimizer, CFD, or response-bank lookup',
        pressure_reference='Unsymmetric-gradient do-nothing traction, exterior pressure zero')
    # Independently probe the physical residual; do not modify this prediction.
    test_xy=np.random.default_rng(91897).uniform([.001,.001],[config.length-.001,.999],size=(1,256,2)).astype(np.float32)
    residual=np.asarray(functions['residual'](params,data,np.array([a.re],np.float32),test_xy))[0]
    diag['momentum_rms']=float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1))))
    diag['divergence_rms']=float(np.sqrt(np.mean(residual[:,2]**2)))
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,x=x,y=y,u=result[...,0],v=result[...,1],p=result[...,2],inlet_y=check_y,
                        inlet_requested=profile.spline(check_y),inlet_applied=inlet[:,0],reynolds=a.re)
    output.with_suffix('.json').write_text(json.dumps(diag,indent=2))
    print(json.dumps(diag,indent=2))


if __name__=='__main__':main()
