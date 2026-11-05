"""Locate physics failures in frozen candidates, without optimizer or CFD targets."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flow_operator.checkpoint import load
from flow_operator.model import make_functions
from flow_operator.profiles import stack
from flow_operator.provenance import runtime_signature
from flow_operator.validation import read_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--references',default='validation/references')
    args=parser.parse_args();directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    params,config,metadata,_=load(args.checkpoint)
    params=jax.tree_util.tree_map(lambda v:v.astype(np.float64),params)
    zero=jax.tree_util.tree_map(lambda v:v*0,params)
    functions=make_functions(config)
    # Graded x resolves the inlet layer; midpoints avoid the physical corners.
    x=config.length*((np.arange(96)+.5)/96)**3;y=(np.arange(128)+.5)/128
    xx,yy=np.meshgrid(x,y,indexing='ij');xy=np.stack([xx,yy],-1).reshape(-1,2)
    report=dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                runtime_signature=runtime_signature(),step=metadata['step'],
                role='repeated_validation_diagnostic_no_parameter_updates',
                sampling='96 cubically graded x midpoints by 128 uniform y midpoints; sampling RMS, not area integral',cases=[])
    edges=[0.,.02,.1,.5,1.,config.length]
    for case in read_cases(args.references):
        profile=stack([case['profile']],np.float64);re=np.array([case['re']]);data={}
        for name,weights in [('analytic_lift_without_neural_correction',zero),('trained_candidate',params)]:
            parts=[]
            for first in range(0,len(xy),512):
                parts.append(np.asarray(functions['residual'](weights,profile,re,xy[None,first:first+512]))[0])
            data[name]=np.concatenate(parts)
        record=dict(name=case['name'],reynolds=case['re'],regions=[])
        for left,right in zip(edges[:-1],edges[1:]):
            mask=(xy[:,0]>=left)&(xy[:,0]<right)
            region=dict(x_from=left,x_to=right,points=int(mask.sum()))
            for name,value in data.items():
                momentum=np.linalg.norm(value[mask,:2],axis=1)
                region[name]=dict(momentum_rms=float(np.sqrt(np.mean(momentum**2))),
                    momentum_p99=float(np.quantile(momentum,.99)),momentum_max=float(momentum.max()),
                    divergence_rms=float(np.sqrt(np.mean(value[mask,2]**2))))
            record['regions'].append(region)
        report['cases'].append(record)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
        maxima=max(np.max(np.linalg.norm(value[:,:2],axis=1)) for value in data.values())
        for ax,(name,value) in zip(axes[:2],data.items()):
            image=ax.pcolormesh(x,y,np.log10(np.maximum(np.linalg.norm(value[:,:2],axis=1),1e-6)).reshape(len(x),len(y)).T,
                vmin=-6,vmax=np.log10(max(maxima,1e-5)),shading='nearest',cmap='magma')
            ax.set(title=name.replace('_',' '),xlabel='x/H',ylabel='y/H',xscale='log')
            fig.colorbar(image,ax=ax,label='log10 unscaled momentum residual norm')
            axes[2].plot(x,np.sqrt(np.mean(np.sum(value[:,:2].reshape(len(x),len(y),2)**2,axis=2),axis=1)),label=name.replace('_',' '))
        axes[2].axhline(.05,color='red',ls='--',label='acceptance RMS threshold')
        axes[2].set(xscale='log',yscale='log',xlabel='x/H',ylabel='RMS over y');axes[2].legend(fontsize=8)
        fig.suptitle(f"{case['name']} | Re={case['re']:g} | frozen weights, no optimization")
        fig.savefig(directory/(case['name']+'_residual.png'),dpi=150);plt.close(fig)
        np.savez_compressed(directory/(case['name']+'_residual.npz'),xy=xy,**data)
    (directory/'residual_regions.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
