"""Independent post-training evaluation of the exported fixed-weight operator."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .v2_model import load,physics,outlet
from .v2_profiles import inlet_values


def evaluate(checkpoint,data_path,output,role,training_data=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    model,state=load(checkpoint);model.double()
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    with np.load(data_path) as d:data={k:d[k].copy() for k in d.files}
    profile_disjoint=None
    if training_data:
        with np.load(training_data) as d:train_profiles=d['profiles'].astype(np.float32)
        held=data['profiles'].astype(np.float32)
        profile_disjoint=not np.any(np.all(train_profiles[:,None,:]==held[None,:,:],axis=-1))
        if not profile_disjoint:raise ValueError('Held-out profile overlaps supervised training inputs')
    prediction=[];records=[];rng=np.random.default_rng(730531)
    for i in range(len(data['profiles'])):
        p,r=t(data['profiles'][i:i+1]),t(data['reynolds'][i:i+1])
        with torch.no_grad():pred=model(p,r,t(data['xy'][None])).numpy()[0]
        truth=data['fields'][i];err=pred-truth;prediction.append(pred)
        xy=t(rng.uniform([.005,.005],[4.995,.995],(1,512,2)))
        mx,my,mass=physics(model,p,r,xy)
        on,ot=outlet(model,p,r,t(np.linspace(.001,.999,129)[None]))
        gx,gy=np.meshgrid(np.linspace(0,5,21),np.linspace(0,1,257),indexing='ij')
        with torch.no_grad():grid=model(p,r,t(np.stack([gx,gy],-1).reshape(1,-1,2))).numpy().reshape(21,257,3)
        flux=np.trapezoid(grid[...,0],gy,axis=-1)
        by=np.linspace(0,1,257)
        with torch.no_grad():inlet=model(p,r,t(np.stack([np.zeros_like(by),by],-1)[None])).numpy()[0]
        inlet_expected=inlet_values(data['coefficients'][i],by)
        records.append({'case':i,'coefficients':data['coefficients'][i].tolist(),'reynolds':float(data['reynolds'][i]),
            'velocity_relative_l2':float(np.linalg.norm(err[:,:2])/np.linalg.norm(truth[:,:2])),
            'u_relative_l2':float(np.linalg.norm(err[:,0])/np.linalg.norm(truth[:,0])),
            'v_relative_l2':float(np.linalg.norm(err[:,1])/max(np.linalg.norm(truth[:,1]),1e-12)),
            'v_rmse':float(np.sqrt(np.mean(err[:,1]**2))),
            'pressure_relative_l2':float(np.linalg.norm(err[:,2])/np.linalg.norm(truth[:,2])),
            'divergence_rms':float(mass.square().mean().sqrt().detach()),
            'momentum_rms':float((mx.square().mean()+my.square().mean()).sqrt().detach()),
            'outlet_rms':float((on.square().mean()+ot.square().mean()).sqrt().detach()),
            'flux_max_error':float(np.abs(flux-1).max()),'wall_max_velocity':float(np.abs(grid[:,[0,-1],:2]).max()),
            'inlet_max_velocity_error':float(max(np.abs(inlet[:,0]-inlet_expected).max(),np.abs(inlet[:,1]).max()))})
        print(json.dumps(records[-1]),flush=True)
    maximum=lambda key:max(c[key] for c in records)
    gates={'velocity':maximum('velocity_relative_l2')<.05,'pressure':maximum('pressure_relative_l2')<.1,
        'momentum':maximum('momentum_rms')<.05,'outlet':maximum('outlet_rms')<.1,'divergence':maximum('divergence_rms')<1e-4,
        'flux':maximum('flux_max_error')<.02,'hard_boundaries':max(maximum('wall_max_velocity'),maximum('inlet_max_velocity_error'))<1e-5}
    report={'step':state['step'],'role':role,'profile_disjoint_from_supervised_training':profile_disjoint,'dtype':'float64 inference/diagnostics from trained float32 weights',
            'config':state['config'],'cases':records,'gates':gates,'passed':all(gates.values()),
            'field_accuracy_passed':gates['velocity'] and gates['pressure']}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    pred=np.array(prediction)
    np.savez_compressed(output/'predictions.npz',prediction=pred,reference=data['fields'],xy=data['xy'],profiles=data['profiles'],reynolds=data['reynolds'])
    worst=int(np.argmax([c['pressure_relative_l2'] for c in records]));nx=len(np.unique(data['xy'][:,0]));ny=len(np.unique(data['xy'][:,1]))
    fig,axes=plt.subplots(3,3,figsize=(13,8),constrained_layout=True)
    for j,label in enumerate(['u','v','p']):
        for k,(name,field) in enumerate([('FEM',data['fields']),('fixed operator',pred),('error',pred-data['fields'])]):
            limits={}
            if k<2:limits=dict(vmin=min(data['fields'][worst,:,j].min(),pred[worst,:,j].min()),vmax=max(data['fields'][worst,:,j].max(),pred[worst,:,j].max()))
            im=axes[j,k].imshow(field[worst,:,j].reshape(nx,ny).T,origin='lower',extent=[.01,4.99,.005,.995],aspect='auto',cmap='coolwarm',**limits)
            axes[j,k].set_title(f'{label}: {name}');fig.colorbar(im,ax=axes[j,k])
    fig.suptitle(f'{role}: worst pressure case, Re={data["reynolds"][worst]:g}, case {worst}')
    fig.savefig(output/'fields.png',dpi=160);plt.close(fig)
    return report
