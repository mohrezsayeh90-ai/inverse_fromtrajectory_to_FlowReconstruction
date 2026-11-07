"""Presentation-specific numerical checks; frozen weights, no training or selection by hidden trials."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flow_operator.checkpoint import load
from flow_operator.profiles import Inlet,stack
from flow_operator.model import make_functions
from validation.fem import solve_reference

HERE=Path(__file__).resolve().parent;ASSETS=HERE/'assets';DATA=HERE/'data'
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11})
CP=ROOT/'models/candidates/pressure_long/checkpoint.npz'
CASES=[('beta5_re20',4,20.),('beta5_re30',4,30.),('beta5_re40',4,40.),('beta4_re30',3,30.)]


def main():
    ASSETS.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    digest=hashlib.sha256(CP.read_bytes()).hexdigest();params,cfg,meta,_=load(CP)
    params=jax.tree_util.tree_map(lambda v:v.astype(np.float64),params);fn=make_functions(cfg)
    rows=[]
    for name,power,re in CASES:
        y=np.linspace(0,1,257);profile=Inlet.from_samples(y,y**power*(1-y)**power)
        path=DATA/(name+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                xy=d['xy'];truth=d['reference'];metadata=json.loads(str(d['reference_metadata']))
        elif name=='beta5_re30':
            with np.load(ROOT/'validation/references/beta.npz',allow_pickle=False) as d:
                xy=d['xy'].copy();truth=d['reference'].copy();metadata=json.loads(str(d['metadata']))
        else:
            xx,yy=np.meshgrid(np.r_[np.linspace(.002,.3,25),np.linspace(.4,4.99,48)],np.linspace(.002,.998,97),indexing='ij')
            xy=np.stack([xx,yy],-1).reshape(-1,2);values=[];meshes=[]
            for nx,ny in [(48,32),(64,48)]:
                print(name,'FEM',nx,ny,flush=True);solver,record=solve_reference(profile,re,nx,ny)
                values.append(np.concatenate([solver(z) for z in np.array_split(xy,16)]));meshes.append(record)
            mesh=dict(velocity_relative_l2=float(np.linalg.norm((values[0]-values[1])[:,:2])/np.linalg.norm(values[1][:,:2])),
                      pressure_relative_l2=float(np.linalg.norm((values[0]-values[1])[:,2])/np.linalg.norm(values[1][:,2])))
            assert mesh['velocity_relative_l2']<.01 and mesh['pressure_relative_l2']<.02,mesh
            metadata=dict(meshes=meshes,mesh_change=mesh);truth=values[1]
        data=stack([profile],np.float64)
        pred=np.concatenate([np.asarray(fn['fields'](params,data,np.array([re]),z[None]))[0] for z in np.array_split(xy,16)])
        error=pred-truth
        row=dict(name=name,power=power,reynolds=re,checkpoint_sha256=digest,step=meta['step'],
                 velocity_relative_l2=float(np.linalg.norm(error[:,:2])/np.linalg.norm(truth[:,:2])),
                 pressure_relative_l2=float(np.linalg.norm(error[:,2])/np.linalg.norm(truth[:,2])),
                 u_relative_l2=float(np.linalg.norm(error[:,0])/np.linalg.norm(truth[:,0])),
                 v_relative_l2=float(np.linalg.norm(error[:,1])/np.linalg.norm(truth[:,1])),
                 v_rmse=float(np.sqrt(np.mean(error[:,1]**2))),mesh_change=metadata['mesh_change'])
        probe=np.random.default_rng(72107).uniform([.001,.001],[4.999,.999],(1,512,2));probe[...,0]=5*(probe[...,0]/5)**2
        residual=np.asarray(fn['residual'](params,data,np.array([re]),probe))[0]
        row['momentum_rms']=float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1))))
        np.savez_compressed(path,xy=xy,reference=truth,prediction=pred,inlet_y=y,inlet_values=profile.values,reynolds=re,reference_metadata=json.dumps(metadata))
        rows.append(row);print(json.dumps(row),flush=True)
        xx,yy=np.unique(xy[:,0]),np.unique(xy[:,1])
        for index,label in [(0,'u'),(1,'v'),(2,'p')]:
            fig,axes=plt.subplots(1,3,figsize=(14,3.15),layout='constrained')
            ref=truth[:,index].reshape(len(xx),len(yy)).T;model=pred[:,index].reshape(len(xx),len(yy)).T
            lo,hi=min(ref.min(),model.min()),max(ref.max(),model.max());lim=max(float(abs(model-ref).max()),1e-12)
            for ax,title,value,diff in zip(axes,['FEM','Frozen DeepONet','Prediction - FEM'],[ref,model,model-ref],[False,False,True]):
                im=ax.pcolormesh(xx,yy,value,shading='nearest',cmap='RdBu_r' if diff else 'viridis',
                    vmin=-lim if diff else lo,vmax=lim if diff else hi)
                ax.set(title=title,xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=ax)
            fig.savefig(ASSETS/f'{name}_{label}.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
        axes[0].plot(profile.values,y,color='#126b83',lw=2.5);axes[0].set(xlabel='Inlet u / Umean',ylabel='y/H',title=f'Beta({power+1},{power+1}), Re={re:g}')
        for target,color in [(.1,'#126b83'),(.5,'#ce6d29'),(2.,'#5e4194')]:
            ix=np.argmin(abs(xx-target));axes[1].plot(truth.reshape(len(xx),len(yy),3)[ix,:,0],yy,color=color,label=f'FEM x={xx[ix]:.2f}')
            axes[1].plot(pred.reshape(len(xx),len(yy),3)[ix,:,0],yy,'--',color=color,label='DeepONet')
        axes[1].set(xlabel='u / Umean',ylabel='y/H',title='Cross-section profiles');axes[1].legend(fontsize=7)
        fig.savefig(ASSETS/f'{name}_profiles.png',dpi=180);plt.close(fig)
    assert hashlib.sha256(CP.read_bytes()).hexdigest()==digest
    (DATA/'forward_metrics.json').write_text(json.dumps(dict(checkpoint_sha256=digest,step=meta['step'],
        all_predeclared_cases=rows,selection_rule='Field illustrations: velocity L2 <5% AND pressure L2 <10%; report all attempted cases; no universal acceptance claim.',
        training_performed=False,repeated_or_exploratory_validation=True,pressure_alignment=False),indent=2)+'\n')


if __name__=='__main__':main()
