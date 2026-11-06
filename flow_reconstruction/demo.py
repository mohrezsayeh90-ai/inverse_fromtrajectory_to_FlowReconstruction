"""Independent numerical-flow example: two particles, positions only as input."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .particles import simulate
from .inverse import FlowReconstructor,InverseConfig
from .io import read_problem,save_fit


ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/reconstruction-fem-beta')
    p.add_argument('--checkpoint',default=str(ROOT/'models/candidates/pressure_long/checkpoint.npz'))
    p.add_argument('--starts',type=int,default=3);p.add_argument('--iterations',type=int,default=250)
    p.add_argument('--profile-points',type=int,default=65);p.add_argument('--noise',type=float,default=.0002)
    p.add_argument('--launch',choices=['corners','oblique'],default='corners')
    p.add_argument('--no-preconditioner',action='store_true')
    p.add_argument('--generate-only',action='store_true');a=p.parse_args()
    if a.noise<0:p.error('Noise must be nonnegative')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'validation/references/beta.npz',allow_pickle=False) as data:
        reference={k:data[k].copy() for k in data.files}
    x,y=np.unique(reference['xy'][:,0]),np.unique(reference['xy'][:,1])
    interpolate=RegularGridInterpolator((x,y),reference['reference'].reshape(len(x),len(y),3)[:,:,:2],bounds_error=False,fill_value=None)
    from flow_operator.profiles import Inlet
    true_inlet=Inlet.from_samples(reference['inlet_y'],reference['inlet_values'])
    def velocity(q):
        if q[1]<=0 or q[1]>=1:return np.zeros(2)
        if q[0]<=0:return np.array([true_inlet.spline(q[1]),0.])
        return interpolate(np.asarray(q)[None])[0]
    rng=np.random.default_rng(71104);items=[]
    if a.launch=='corners':
        time=np.linspace(0,1.2,101);tau=.42
        launches=[('upper_left',[0.,1.],[6.,-1.2]),('lower_right',[5.,0.],[-6.,1.2])]
    else:
        time=np.linspace(0,.55,81);tau=.12
        launches=[('upper_left',[.03,.97],[6.,-4.]),('lower_right',[4.97,.03],[-6.,4.])]
    clean=[]
    for i,(name,q0,v0) in enumerate(launches):
        truth=simulate(velocity,time,q0,v0,tau)
        if not truth['completed']:raise RuntimeError('Synthetic trajectory exits before last observation')
        position=truth['position']+rng.normal(0,a.noise,truth['position'].shape);position[0]=q0
        filename=f'track_{i+1}.csv';np.savetxt(out/filename,np.column_stack([time,position]),delimiter=',',header='t,x,y',comments='')
        clean.append(truth['position'])
        items.append(dict(name=name,csv=filename,initial_position=q0,initial_velocity=v0,stokes_time=tau))
    config=InverseConfig(reference_reynolds=float(reference['reynolds']),starts=a.starts,max_iterations=a.iterations,
                         profile_points=a.profile_points,position_scale=max(.001,2*a.noise),precondition_shape=not a.no_preconditioner)
    from dataclasses import asdict
    specification=dict(units='nondimensional',tracks=items,inverse=asdict(config))
    (out/'problem.json').write_text(json.dumps(specification,indent=2)+'\n')
    np.savez_compressed(out/'synthetic_truth.npz',**reference,particle_time=time,clean_trajectories=np.stack(clean))
    (out/'truth_provenance.json').write_text(json.dumps(dict(source='independent Taylor-Hood FEM validation reference beta.npz',
        velocity_between_reference_points='linear RegularGridInterpolator',particle_solver='adaptive solve_ivp RK45',
        trajectory_noise_std=a.noise,noise_seed=71104,forward_operator_used_to_generate_truth=False,
        reference_fields_used_in_inverse_objective=False,truth_mean_speed=1.,truth_reynolds=float(reference['reynolds'])),indent=2)+'\n')
    if a.generate_only:return
    tracks,cfg,_=read_problem(out/'problem.json')
    model=FlowReconstructor(a.checkpoint,cfg,allow_unvalidated=True);result=model.fit(tracks)
    save_fit(model,result,out/'result')
    prediction=model.fields(result['theta'],reference['xy']);delta=prediction-reference['reference']
    metrics=dict(velocity_relative_l2=float(np.linalg.norm(delta[:,:2])/np.linalg.norm(reference['reference'][:,:2])),
                 pressure_relative_l2=float(np.linalg.norm(delta[:,2])/np.linalg.norm(reference['reference'][:,2])),
                 v_relative_l2=float(np.linalg.norm(delta[:,1])/np.linalg.norm(reference['reference'][:,1])),
                 mean_speed_absolute_error=abs(result['report']['mean_speed']-1),
                 comparison_role='synthetic independent FEM truth, fields not supplied to inversion',
                 reference_grid_points=len(reference['xy']),unique_recovery_claim=False)
    from flow_operator.profiles import Inlet
    true_inlet=Inlet.from_samples(reference['inlet_y'],reference['inlet_values']);yy=np.linspace(0,1,1025)
    predicted_inlet=result['inlet'].spline(yy)*result['report']['mean_speed']
    metrics['inlet_relative_l2']=float(np.linalg.norm(predicted_inlet-true_inlet.spline(yy))/np.linalg.norm(true_inlet.spline(yy)))
    np.savez_compressed(out/'result/truth_comparison.npz',xy=reference['xy'],reference=reference['reference'],prediction=prediction,
                        inlet_y=yy,inlet_truth=true_inlet.spline(yy),inlet_prediction=predicted_inlet)
    (out/'result/truth_comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,9),layout='constrained')
    for row,label in enumerate(['u','v','p']):
        ref=reference['reference'][:,row].reshape(len(x),len(y)).T;pred=prediction[:,row].reshape(len(x),len(y)).T
        lo,hi=min(ref.min(),pred.min()),max(ref.max(),pred.max())
        for col,(title,value) in enumerate([('FEM truth',ref),('inferred flow',pred),('inferred - truth',pred-ref)]):
            lim=np.max(abs(value))+1e-15
            bounds=dict(vmin=lo,vmax=hi) if col<2 else dict(vmin=-lim,vmax=lim)
            im=axes[row,col].pcolormesh(x,y,value,shading='nearest',cmap='viridis' if col<2 else 'RdBu_r',**bounds)
            axes[row,col].set(title=label+': '+title,xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=axes[row,col])
    fig.suptitle('Two-particle reconstruction against independent FEM truth; experimental forward surrogate')
    fig.savefig(out/'result/fem_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.plot(yy,true_inlet.spline(yy),label='FEM inlet truth')
    ax.plot(yy,predicted_inlet,label='Reconstructed inlet');ax.legend();ax.set(xlabel='y/H',ylabel='velocity / U_ref')
    fig.savefig(out/'result/inlet_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(trajectory_result=result['report']['trajectories'],independent_truth=metrics,
                         local_data_rank=result['report']['data_jacobian_effective_rank'],parameters=result['report']['free_parameters']),indent=2))


if __name__=='__main__':main()
