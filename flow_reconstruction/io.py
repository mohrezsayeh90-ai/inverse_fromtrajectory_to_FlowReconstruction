"""Portable trajectory inputs and reproducible inverse-fit artifacts."""
import json
import hashlib
from pathlib import Path
import numpy as np
from flow_operator.provenance import runtime_signature
from .particles import Track
from .inverse import FlowReconstructor,InverseConfig


def inverse_signature():
    h=hashlib.sha256();root=Path(__file__).resolve().parent
    for name in ['inverse.py','profile.py','particles.py']:
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()


def read_problem(path):
    path=Path(path);spec=json.loads(path.read_text());tracks=[]
    if spec.get('units','nondimensional')!='nondimensional':
        raise ValueError('Convert inputs to the documented H,U_ref nondimensional units first')
    for item in spec['tracks']:
        data=np.genfromtxt(path.parent/item['csv'],delimiter=',',names=True)
        if data.dtype.names!=('t','x','y'):raise ValueError('Trajectory CSV header must be t,x,y')
        q=np.column_stack([data['x'],data['y']])
        if 'initial_position' in item and not np.allclose(q[0],item['initial_position'],atol=1e-10,rtol=0):
            raise ValueError('First trajectory sample differs from known launch position')
        tracks.append(Track(item['name'],np.atleast_1d(data['t']),q,item['initial_velocity'],item['stokes_time']))
    if len(tracks)!=2:raise ValueError('This reconstruction interface expects exactly two particle tracks')
    return tracks,InverseConfig(**spec.get('inverse',{})),spec


def save_fit(model,result,output,nx=101,ny=65):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=result['report'];theta=result['theta'];profile=result['inlet']
    replay=dict(units='nondimensional',inverse=report['config'],tracks=[])
    for i,track in enumerate(model.tracks):
        filename=f'observed_track_{i+1}.csv'
        np.savetxt(output/filename,np.column_stack([track.time,track.position]),delimiter=',',header='t,x,y',comments='')
        replay['tracks'].append(dict(name=track.name,csv=filename,initial_position=track.position[0].tolist(),
                                    initial_velocity=track.initial_velocity.tolist(),stokes_time=track.stokes_time))
    (output/'problem.json').write_text(json.dumps(replay,indent=2)+'\n')
    np.savez_compressed(output/'candidate_parameters.npz',theta=np.stack([c['theta'] for c in result['candidates']]),
                        selected_start=report['selected_start'],valid_starts=result['valid_candidates'])
    x,y=np.linspace(0,model.operator.config.length,nx),np.linspace(0,1,ny)
    xx,yy=np.meshgrid(x,y,indexing='ij');xy=np.stack([xx,yy],-1).reshape(-1,2)
    ensemble=np.stack([model.fields(result['candidates'][i]['theta'],xy) for i in result['valid_candidates']])
    selected=ensemble[result['valid_candidates'].index(report['selected_start'])].reshape(nx,ny,3)
    ensemble_profiles=np.stack([model.builder.inlet(model.decode(c['theta'])[0]).spline(y)*c['record']['mean_speed'] for c in result['candidates'] if c['record']['valid_inlet']])
    np.savez_compressed(output/'reconstructed_flow.npz',x=x,y=y,u=selected[...,0],v=selected[...,1],p=selected[...,2],
                        candidate_fields=ensemble.reshape(-1,nx,ny,3),candidate_field_std=ensemble.std(axis=0).reshape(nx,ny,3),
                        inlet_y=y,candidate_inlets=ensemble_profiles,mean_speed=report['mean_speed'],reynolds=report['reynolds'])
    np.savetxt(output/'inlet_normalized.csv',np.column_stack([profile.y,profile.values]),delimiter=',',header='y,u',comments='')
    np.savetxt(output/'inlet_reference_units.csv',np.column_stack([profile.y,profile.values*report['mean_speed']]),delimiter=',',header='y,u',comments='')
    fitted=dict(format='stokes_trajectory_inverse_v1',checkpoint_sha256=model.operator.sha256,
                runtime_signature=runtime_signature(),inverse_runtime_signature=inverse_signature(),config=report['config'],theta=theta.tolist(),
                mean_speed=report['mean_speed'],reynolds=report['reynolds'])
    (output/'fitted_model.json').write_text(json.dumps(fitted,indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'optimization.jsonl').open('w') as f:
        for i,history in enumerate(result['histories']):
            for record in history:f.write(json.dumps(dict(start=i,**record))+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');offset=0
    for i,(track,sim) in enumerate(zip(model.tracks,result['simulations'])):
        n=len(track.time)-1;color=['#1266a8','#d4781d'][i]
        axes[0].plot(track.position[:,0],track.position[:,1],'.',color=color,label=track.name+' observed')
        axes[0].plot(sim['position'][:,0],sim['position'][:,1],'-',color=color,label=track.name+' forward ODE')
        axes[0].scatter(*track.position[0],color=color,marker='s')
        axisdata=np.column_stack([track.time[1:],track.position[1:],result['integral_prediction'][offset:offset+n]])
        np.savetxt(output/(f'track_{i+1}_integral_fit.csv'),axisdata,delimiter=',',header='t,x_observed,y_observed,x_integral,y_integral',comments='')
        np.savetxt(output/(f'track_{i+1}_forward.csv'),np.column_stack([sim['time'],sim['position'],sim['velocity']]),delimiter=',',header='t,x,y,vx,vy',comments='')
        offset+=n
    axes[0].set(xlim=(0,model.operator.config.length),ylim=(0,1),xlabel='x/H',ylabel='y/H',title='Independent particle integration');axes[0].legend(fontsize=8)
    axes[1].plot(y,profile.spline(y)*report['mean_speed'],color='#1266a8',label='Selected inferred inlet')
    axes[1].fill_between(y,ensemble_profiles.min(axis=0),ensemble_profiles.max(axis=0),alpha=.2,label='Optimizer candidate range (not confidence interval)')
    axes[1].set(xlabel='y/H',ylabel='axial velocity / U_ref',title='Regularized inlet estimate');axes[1].legend(fontsize=7)
    fig.savefig(output/'trajectories_and_inlet.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,8),layout='constrained')
    for j,(ax,label) in enumerate(zip(axes,['u/U_ref','v/U_ref','p/(rho U_ref^2)'])):
        im=ax.pcolormesh(x,y,selected[...,j].T,shading='nearest');fig.colorbar(im,ax=ax,label=label)
        for track in model.tracks:ax.plot(*track.position.T,color='white',lw=1)
        ax.set(xlabel='x/H',ylabel='y/H',title=label+' inferred from two tracks')
    fig.suptitle('Experimental inverse estimate: forward operator has not passed all validation gates')
    fig.savefig(output/'reconstructed_fields.png',dpi=160);plt.close(fig)
    return output


def load_fit(path,checkpoint,allow_unvalidated=False):
    data=json.loads(Path(path).read_text())
    if data['format']!='stokes_trajectory_inverse_v1' or data['runtime_signature']!=runtime_signature():
        raise ValueError('Saved inverse result format or forward implementation differs')
    if data.get('inverse_runtime_signature')!=inverse_signature():raise ValueError('Inverse parameter interpretation differs')
    model=FlowReconstructor(checkpoint,InverseConfig(**data['config']),allow_unvalidated)
    if model.operator.sha256!=data['checkpoint_sha256']:raise ValueError('Forward checkpoint differs from the fitted result')
    return model,np.asarray(data['theta'],float)
