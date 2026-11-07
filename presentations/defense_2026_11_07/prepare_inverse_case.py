"""Second inlet family for the presentation; independent numerical trajectory generation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import asdict
from flow_operator.profiles import Inlet
from flow_reconstruction.particles import simulate,Track
from flow_reconstruction.inverse import FlowReconstructor,InverseConfig
from flow_reconstruction.io import save_fit


def main():
    here=Path(__file__).resolve().parent;out=here/'data/inverse_beta4';out.mkdir(exist_ok=True)
    with np.load(here/'data/beta4_re30.npz',allow_pickle=False) as d:raw={k:d[k].copy() for k in d.files}
    x,y=np.unique(raw['xy'][:,0]),np.unique(raw['xy'][:,1])
    interp=RegularGridInterpolator((x,y),raw['reference'].reshape(len(x),len(y),3)[:,:,:2],bounds_error=False,fill_value=None)
    inlet=Inlet.from_samples(raw['inlet_y'],raw['inlet_values'])
    def velocity(q):
        if q[1]<=0 or q[1]>=1:return np.zeros(2)
        if q[0]<=0:return np.array([inlet.spline(q[1]),0.])
        return interp(q[None])[0]
    time=np.linspace(0,1.2,101);rng=np.random.default_rng(72109);tracks=[];items=[]
    for i,(name,q,v) in enumerate([('upper_left',[0.,1.],[6.,-1.2]),('lower_right',[5.,0.],[-6.,1.2])],1):
        truth=simulate(velocity,time,q,v,.42);assert truth['completed']
        positions=truth['position']+rng.normal(0,.0002,truth['position'].shape);positions[0]=q
        track=Track(name,time,positions,v,.42);tracks.append(track)
        np.savetxt(out/f'track_{i}.csv',np.column_stack([time,positions]),delimiter=',',header='t,x,y',comments='')
        items.append(dict(name=name,csv=f'track_{i}.csv',initial_position=q,initial_velocity=v,stokes_time=.42))
    cfg=InverseConfig(reference_reynolds=30.,starts=2,max_iterations=400)
    (out/'problem.json').write_text(json.dumps(dict(units='nondimensional',tracks=items,inverse=asdict(cfg)),indent=2))
    model=FlowReconstructor(ROOT/'models/candidates/pressure_long/checkpoint.npz',cfg,allow_unvalidated=True)
    result=model.fit(tracks);save_fit(model,result,out/'result')
    pred=model.fields(result['theta'],raw['xy']);delta=pred-raw['reference'];yy=np.linspace(0,1,1025)
    report=dict(generator='independent FEM, Beta(4,4), Re=30',noise_std=.0002,noise_seed=72109,
        reference_fields_used_in_inverse_loss=False,velocity_relative_l2=float(np.linalg.norm(delta[:,:2])/np.linalg.norm(raw['reference'][:,:2])),
        inlet_relative_l2=float(np.linalg.norm(result['inlet'].spline(yy)*result['report']['mean_speed']-inlet.spline(yy))/np.linalg.norm(inlet.spline(yy))),
        pressure_relative_l2=float(np.linalg.norm(delta[:,2])/np.linalg.norm(raw['reference'][:,2])))
    (out/'truth_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(result['report']['trajectories']),flush=True);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
