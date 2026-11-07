"""Declared exploratory profile sweep with unchanged PI-CFO weights and independent FEM."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from scipy.signal import find_peaks
from flow_operator.checkpoint import load
from flow_operator.model import make_functions
from flow_operator.profiles import Inlet,stack
from validation.fem import solve_reference

HERE=Path(__file__).resolve().parent;DATA=HERE/'data'
CASES=[('skew_b3', 'Asymmetric inlet', 'skew',2,30.,'1+0.6*cos(pi*y)'),
       ('skew_b4', 'Asymmetric inlet', 'skew',3,30.,'1+0.45*cos(pi*y)'),
       ('twin_b3_r20','Twin-peaked inlet','twin',2,20.,'1+0.55*cos(2*pi*y)'),
       ('twin_b3_r30','Twin-peaked inlet','twin',2,30.,'1+0.55*cos(2*pi*y)'),
       ('twin_b4','Twin-peaked inlet','twin',3,30.,'1+0.65*cos(2*pi*y)'),
       ('lobes4_r20','Three-lobed inlet','lobes',2,20.,'1+0.28*cos(8*pi*y)'),
       ('lobes4_r30','Three-lobed inlet','lobes',2,30.,'1+0.28*cos(8*pi*y)'),
       ('waves3_r20','Wavy inlet','waves',2,20.,'1+0.25*cos(6*pi*y)'),
       ('waves3_r30','Wavy inlet','waves',2,30.,'1+0.25*cos(6*pi*y)'),
       ('mixed','Asymmetric multi-scale inlet','mixed',2,30.,'1+0.18*sin(4*pi*y)+0.10*cos(6*pi*y)'),
       ('nine','Nine-cycle modulation','nine',2,55.,'1+0.18*cos(18*pi*y)'),
       ('shoulder','Off-centre shoulder','shoulder',2,30.,'1+0.65*exp(-((y-0.27)/0.09)**2)')]


def modulation(name,y):
    if name=='skew_b3':return 1+.6*np.cos(np.pi*y)
    if name=='skew_b4':return 1+.45*np.cos(np.pi*y)
    if name.startswith('twin_b3'):return 1+.55*np.cos(2*np.pi*y)
    if name=='twin_b4':return 1+.65*np.cos(2*np.pi*y)
    if name.startswith('lobes4'):return 1+.28*np.cos(8*np.pi*y)
    if name.startswith('waves3'):return 1+.25*np.cos(6*np.pi*y)
    if name=='mixed':return 1+.18*np.sin(4*np.pi*y)+.10*np.cos(6*np.pi*y)
    if name=='nine':return 1+.18*np.cos(18*np.pi*y)
    if name=='shoulder':return 1+.65*np.exp(-((y-.27)/.09)**2)
    raise ValueError(name)


def main():
    DATA.mkdir(exist_ok=True);cp=ROOT/'models/candidates/pressure_long/checkpoint.npz';digest=hashlib.sha256(cp.read_bytes()).hexdigest()
    params,cfg,meta,_=load(cp);params=jax.tree_util.tree_map(lambda v:v.astype(np.float64),params);fn=make_functions(cfg)
    xx,yy=np.meshgrid(np.r_[np.linspace(.002,.3,25),np.linspace(.4,4.99,48)],np.linspace(.002,.998,97),indexing='ij')
    xy=np.stack([xx,yy],-1).reshape(-1,2);records=[]
    def evaluate_fem(profile,re,nx,ny):
        solver,record=solve_reference(profile,re,nx,ny)
        return np.concatenate([solver(z) for z in np.array_split(xy,16)]),record
    for name,label,group,power,re,formula in CASES:
        record_file=DATA/(name+'.json')
        if record_file.exists():
            cached=json.loads(record_file.read_text())
            assert (cached['checkpoint_sha256'],cached['reynolds'],cached['power'],cached['modulation'])==(digest,re,power,formula), 'Cached case differs; move old data before recomputing'
            assert (DATA/(name+'.npz')).exists()
            records.append(cached);continue
        y=np.linspace(0,1,257);profile=Inlet.from_samples(y,y**power*(1-y)**power*modulation(name,y));data=stack([profile],np.float64)
        pred=np.concatenate([np.asarray(fn['fields'](params,data,np.array([re]),z[None]))[0] for z in np.array_split(xy,16)])
        rough=group in ['waves','lobes','mixed','nine'];ny=64 if rough else 40
        print('START',name,re,flush=True)
        coarse,mesh0=evaluate_fem(profile,re,48,ny)
        def errors(reference):
            return dict(velocity_relative_l2=float(np.linalg.norm((pred-reference)[:,:2])/np.linalg.norm(reference[:,:2])),
                        pressure_relative_l2=float(np.linalg.norm((pred-reference)[:,2])/np.linalg.norm(reference[:,2])))
        preliminary=errors(coarse);fine=coarse;meshes=[mesh0];change=None;confirmed=False
        # Refine every potentially selected case. All coarse failures are kept.
        if preliminary['velocity_relative_l2']<.06 and preliminary['pressure_relative_l2']<.12:
            fine,mesh1=evaluate_fem(profile,re,72,96 if rough else 64);meshes.append(mesh1)
            change=dict(velocity_relative_l2=float(np.linalg.norm((fine-coarse)[:,:2])/np.linalg.norm(fine[:,:2])),
                        pressure_relative_l2=float(np.linalg.norm((fine-coarse)[:,2])/np.linalg.norm(fine[:,2])))
            confirmed=change['velocity_relative_l2']<.01 and change['pressure_relative_l2']<.02
        result=errors(fine);dense=np.linspace(0,1,4097);values=profile.spline(dense)
        peaks=len(find_peaks(values,prominence=.03*values.max())[0])
        row=dict(name=name,label=label,group=group,reynolds=re,power=power,modulation=formula,profile_formula='normalize(y^power*(1-y)^power*modulation)',
                 checkpoint_sha256=digest,step=meta['step'],**result,coarse_errors=preliminary,mesh_change=change,meshes=meshes,
                 mesh_verified=confirmed,selected=bool(confirmed and result['velocity_relative_l2']<.05 and result['pressure_relative_l2']<.10),
                 prominent_peaks=peaks,profile_max=float(values.max()),profile_min=float(values.min()))
        np.savez_compressed(DATA/(name+'.npz'),xy=xy,prediction=pred,reference=fine,inlet_y=y,inlet_values=profile.values,reynolds=re)
        record_file.write_text(json.dumps(row,indent=2)+'\n');records.append(row)
        (HERE/'sweep.json').write_text(json.dumps(dict(cases=records,declared_candidates=CASES,selection_rule='Mesh-verified velocity vector L2<5%, pressure L2<10%; all attempts retained',training_performed=False),indent=2)+'\n')
        print(json.dumps(row),flush=True)
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==digest
    (HERE/'sweep.json').write_text(json.dumps(dict(cases=records,declared_candidates=CASES,selection_rule='Mesh-verified velocity vector L2<5%, pressure L2<10%; all attempts retained',training_performed=False),indent=2)+'\n')


if __name__=='__main__':main()
