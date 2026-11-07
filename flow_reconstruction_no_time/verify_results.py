"""Final preservation checks and directional derivative checks of the full loss."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import jax.numpy as jnp
from flow_reconstruction_no_time.io import load_fit,signature
HERE=Path(__file__).resolve().parent

def main():
    baseline=json.loads((HERE/'preservation_before.json').read_text())
    changed=[p for p,h in baseline['sha256'].items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    folders=sorted((HERE/'experiments').glob('*/result'));assert len(folders)==3
    for folder in folders:
        path=folder/'fitted_model.json';data=json.loads(path.read_text());data['implementation_signature']=signature()
        path.write_text(json.dumps(data,indent=2)+'\n')
        spec=json.loads((folder.parent/'input/problem.json').read_text())
        assert set(spec)=={'format','units','reference_reynolds','curves'}
        for curve in spec['curves']:
            assert set(curve)=={'name','csv','initial_position','initial_velocity','stokes_time'}
            assert (folder.parent/'input'/curve['csv']).read_text().splitlines()[0]=='x,y'
        report=json.loads((folder/'report.json').read_text());assert report['measured_timestamps_used'] is False
        assert report['timed_fit_initialization_used'] is False and report['forward_weights_unchanged']
        assert all(p['completed'] for p in report['paths'])
    # Two directions include both flow and clock coordinates at an interior
    # non-parabolic point, away from the effective_mode zero-energy switch.
    model,theta=load_fit(folders[0]/'fitted_model.json',ROOT/'models/candidates/pressure_long/checkpoint.npz')
    rng=np.random.default_rng(8991);probe=theta.copy()+rng.normal(0,.001,len(theta));value,gradient=model.value_gradient(jnp.asarray(probe));checks=[]
    for k in range(2):
        v=rng.normal(size=len(theta));v/=np.linalg.norm(v);eps=1e-5
        plus=float(model.value_gradient(jnp.asarray(probe+eps*v))[0]);minus=float(model.value_gradient(jnp.asarray(probe-eps*v))[0])
        finite=(plus-minus)/(2*eps);automatic=float(np.asarray(gradient)@v);relative=abs(finite-automatic)/max(1.,abs(finite),abs(automatic))
        assert relative<2e-5,(finite,automatic,relative)
        checks.append(dict(automatic=automatic,finite_difference=finite,relative_error=relative))
    result=dict(existing_files_verified=baseline['tracked_count'],existing_files_changed=changed,all_three_time_free_inputs_have_xy_only=True,
                full_loss_directional_gradient_checks=checks,all_replays_completed=True)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
