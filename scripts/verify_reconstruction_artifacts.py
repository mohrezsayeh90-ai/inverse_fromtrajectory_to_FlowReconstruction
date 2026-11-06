"""Verify complete inverse artifacts and replay each saved flow without fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from flow_reconstruction.io import load_fit,read_problem

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'flow_reconstruction';manifest=json.loads((base/'ARTIFACT_MANIFEST.json').read_text())
    for name,entry in manifest.items():
        path=base/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],name
    results=[]
    for path in sorted((base/'examples').rglob('result/fitted_model.json')):
        model,theta=load_fit(path,ROOT/'models/candidates/pressure_long/checkpoint.npz',allow_unvalidated=True)
        tracks,config,_=read_problem(path.parent/'problem.json');assert len(tracks)==2
        with np.load(path.with_name('reconstructed_flow.npz'),allow_pickle=False) as data:
            ix=np.array([2,17,45,70,98]);iy=np.array([2,15,30,45,60]);xy=np.column_stack([data['x'][ix],data['y'][iy]])
            expected=np.stack([data[k][ix,iy] for k in ['u','v','p']],axis=1)
            error=float(np.max(abs(model.fields(theta,xy)-expected)));assert error<1e-9,error
        report=json.loads(path.with_name('report.json').read_text());assert report['forward_weights_unchanged']
        results.append(dict(example=str(path.parent.relative_to(base)),field_replay_max_error=error,
                            trajectory_consistency_passed=report['trajectory_consistency_passed'],
                            forward_operator_benchmark_accepted=report['forward_operator_benchmark_accepted']))
    assert len(results)>=4
    print(json.dumps(dict(artifact_hashes_verified=len(manifest),saved_fits_replayed=results,
                         global_flow_accuracy_guaranteed=False),indent=2))


if __name__=='__main__':main()
