"""Fit a regularized inlet and flow to two timestamped inertial-particle paths."""
import argparse,json
from pathlib import Path
from .inverse import FlowReconstructor
from .io import read_problem,save_fit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem',required=True)
    p.add_argument('--checkpoint',default=str(Path(__file__).resolve().parents[1]/'models/candidates/pressure_long/checkpoint.npz'))
    p.add_argument('--output',default='runs/flow-reconstruction');p.add_argument('--allow-unvalidated',action='store_true')
    a=p.parse_args();tracks,config,spec=read_problem(a.problem)
    model=FlowReconstructor(a.checkpoint,config,a.allow_unvalidated);result=model.fit(tracks)
    output=save_fit(model,result,a.output)
    (output/'problem_specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    # Preserve observations beside the fitted model for independent replay.
    import numpy as np
    for i,track in enumerate(tracks):
        np.savetxt(output/(f'observed_track_{i+1}.csv'),np.column_stack([track.time,track.position]),delimiter=',',header='t,x,y',comments='')
    print(json.dumps(result['report'],indent=2))


if __name__=='__main__':main()
