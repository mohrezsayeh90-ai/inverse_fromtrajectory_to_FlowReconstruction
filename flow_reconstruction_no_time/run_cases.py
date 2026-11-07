"""Export x,y-only inputs once, then fit without reading timestamped sources."""
import argparse,hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flow_reconstruction.inverse import InverseConfig
from flow_reconstruction_no_time.geometry import Curve,resample_curve
from flow_reconstruction_no_time.inverse import GeometryReconstructor,GeometryConfig

HERE=Path(__file__).resolve().parent
SOURCES={'corners_beta4':'presentations/defense_2026_11_07/data/inverse_beta4/result',
         'corners_beta5':'flow_reconstruction/examples/corner_launch_preconditioned/result',
         'oblique_beta5':'flow_reconstruction/examples/fem_beta/result'}

def prepare():
    for name,path in SOURCES.items():
        source=ROOT/path;dest=HERE/'experiments'/name/'input';dest.mkdir(parents=True,exist_ok=True)
        old=json.loads((source/'problem.json').read_text());spec=dict(format='ordered_xy_geometry_v1',units='nondimensional',reference_reynolds=30.,curves=[])
        for i,item in enumerate(old['tracks']):
            d=np.genfromtxt(source/item['csv'],delimiter=',',names=True)
            # This export is the sole contact with old observations; discard t
            # and erase their nonuniform density before the optimizer sees them.
            q=np.column_stack([d['x'],d['y']]);xy,_=resample_curve(q,257)
            filename=f'curve_{i+1}.csv'
            np.savetxt(dest/filename,xy,delimiter=',',header='x,y',comments='')
            spec['curves'].append(dict(name=item['name'],csv=filename,initial_position=xy[0].tolist(),initial_velocity=item['initial_velocity'],stokes_time=item['stokes_time']))
        (dest/'problem.json').write_text(json.dumps(spec,indent=2)+'\n')
    (HERE/'comparison_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')

def read_geometry(path):
    path=Path(path);spec=json.loads(path.read_text())
    if set(spec)!={'format','units','reference_reynolds','curves'} or spec['format']!='ordered_xy_geometry_v1':raise ValueError('Unexpected metadata; no timing fields accepted')
    if spec['units']!='nondimensional':raise ValueError('Convert physical units first')
    curves=[Curve.read(path.parent/c['csv'],c['name'],c['initial_velocity'],c['stokes_time']) for c in spec['curves']]
    for curve,item in zip(curves,spec['curves']):
        if set(item)!={'name','csv','initial_position','initial_velocity','stokes_time'}:raise ValueError('Unexpected curve metadata; timing fields forbidden')
        if not np.allclose(curve.position[0],item['initial_position'],rtol=0,atol=1e-12):raise ValueError('Launch position differs from the curve start')
    return curves,spec

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',choices=list(SOURCES));p.add_argument('--iterations',type=int,default=700);p.add_argument('--starts',type=int,default=2)
    args=p.parse_args()
    if args.prepare:prepare();return
    for name in ([args.case] if args.case else SOURCES):
        output=HERE/'experiments'/name/'result'
        if (output/'report.json').exists():print('Already saved',name,flush=True);continue
        curves,spec=read_geometry(HERE/'experiments'/name/'input/problem.json')
        print('START',name,flush=True)
        model=GeometryReconstructor(ROOT/'models/candidates/pressure_long/checkpoint.npz',GeometryConfig(starts=args.starts,max_iterations=args.iterations),InverseConfig(reference_reynolds=spec['reference_reynolds']))
        report=model.fit(curves,output);print('FINISHED',name,json.dumps(report),flush=True)

if __name__=='__main__':main()
