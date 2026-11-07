"""Load fitted geometry-only parameters with explicit implementation checks."""
import hashlib,json
from pathlib import Path
import numpy as np
from flow_operator.provenance import runtime_signature
from flow_reconstruction.io import inverse_signature
from flow_reconstruction.inverse import InverseConfig
from .inverse import GeometryReconstructor,GeometryConfig

def signature():
    h=hashlib.sha256();root=Path(__file__).resolve().parent
    for name in ['inverse.py','geometry.py']:
        h.update(name.encode());h.update((root/name).read_bytes())
    return dict(geometry=h.hexdigest(),forward=runtime_signature(),shared_inverse=inverse_signature())

def load_fit(path,checkpoint,problem=None):
    from .run_cases import read_geometry
    path=Path(path);data=json.loads(path.read_text())
    if data['format']!='geometry_only_inverse_v1':raise ValueError('Unknown fitted model format')
    if data.get('implementation_signature')!=signature():raise ValueError('Implementation differs or result has not been finalized')
    model=GeometryReconstructor(checkpoint,GeometryConfig(**data['config']),InverseConfig(**data['flow_config']))
    if model.forward.operator.sha256!=data['checkpoint_sha256']:raise ValueError('Forward checkpoint differs')
    curves,_=read_geometry(problem or path.parent.parent/'input/problem.json');model.prepare(curves)
    return model,np.asarray(data['theta'],float)
