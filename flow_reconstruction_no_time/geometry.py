"""Ordered curves and genuinely time-free comparison metrics."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def resample_curve(position,count=129):
    if not isinstance(count,(int,np.integer)) or count<4:raise ValueError('Use at least four resampling points')
    q=np.asarray(position,float)
    if q.ndim!=2 or q.shape[1]!=2 or len(q)<4 or not np.isfinite(q).all():
        raise ValueError('Supply at least four finite ordered (x,y) points')
    lengths=np.linalg.norm(np.diff(q,axis=0),axis=1)
    q=q[np.r_[True,lengths>1e-12]]
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if len(q)<4 or s[-1]<1e-8:raise ValueError('Degenerate curve')
    target=np.linspace(0,s[-1],count)
    return np.column_stack([np.interp(target,s,q[:,j]) for j in range(2)]),target

@dataclass
class Curve:
    name: str
    position: np.ndarray
    initial_velocity: np.ndarray
    stokes_time: float

    def __post_init__(self):
        self.position=np.asarray(self.position,float);self.initial_velocity=np.asarray(self.initial_velocity,float)
        resample_curve(self.position)
        if self.initial_velocity.shape!=(2,) or not np.isfinite(self.initial_velocity).all() or np.linalg.norm(self.initial_velocity)<1e-8:
            raise ValueError('A known nonzero launch velocity is required')
        if not np.isfinite(self.stokes_time) or self.stokes_time<=0:raise ValueError('Known positive Stokes time is required')
        if np.any(self.position<0) or np.any(self.position[:,0]>5) or np.any(self.position[:,1]>1):raise ValueError('Curve outside channel')
        for axis,bound,sign in [(0,0.,1),(0,5.,-1),(1,0.,1),(1,1.,-1)]:
            if self.position[0,axis]==bound and sign*self.initial_velocity[axis]<=0:
                raise ValueError('A boundary launch must point into the channel')

    @classmethod
    def read(cls,path,name,initial_velocity,stokes_time):
        d=np.genfromtxt(Path(path),delimiter=',',names=True)
        if d.dtype.names!=('x','y'):raise ValueError('Time-free CSV must contain exactly x,y; t is forbidden')
        return cls(name,np.column_stack([d['x'],d['y']]),initial_velocity,stokes_time)

def point_polyline_distance(points,line):
    p=np.asarray(points);line=np.asarray(line);a=line[:-1];v=np.diff(line,axis=0)
    den=np.maximum(np.sum(v*v,axis=1),1e-30)
    relative=p[:,None,:]-a[None,:,:]
    alpha=np.clip(np.sum(relative*v[None,:,:],axis=-1)/den,0,1)
    distance=np.linalg.norm(relative-alpha[...,None]*v[None,:,:],axis=-1)
    return distance.min(axis=1)

def curve_metrics(observed,predicted,count=1001):
    a,sa=resample_curve(observed,count);b,sb=resample_curve(predicted,count)
    ab=point_polyline_distance(a,b);ba=point_polyline_distance(b,a)
    ordered=np.linalg.norm(a-b,axis=1)
    return dict(symmetric_distance_rmse_H=float(np.sqrt((np.mean(ab**2)+np.mean(ba**2))/2)),
                symmetric_hausdorff_H=float(max(ab.max(),ba.max())),ordered_arclength_rmse_H=float(np.sqrt(np.mean(ordered**2))),
                endpoint_error_H=float(np.linalg.norm(a[-1]-b[-1])),observed_length_H=float(sa[-1]),predicted_length_H=float(sb[-1]))
