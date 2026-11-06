"""One-way point particles: dq/dt=w, dw/dt=(U(q)-w)/St."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Track:
    name: str
    time: np.ndarray
    position: np.ndarray
    initial_velocity: np.ndarray
    stokes_time: float

    def __post_init__(self):
        self.time=np.asarray(self.time,float);self.position=np.asarray(self.position,float)
        self.initial_velocity=np.asarray(self.initial_velocity,float)
        if self.time.ndim!=1 or len(self.time)<4 or self.position.shape!=(len(self.time),2):
            raise ValueError('A track needs at least four timestamped (t,x,y) samples')
        if not all(np.isfinite(a).all() for a in [self.time,self.position,self.initial_velocity]):
            raise ValueError('Track values must be finite')
        if np.any(np.diff(self.time)<=0) or self.initial_velocity.shape!=(2,):
            raise ValueError('Times must increase strictly and launch velocity must have two components')
        if not np.isfinite(self.stokes_time) or self.stokes_time<=0:
            raise ValueError('Known Stokes relaxation time must be positive')


def simulate(velocity,time,initial_position,initial_velocity,stokes_time,length=5.,rtol=2e-9,atol=2e-11):
    """Independent adaptive ODE solve; ends at a boundary, never reflects or clips."""
    time=np.asarray(time,float);q0=np.asarray(initial_position,float);v0=np.asarray(initial_velocity,float)
    if time.ndim!=1 or len(time)<2 or np.any(np.diff(time)<=0) or not np.isfinite(time).all():
        raise ValueError('Simulation times must be finite and strictly increasing')
    if q0.shape!=(2,) or v0.shape!=(2,) or not np.isfinite(np.r_[q0,v0,stokes_time,length]).all() or min(stokes_time,length)<=0:
        raise ValueError('Invalid launch state or relaxation time')
    if not (0<=q0[0]<=length and 0<=q0[1]<=1):raise ValueError('Launch is outside the channel')
    for axis,bound,sign in [(0,0.,1),(0,length,-1),(1,0.,1),(1,1.,-1)]:
        if q0[axis]==bound and sign*v0[axis]<=0:raise ValueError('A boundary launch must point strictly into the channel')
    def rhs(t,state):
        # Event root finding can probe outside. Only the RHS extension is clipped;
        # the particle path is terminated at the first actual boundary crossing.
        point=np.clip(state[:2],[0.,0.],[length,1.])
        fluid=np.asarray(velocity(point),float)
        if fluid.shape!=(2,) or not np.isfinite(fluid).all():raise ValueError('Invalid fluid velocity')
        return np.r_[state[2:],(fluid-state[2:])/stokes_time]
    events=[]
    for axis,bound,sign in [(0,0.,1),(0,length,-1),(1,0.,1),(1,1.,-1)]:
        def event(t,s,axis=axis,bound=bound,sign=sign):return sign*(s[axis]-bound)
        event.terminal=True;event.direction=-1;events.append(event)
    sol=solve_ivp(rhs,(time[0],time[-1]),np.r_[q0,v0],t_eval=time,events=events,
                  rtol=rtol,atol=atol,max_step=stokes_time/3)
    if not sol.success:raise RuntimeError(sol.message)
    return dict(time=sol.t,position=sol.y[:2].T,velocity=sol.y[2:].T,
                completed=bool(len(sol.t)==len(time)),boundary_exit_time=next((float(t[0]) for t in sol.t_events if len(t)),None))


def integral_observations(tracks,length=5.,order=3):
    """Gauss integration of q=q0+St*w0*(1-exp(-dt/St))+int(1-exp(-(t-s)/St))*U(q(s))ds.

    Interpolation uses measured positions, not their noisy accelerations.
    The independent final ODE solve is required to check the inferred field.
    """
    if order<2:raise ValueError('Use at least two quadrature points per interval')
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    points=[];weights=[];known=[];observed=[];scales=[];counts=[]
    for track in tracks:
        t=track.time-track.time[0];q=track.position;tau=track.stokes_time
        if np.any(q<0) or np.any(q[:,0]>length) or np.any(q[:,1]>1):
            raise ValueError('Observed trajectories must stay inside the channel; truncate at exit')
        dt=np.diff(t);s=(t[:-1,None]+dt[:,None]*z).ravel();qw=(dt[:,None]*w).ravel()
        xy=(q[:-1,None,:]*(1-z)[None,:,None]+q[1:,None,:]*z[None,:,None]).reshape(-1,2)
        lag=t[1:,None]-s[None,:]
        matrix=np.where(lag>0,-np.expm1(-np.maximum(lag,0)/tau),0)*qw[None,:]
        points.append(xy);weights.append(matrix)
        known.append(q[0]+tau*(-np.expm1(-t[1:]/tau))[:,None]*track.initial_velocity)
        observed.append(q[1:]);scales.append(np.full(len(t)-1,1/np.sqrt(len(t)-1)))
        counts.append(len(t)-1)
    if not points:raise ValueError('No trajectories supplied')
    matrix=np.zeros((sum(counts),sum(len(p) for p in points)));i=j=0
    for m in weights:matrix[i:i+len(m),j:j+m.shape[1]]=m;i+=len(m);j+=m.shape[1]
    return dict(xy=np.concatenate(points),weights=matrix,known=np.concatenate(known),
                observed=np.concatenate(observed),scales=np.concatenate(scales),counts=counts)
