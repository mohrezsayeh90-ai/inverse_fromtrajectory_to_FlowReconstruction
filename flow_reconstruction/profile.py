"""Differentiable full cubic spline, with configurable nodal resolution."""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from flow_operator.profiles import Inlet,quadrature


class ProfileBuilder:
    def __init__(self,points=65,encoder_points=257):
        if points<9:raise ValueError('Use at least nine inlet knots')
        self.y=np.linspace(0,1,points);self.base=6*self.y*(1-self.y)
        basis=CubicSpline(self.y,np.eye(points),axis=0);h=np.diff(self.y)
        self.coeff=jnp.asarray(basis.c*np.stack([h**3,h**2,h,np.ones_like(h)])[:,:,None])
        self.mass=jnp.asarray(basis.integrate(0,1));self.h=jnp.asarray(h)
        self.ey,self.ew=map(jnp.asarray,quadrature(encoder_points));self.encoder=jnp.asarray(basis(np.asarray(self.ey)))
        dense=np.linspace(0,1,max(1025,4*points));self.dense_y=jnp.asarray(dense)
        self.dense=jnp.asarray(basis(dense));self.derivative=jnp.asarray(basis(dense,1))
        self.second_difference=np.diff(np.eye(points-2),n=2,axis=0)/(h[0]**2)

    def arrays(self,z):
        raw=jnp.asarray(self.base)*jnp.exp(jnp.pad(z,(1,1)))
        values=raw/(self.mass@raw);coeff=jnp.einsum('ijn,n->ij',self.coeff,values)
        segment=self.h*jnp.sum(coeff*jnp.array([.25,1/3,.5,1.])[:,None],axis=0)
        dense=self.dense@values;delta=dense-6*self.dense_y*(1-self.dense_y)
        derivative=self.derivative@values-(6-12*self.dense_y)
        energy=jnp.trapezoid(delta**2,self.dense_y)
        numerator=jnp.maximum(jnp.trapezoid(derivative**2,self.dense_y),1e-20)
        mode=jnp.where(energy>1e-10,jnp.sqrt(numerator/jnp.maximum(energy,1e-12))/jnp.pi,1.)
        return dict(knots=jnp.asarray(self.y),coeff=coeff,prefix=jnp.r_[jnp.array([0.]),jnp.cumsum(segment)],
                    encoder_y=self.ey,encoder_w=self.ew,encoder_f=self.encoder@values,effective_mode=jnp.clip(mode,1,128))

    def inlet(self,z):
        values=self.base*np.exp(np.pad(np.asarray(z),(1,1)))
        return Inlet.from_samples(self.y,values)

    def minimum(self,z):
        p=self.inlet(z);critical=p.spline.derivative().roots(extrapolate=False)
        return float(np.min(p.spline(np.r_[0.,critical[(critical>=0)&(critical<=1)],1.])))
