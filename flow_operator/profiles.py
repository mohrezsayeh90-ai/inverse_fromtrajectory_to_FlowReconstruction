"""Preserve the complete supplied spline, without a Fourier-family projection."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


@lru_cache(maxsize=16)
def quadrature(points):
    z,w=np.polynomial.legendre.leggauss(points)
    return (z+1)/2,w/2


@dataclass
class Inlet:
    y: np.ndarray
    values: np.ndarray
    mean_velocity: float
    spline: CubicSpline
    encoder_points: int = 257

    @classmethod
    def from_samples(cls, y, values, encoder_points=257):
        y, values = np.asarray(y, float), np.asarray(values, float)
        if y.ndim != 1 or values.shape != y.shape or len(y) < 4:
            raise ValueError('Supply at least four (y, axial velocity) samples')
        if not np.isfinite(y).all() or not np.isfinite(values).all() or np.any(np.diff(y) <= 0):
            raise ValueError('Coordinates must increase strictly; all samples must be finite')
        if abs(y[0]) > 1e-12 or abs(y[-1]-1) > 1e-12:
            raise ValueError('The nondimensional inlet coordinate must span [0,1]')
        if np.max(abs(values[[0,-1]])) > 1e-9*max(1.,np.max(abs(values))):
            raise ValueError('The no-slip walls require zero velocity at y=0 and y=1')
        if encoder_points < 16:
            raise ValueError('Use at least 16 encoder quadrature points')
        spline = CubicSpline(y, values)
        mean = float(spline.integrate(0,1))
        if not mean > 1e-12:
            raise ValueError('A positive mean is required by the chosen inlet-flow scaling')
        values = values/mean
        return cls(y.copy(), values, mean, CubicSpline(y,values), encoder_points)

    @classmethod
    def from_csv(cls, path, encoder_points=257):
        data = np.loadtxt(Path(path), delimiter=',',skiprows=1)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError('CSV must have header y,u followed by two numeric columns')
        return cls.from_samples(data[:,0],data[:,1],encoder_points)

    def arrays(self, dtype=np.float32):
        # Use a local t in [0,1] to avoid large inverse-mesh-power coefficients.
        h = np.diff(self.y)
        coeff = self.spline.c*np.stack([h**3,h**2,h,np.ones_like(h)])
        integral = self.spline.antiderivative()
        ey,ew = quadrature(self.encoder_points)
        dense = np.linspace(0,1,max(1025,4*len(self.y)))
        delta = self.spline(dense)-6*dense*(1-dense)
        derivative = self.spline(dense,1)-(6-12*dense)
        energy = np.trapezoid(delta**2,dense)
        effective_mode = np.sqrt(np.trapezoid(derivative**2,dense)/max(energy,1e-12))/np.pi if energy>1e-10 else 1.
        return {k:np.asarray(v,dtype=dtype) for k,v in dict(
            knots=self.y, coeff=coeff, prefix=integral(self.y)-integral(0),
            encoder_y=ey,encoder_w=ew,encoder_f=self.spline(ey),
            effective_mode=np.clip(effective_mode,1,128)).items()}

    def diagnostics(self):
        y = np.linspace(0,1,max(4097,8*len(self.y)))
        return dict(input_samples=len(self.y), encoder_points=self.encoder_points,
            normalization_mean=self.mean_velocity, normalized_flux=float(self.spline.integrate(0,1)),
            normalized_min=float(self.spline(y).min()), normalized_max=float(self.spline(y).max()),
            local_reverse_flow=bool(self.spline(y).min() < -1e-8),
            sample_preservation_max_error=float(np.max(abs(self.spline(self.y)-self.values))),
            representation='C2 interpolating cubic spline through every supplied sample; no mode projection',
            resolution_notice='Samples alone cannot prove that an unknown underlying function is resolved.')


def stack(profiles, dtype=np.float32):
    values=[p.arrays(dtype) for p in profiles]
    return {k:np.stack([p[k] for p in values]) for k in values[0]}
