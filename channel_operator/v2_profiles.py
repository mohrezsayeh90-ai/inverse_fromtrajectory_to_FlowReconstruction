"""Analytic smooth four-mode inlets; no spatial interpolation inside PDE derivatives."""
import numpy as np
from .profiles import Y

MODES = np.arange(1, 5)
INTEGRALS = -6*(1+(-1.)**MODES)/(np.pi*MODES)**2
A = 6*Y[:, None]*(1-Y[:, None])*np.cos(np.pi*Y[:, None]*np.arange(5))
PROJECT = np.linalg.pinv(A)


def coefficients_to_b(a):
    a = np.asarray(a, dtype=float)
    norm = 1+a@INTEGRALS
    return np.concatenate([np.ones((*a.shape[:-1], 1)), a], -1)/norm[..., None]


def inlet_values(a, y=Y):
    b = coefficients_to_b(a)
    y = np.asarray(y)
    return (b@np.cos(np.arange(5)[:, None]*np.pi*y.ravel())).reshape((*b.shape[:-1], *y.shape))*6*y*(1-y)


def sample(rng, n, radius=.8):
    """Cover mixtures AND axes; monitor profiles at exact |a_k|=.6 are excluded."""
    a = rng.normal(size=(n, 4))
    a *= rng.uniform(.05, radius, (n, 1))/np.abs(a).sum(1, keepdims=True)
    kind = rng.random(n)
    for i in range(n):
        if kind[i] < .35:
            a[i] = 0
            k = rng.integers(4)
            value = rng.uniform(-radius, radius)
            if abs(abs(value)-.6)<1e-4:
                value *= .99
            a[i,k] = value
        elif kind[i] < .55:
            keep = rng.choice(4, 2, replace=False)
            mask = np.zeros(4); mask[keep] = 1
            a[i] *= mask
            a[i] *= rng.uniform(.4, radius)/max(np.abs(a[i]).sum(),1e-12)
    re = np.exp(rng.uniform(np.log(10), np.log(100), n))
    edge = rng.random(n)<.25
    re[edge] = rng.choice([10.,100.],size=edge.sum())
    return inlet_values(a), re, a


def monitoring_conditions():
    a, res = [], []
    for mode in range(4):
        for sign in [-1,1]:
            for re in [10.,100.]:
                c=np.zeros(4); c[mode]=sign*.6
                a.append(c); res.append(re)
    a=np.array(a)
    return inlet_values(a), np.array(res), a
