"""Multi-scale Gaussian-field inlet generator, adapted from the paper's code.

Source: PredictiveIntelligenceLab/Physics-informed-DeepONets, commit
77e7c747d653b34e79e43df7d88bad87de5c27d8,
Antiderivative/PI_DeepONet_FF_antideriv.ipynb, RBF and GP sampling cells.

The covariance and Cholesky sampling follow the official notebook. NumPy replaces
JAX for offline generation. The scale mixture and positive, no-slip, unit-flux
channel transformation below are adaptations, not claims about the paper's code.
No PDE solver or response labels are used. These inputs require a NEW operator;
they are not supported by the existing four-mode V2 checkpoint.
"""
from functools import lru_cache

import numpy as np
from scipy.linalg import cholesky

LENGTH_SCALES = (.01, .02, .04, .08, .15, .30)
SOURCE_COMMIT = '77e7c747d653b34e79e43df7d88bad87de5c27d8'
SOURCE_URL = ('https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/'
              'blob/'+SOURCE_COMMIT+'/Antiderivative/PI_DeepONet_FF_antideriv.ipynb')


@lru_cache(maxsize=12)
def gaussian_factor(length_scale, resolution=512):
    """RBF covariance and 1e-10 jitter as in the reference notebook."""
    x = np.linspace(0., 1., resolution)
    covariance = np.exp(-.5*((x[:, None]-x[None, :])/length_scale)**2)
    factor = cholesky(covariance+1e-10*np.eye(resolution), lower=True)
    return x, factor


def sample_inlets(count, seed, sensors=257, max_velocity=2.875):
    """Return sampled inlet functions and parameters, without u/v/p responses.

    Reject entire draws above the speed cap; do not clip or project frequencies.
    The flux is normalized by trapezoid quadrature on the returned sensor grid.
    """
    if count < 1 or sensors < 201 or max_velocity <= 1.5:
        raise ValueError('Require positive count, >=201 sensors, and speed cap >1.5')
    rng = np.random.default_rng(seed)
    y = np.linspace(0., 1., sensors)
    profiles = np.empty((count, sensors), dtype=np.float32)
    scales = np.empty(count); amplitudes = np.empty(count)
    # Balanced scales guarantee coverage even in a small generated bank.
    classes = np.arange(count) % len(LENGTH_SCALES)
    rng.shuffle(classes)
    rejected = 0
    for group, length_scale in enumerate(LENGTH_SCALES):
        indices = np.flatnonzero(classes == group)
        if not len(indices):
            continue
        grid, factor = gaussian_factor(length_scale)
        offset = 0
        while offset < len(indices):
            n = min(256, len(indices)-offset)
            fields = rng.standard_normal((n, len(grid))) @ factor.T
            amplitude = rng.uniform(.15, .75, n)
            sampled = np.array([np.interp(y, grid, g) for g in fields])
            shaped = 6*y*(1-y)*(1+amplitude[:, None]*np.tanh(sampled))
            shaped /= np.trapezoid(shaped, y, axis=1)[:, None]
            valid = np.max(shaped, axis=1) <= max_velocity
            rejected += int((~valid).sum())
            chosen = indices[offset:offset+int(valid.sum())]
            profiles[chosen] = shaped[valid]
            scales[chosen] = length_scale; amplitudes[chosen] = amplitude[valid]
            offset += len(chosen)
    reynolds = np.exp(rng.uniform(np.log(10.), np.log(100.), count)).astype(np.float32)
    return dict(y=y, profiles=profiles, reynolds=reynolds,
                length_scales=scales, amplitudes=amplitudes,
                rejected_speed_draws=np.array(rejected))
