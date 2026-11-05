"""No-slip inlet profiles, represented by normalized cubic-spline sensors."""
import numpy as np
from scipy.interpolate import CubicSpline

SENSORS = 33
Y = np.linspace(0., 1., SENSORS)
# Not-a-knot interpolation reproduces quadratics (including Poiseuille) exactly.
_basis = CubicSpline(Y, np.eye(SENSORS), axis=0)
SPLINE_MAP = _basis.c
INTEGRAL_WEIGHTS = _basis.integrate(0., 1.)


def normalize(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != SENSORS or not np.isfinite(values).all():
        raise ValueError(f"Expected {SENSORS} finite inlet samples")
    if np.max(np.abs(values[..., [0, -1]])) > 1e-8:
        raise ValueError("Inlet must be zero at both no-slip walls")
    dense = CubicSpline(Y, values, axis=-1)(np.linspace(0., 1., 257))
    if dense.min() < -1e-8:
        raise ValueError("Supported inlets must be nonnegative, including spline interpolation")
    mean = values @ INTEGRAL_WEIGHTS
    if np.any(mean <= 0):
        raise ValueError("Inlet must have a positive mean velocity")
    return values / mean[..., None]


def from_coefficients(coefficients):
    a = np.asarray(coefficients)
    modes = np.arange(1, a.shape[-1] + 1)
    modulation = 1 + a @ np.cos(np.pi * modes[:, None] * Y)
    return normalize(6 * Y * (1-Y) * modulation)


def sample_profiles(rng, count, modes=4, radius=.6):
    a = rng.normal(size=(count, modes))
    a *= rng.uniform(0., radius, (count, 1)) / np.abs(a).sum(axis=1, keepdims=True)
    return from_coefficients(a), a


def load_csv(path):
    """CSV with header y,u; y is nondimensional and spans [0,1]."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 4:
        raise ValueError("CSV must contain at least four y,u rows")
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("CSV coordinates must be finite and strictly increasing")
    if not np.allclose(data[[0, -1], 0], [0, 1], atol=1e-10):
        raise ValueError("CSV y must span [0, 1]")
    return normalize(CubicSpline(data[:, 0], data[:, 1])(Y))
