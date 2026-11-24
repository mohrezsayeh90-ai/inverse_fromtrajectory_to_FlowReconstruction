"""Matched geometry-only reconstruction on the timed study's release pools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from flow_operator.model import context
from flow_operator.profiles import Inlet
from flow_reconstruction.inverse import FlowReconstructor, InverseConfig
from flow_reconstruction_no_time.geometry import resample_curve
from flow_reconstruction_no_time.inverse import integral_positions, positive_clock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/soap_particle_count'))
from cached_operator import CachedObjective

BASE = ROOT / 'New_Adaptive_Particle_Results/inverse_rewrite'
OUT = ROOT / 'flow_reconstruction_no_time/soap_campaign/results'
CHECKPOINT = ROOT / 'reports/experiments/soap_cosine_200k/final_checkpoint.npz'
SHA = 'e46687f8d50818329072eca1af7c70136fb9e3d314207f89e9427ccc140dfbf9'
PROFILES = ('beta5', 'skew_b3', 'flat30', 'twin_b3_r30', 'shoulder', 'mixed', 'waves3_r30', 'lobes4_r30', 'nine30')
SEEDS = (39421, 53129)
COUNTS = (2, 4, 8, 16, 32, 48)
POINTS, KNOTS, ITERATIONS = 65, 17, 250
STOKES = .42
FLOW = InverseConfig(reference_reynolds=30.)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def pool(folder, design, count):
    """Load ordered x/y only, preserving release order and known velocities."""
    sizes = {'random': count, 'near': 0}
    keys = [('random', i) for i in range(1, count + 1)]
    if design == 'hybrid':
        keys = keys[:2]
        sizes = dict(random=2, near=0)
        for i in range(2, count):
            kind = 'random' if (i - 2) % 3 == 2 else 'near'
            sizes[kind] += 1
            keys.append((kind, sizes[kind]))
    launches = {kind: json.loads((folder / kind / 'launches.json').read_text())['launches']
                for kind, n in sizes.items() if n}
    positions, velocities, sources = [], [], []
    for kind, index in keys:
        path = folder / kind / f'track_{index}.csv'
        xy = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(1, 2))
        v0 = launches[kind][index-1]['velocity']
        positions.append(xy)
        velocities.append(v0)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                            launch_sha256=digest(folder / kind / 'launches.json')))
    return positions, np.asarray(velocities), sources


def integrate(fluid, clock, initial, velocity, z, w):
    # The original Stokes Green-function quadrature, written as a causal scan.
    dt = jnp.diff(clock)
    remaining = dt[:, None] * (1 - z)
    gain = -jnp.expm1(-remaining / STOKES)
    dv = jnp.sum((dt[:, None] * w * (1 - gain) / STOKES)[..., None] * fluid, axis=1)
    dq = jnp.sum((dt[:, None] * w * gain)[..., None] * fluid, axis=1)

    def step(state, impulse):
        q, v = state
        h, dp, kick = impulse
        decay = -jnp.expm1(-h / STOKES)
        q = q + STOKES * decay * v + dp
        v = (1 - decay) * v + kick
        return (q, v), q

    return jax.lax.scan(step, (initial, velocity), (dt, dq, dv))[1]


class Estimator:
    def __init__(self):
        self.m = FlowReconstructor(CHECKPOINT, FLOW, allow_unvalidated=True)
        m = self.m
        self.cached = CachedObjective(self, FLOW)
        self.size = len(m.bounds)
        z, w = np.polynomial.legendre.leggauss(3)
        self.z, w = jnp.asarray((z + 1) / 2), jnp.asarray(w / 2)
        self.basis = jnp.asarray(CubicSpline(np.linspace(0, 1, KNOTS), np.eye(KNOTS), axis=0)(
            np.linspace(0, 1, POINTS)))
        self.parity = self.verify_integrator(w)

        def clocks(theta, data):
            controls = theta[self.size:].reshape(-1, KNOTS - 1)
            speed = jnp.linalg.norm(data['velocity'], axis=1)
            return jax.vmap(positive_clock, in_axes=(0, None, 0, 0))(
                controls, self.basis, data['ds'], speed)

        def loss(theta, data):
            inlet, speed, reynolds = m.decode(theta[:self.size])
            profile = m.builder.arrays(inlet)
            branch = context(m.params, profile, reynolds, m.operator.config)
            fields = self.cached.velocity(profile, reynolds, branch, data['xy'], data['basis'])
            fields = fields.reshape(-1, POINTS - 1, 3, 2) * speed
            clock = clocks(theta, data)
            prediction = jax.vmap(integrate, in_axes=(0, 0, 0, 0, None, None))(
                fields, clock, data['q'][:, 0], data['velocity'], self.z, w)
            path = jnp.mean(jnp.sum(((prediction - data['q'][:, 1:]) / FLOW.position_scale)**2, axis=-1))
            a, b = profile['coeff'][:2]
            curvature = jnp.sum((12*a*a + 12*a*b + 4*b*b) / m.builder.h**3)
            physical = self.cached.momentum(profile, reynolds, branch)
            prior = FLOW.smoothness*curvature + FLOW.shape_prior*jnp.mean(inlet**2)
            prior += FLOW.mean_prior*theta[self.size-1]**2
            controls = jnp.pad(theta[self.size:].reshape(-1, KNOTS-1), ((0, 0), (1, 0)))
            clock_prior = .001*jnp.mean(jnp.diff(controls, n=2, axis=1)**2)
            negative = jnp.minimum(m.builder.dense@(jnp.asarray(m.builder.base)*jnp.exp(jnp.pad(inlet,(1,1)))), 0)
            residual = FLOW.physics_weight*jnp.mean(jnp.sum((speed**2*physical/FLOW.physics_scale)**2, axis=-1))
            return path + prior + clock_prior + 1e5*jnp.mean(negative**2) + residual

        self.clocks = jax.jit(clocks)
        self.evaluate = jax.jit(jax.value_and_grad(loss))

    def verify_integrator(self, w):
        rng = np.random.default_rng(23)
        fluid = jnp.asarray(rng.normal(size=(POINTS-1, 3, 2)))
        clock = jnp.asarray(np.r_[0, np.cumsum(rng.uniform(.002, .04, POINTS-1))])
        initial, velocity = jnp.array([.4, .6]), jnp.array([.8, -.2])
        fast = lambda u,t: integrate(u,t,initial,velocity,self.z,w)
        reference = lambda u,t: integral_positions(u.reshape(-1,2),t-t[0],initial,velocity,STOKES,self.z,w)
        np.testing.assert_allclose(fast(fluid,clock), reference(fluid,clock), atol=1e-13)
        for f, r in zip(jax.grad(lambda u,t:jnp.sum(fast(u,t)**2), (0,1))(fluid,clock),
                        jax.grad(lambda u,t:jnp.sum(reference(u,t)**2), (0,1))(fluid,clock), strict=True):
            np.testing.assert_allclose(f,r,atol=1e-11,rtol=1e-10)
        return float(np.max(abs(fast(fluid,clock)-reference(fluid,clock))))

    def data(self, observations, velocities):
        entries = [resample_curve(q, POINTS) for q in observations]
        q = jnp.asarray(np.stack([entry[0] for entry in entries]))
        ds = jnp.asarray([entry[1][-1]/(POINTS-1) for entry in entries])
        xy = (q[:, :-1, None]*(1-self.z)[None,None,:,None] +
              q[:, 1:, None]*self.z[None,None,:,None]).reshape(-1, 2)
        return dict(q=q, ds=ds, velocity=jnp.asarray(velocities), xy=xy,
                    basis=self.cached.basis_velocity(xy))

    def release_clock(self, data):
        """Initialize clocks from arc length and Stokes relaxation in a simple field."""
        u = np.asarray(data['q'])
        v0 = np.asarray(data['velocity'])
        lengths = np.asarray(data['ds']) * (POINTS-1)
        initial_speed = np.linalg.norm(v0, axis=1)
        y = u[:,:,1].mean(axis=1)
        fluid = np.column_stack([6*y*(1-y),np.zeros(len(y))])
        # A deliberately broad model-time grid, independent of the archived observation windows.
        t = np.r_[0., np.geomspace(1e-6,128.,8192)]
        fraction = np.linspace(0,1,KNOTS)[1:]
        controls = []
        for launch, speed, flow, length in zip(v0,initial_speed,fluid,lengths,strict=True):
            v = flow + (launch-flow)*np.exp(-t[:,None]/STOKES)
            tangential_speed = np.linalg.norm(v,axis=1)
            arc = np.r_[0.,np.cumsum(np.diff(t)*(tangential_speed[:-1]+tangential_speed[1:])*0.5)]
            assert arc[-1]>=length, (arc[-1],length)
            times = np.interp(fraction*length,arc,t)
            speed_ratio = speed/np.interp(times,t,tangential_speed)
            controls.append(5*np.arctanh(np.clip(np.log(speed_ratio)/5,-.95,.95)))
        return np.concatenate(controls)

    def fit(self, data, budget=ITERATIONS, use_release_clock=False):
        n = len(data['q'])
        initial = np.r_[np.zeros(self.size), np.tile(2*np.linspace(0,1,KNOTS)[1:], n)]
        if use_release_clock:
            initial[self.size:] = self.release_clock(data)
        rng = np.random.default_rng(81206)
        perturbed = initial.copy()
        perturbed[:self.size-1] += rng.normal(0,.04,self.size-1)
        perturbed[self.size:] += rng.normal(0,.05,len(initial)-self.size)
        limits = self.m.bounds + [(None,None)]*(len(initial)-self.size)

        def objective(theta):
            value, gradient = self.evaluate(theta, data)
            if not (np.isfinite(value) and np.isfinite(gradient).all()):
                raise FloatingPointError('Non-finite geometry objective')
            return float(value), np.asarray(gradient)

        candidates = []
        for start in (initial, perturbed):
            result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=limits,
                              options=dict(maxiter=budget,maxls=50,maxcor=30,ftol=1e-12,gtol=1e-6))
            inlet, _, _ = self.m.decode(jnp.asarray(result.x[:self.size]))
            candidates.append(dict(theta=result.x.tolist(), value=float(result.fun), iterations=int(result.nit),
                                   stop=str(result.message), converged=bool(result.success),
                                   valid=self.m.builder.minimum(inlet) >= -1e-7))
        return min((c for c in candidates if c['valid']), key=lambda c:c['value']), candidates


def run_case(model, name, seed, designs):
    archive = BASE / f'seed_{seed}' / name
    reference = ROOT / 'results/particle_count/reference' / name / 'reference.npz'
    y = np.linspace(0,1,4097)
    with np.load(reference, allow_pickle=False) as raw:
        truth = Inlet.from_samples(raw['inlet_y'], raw['inlet_values']).spline(y)
    shared = None
    for design in designs:
        for n in COUNTS:
            dest = OUT / f'seed_{seed}' / name / design / f'N{n}'
            if (dest / 'report.json').exists():
                continue
            observations, velocities, sources = pool(archive/'observations',design,n)
            data = model.data(observations,velocities)
            if n==2 and shared is not None:
                selected, candidates = shared
            else:
                selected, candidates = model.fit(data)
            if n==2:
                shared = selected, candidates
            theta = jnp.asarray(selected['theta'])
            inlet, speed, re = model.m.decode(theta[:model.size])
            curves = np.asarray(data['q'])
            clock = np.asarray(model.clocks(theta,data))
            assert np.all(np.diff(clock,axis=1)>0)
            prediction = model.m.builder.inlet(inlet).spline(y)*float(speed)
            error = float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth))
            old = json.loads((archive/design/f'N{n}'/'report.json').read_text())
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'arrays.npz', y=y, truth=truth, inlet=prediction, curves=curves, clock=clock,
                                launch_velocity=velocities, theta=theta)
            write_json(dest/'report.json',dict(profile=name, seed=seed, design=design, count=n, error=error,
                       timed_error=old['error'], speed=float(speed), reynolds=float(re), sources=sources,
                       reference_sha256=digest(reference), checkpoint_sha256=SHA, integrator_parity=model.parity,
                       points=POINTS, clock_knots=KNOTS, budget=ITERATIONS, selected=selected, candidates=candidates))
            print(seed,name,design,n,round(100*error,3),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True,choices=SEEDS)
    parser.add_argument('--designs',nargs='+',choices=('random','hybrid'),default=['random','hybrid'])
    parser.add_argument('profiles',nargs='+',choices=PROFILES)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    assert digest(CHECKPOINT)==SHA
    model = Estimator()
    for profile in args.profiles:
        run_case(model,profile,args.seed,args.designs)
