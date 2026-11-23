"""Repeat the adaptive particle campaign with the frozen 314k SOAP checkpoint."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'New_Adaptive_Particle_Results'
NAMES = ['beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed',
         'waves3_r30','lobes4_r30','nine30']
CHECKPOINT = 'reports/experiments/soap_cosine_200k/final_checkpoint.npz'
SHA = 'e46687f8d50818329072eca1af7c70136fb9e3d314207f89e9427ccc140dfbf9'
REVISION = 'f6b8690059716e809c2dac27c8715e65fcb1d267'


def run(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    sys.path.insert(0,str(args.flow.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.particle_count import fit_campaign as legacy
    from flow_operator.model import context
    from flow_operator.profiles import Inlet
    from flow_reconstruction.particles import Track, integral_observations, simulate
    from cached_operator import CachedObjective

    out = BASE/args.variant
    observations = out/'observations' if args.extend_near else BASE/('near_inlet/observations' if args.near_inlet else 'observations')
    out.mkdir(exist_ok=True)
    link = out/'observations'
    if args.extend_near and not (observations/'nine30/launches.json').exists():
        raise ValueError('Generate the extended observations first')
    try:
        link.symlink_to(os.path.relpath(observations,out))
    except FileExistsError:
        if link.resolve() != observations.resolve():
            raise ValueError('Existing observations differ from the selected design')

    legacy.OUT = out
    legacy.CP = args.flow/CHECKPOINT
    legacy.SHA = SHA
    if legacy.sha(legacy.CP) != SHA:
        raise ValueError('SOAP checkpoint SHA-256 differs from the published archive')
    if args.near_inlet:
        legacy.CONFIG = dataclasses.replace(legacy.CONFIG,smoothness=1e-8,physics_weight=0.,max_iterations=400)
    config = json.loads(json.dumps(dataclasses.asdict(legacy.CONFIG)))

    def reference(name):
        return args.flow/f'results/particle_count/reference/{name}/reference.npz'

    def profile(name):
        with np.load(reference(name),allow_pickle=False) as data:
            return Inlet.from_samples(data['inlet_y'],data['inlet_values'])

    legacy.profile = profile
    fitter = legacy.CampaignFitter()
    if fitter.m.operator.metadata['step'] != 314000:
        raise ValueError('Unexpected training step')
    fast = CachedObjective(fitter,legacy.CONFIG)
    uncached = fitter.loss
    fitter.loss = fast.loss
    fitter.data = lambda tracks: fast.data(tracks,integral_observations)
    model = fitter.m

    def replay(theta, track):
        z,speed,re = model.decode(jnp.asarray(theta))
        inlet = model.builder.arrays(z)
        branch = context(model.params,inlet,re,model.operator.config)
        return simulate(lambda xy: np.asarray(fast.point(jnp.asarray(xy),branch,inlet,re,speed)),
                        track.time,track.position[0],track.initial_velocity,track.stokes_time,
                        model.operator.config.length)

    model.simulate = replay
    minimize = legacy.minimize
    for name in args.profiles or NAMES:
        progress = out/'progress'/f'{name}.json'
        progress.parent.mkdir(exist_ok=True)
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved['phase']=='completed':
                if saved['checkpoint_sha256']!=SHA or saved['config']!=config:
                    raise ValueError(f'{progress}: existing run uses other weights or inverse settings')
                continue
        launch = json.loads((observations/name/'launches.json').read_text())
        if launch['reference_sha256'] != legacy.sha(reference(name)):
            raise ValueError(f'{name}: reference field changed')
        tracks = []
        hashes = {}
        for k,item in enumerate(launch['launches'],1):
            path = observations/name/f'track_{k}.csv'
            d = np.genfromtxt(path,delimiter=',',names=True)
            tracks.append(Track(f'p{k}',d['t'],np.column_stack([d['x'],d['y']]),item['velocity'],launch['St']))
            hashes[path.name] = legacy.sha(path)
        if args.parity and not (out/'objective_parity.json').exists():
            samples = tracks[:2]
            original = {k:jnp.asarray(v) for k,v in integral_observations(samples,5,legacy.CONFIG.quadrature_order).items() if k!='counts'}
            cached = fitter.data(samples)
            checks = []
            for theta in [np.zeros(len(model.bounds)),np.random.default_rng(29).normal(0,.02,len(model.bounds))]:
                p,g = uncached(theta,original,2.)
                q,h = fast.loss(theta,cached,2.)
                np.testing.assert_allclose(float(p),float(q),rtol=1e-9,atol=1e-10)
                np.testing.assert_allclose(g,h,rtol=1e-7,atol=1e-8)
                checks.append(dict(absolute_loss_difference=float(abs(p-q)),max_gradient_difference=float(np.max(np.abs(g-h)))))
            legacy.dump(out/'objective_parity.json',dict(passed=True,profile=name,checkpoint=SHA,jax=jax.__version__,checks=checks))
            print('PARITY',checks,flush=True)

        stages = {}
        def fit(count):
            if count in stages:
                return stages[count]
            folder = out/'fits'/name/f'N{count}'
            report = folder/'report.json'
            if args.extend_near and count<=24 and not report.exists():
                from shutil import copytree
                earlier = BASE/'soap_314k_near/fits'/name/f'N{count}'
                if (earlier/'report.json').exists():
                    for k in range(1,count+1):
                        track = f'track_{k}.csv'
                        assert hashes[track]==legacy.sha(BASE/'near_inlet/observations'/name/track)
                    copytree(earlier,folder)
            if report.exists():
                info = json.loads(report.read_text())
                if info['checkpoint_sha256']!=SHA or info['config']!=config:
                    raise ValueError(f'{report}: wrong checkpoint or settings')
            else:
                lower = [n for n in stages if n<count]
                prior = max(lower,default=None)
                warm = None
                if prior is not None:
                    previous = stages[prior]['report']
                    warm = np.array(previous['candidates'][previous['selected_start']]['theta'])
                def start_from_previous(fun,x,**kwargs):
                    start = warm.copy() if warm is not None and np.count_nonzero(x)>0 else x
                    return minimize(fun,start,**kwargs)
                legacy.minimize = start_from_previous
                try:
                    info = fitter.fit(name,tracks[:count])
                finally:
                    legacy.minimize = minimize
                info['initialization_source_N'] = prior
                info['forward_training_step'] = 314000
                info['source_revision'] = REVISION
                legacy.dump(report,info)
            result = dict(N=count,error=info['inlet_relative_l2'],complete=info['status']=='completed',
                          optimizer_converged=info['candidates'][info['selected_start']]['optimizer_success'])
            stages[count] = {**result,'report':info}
            return stages[count]

        adaptive = []
        n = 2
        reason = 'maximum count reached'
        while True:
            row = fit(n)
            adaptive.append(n)
            legacy.dump(progress,dict(profile=name,phase='adaptive',adaptive_counts=adaptive,
                                     evaluated=[{k:v for k,v in s.items() if k!='report'} for s in stages.values()]))
            if row['error']<=.05 and row['complete']:
                reason = 'evaluation target reached'
                break
            if n==len(tracks):
                break
            if not args.near_inlet and len(adaptive)>=4 and all(
                    stages[k]['error']>=stages[adaptive[-4]]['error']*.95 for k in adaptive[-3:]):
                reason = 'error plateau under fixed fitting budget'
                break
            next_count = max(n+2,int(np.ceil(n*(1.8 if row['error']>.25 else 1.5))))
            # Retain the exactly matched 24-track endpoint before extending.
            n = min(24 if n<24 else len(tracks),next_count)

        # Use real intermediate fits. When the threshold is found very early,
        # continue to six particles to give the figure five distinct stages.
        last = max(n,6)
        display = sorted(set(np.rint(np.linspace(2,last,min(6,last-1))).astype(int).tolist()))
        if args.extend_near and len(adaptive)>=6:
            display = sorted({2,4,8,15,24,last} if last <= 36 else {2,8,15,24,36,last})
        for count in display:
            fit(count)
        evaluated = [{k:v for k,v in stages[count].items() if k!='report'} for count in sorted(stages)]
        legacy.dump(progress,dict(profile=name,phase='completed',adaptive_counts=adaptive,
            evaluated=evaluated,display_counts=display,final_count=last,search_stop_N=n,stopping_reason=reason,
            target_relative_l2=.05,checkpoint_sha256=SHA,source_revision=REVISION,config=config,
            reference_sha256=launch['reference_sha256'],observation_sha256=hashes))
        print('COMPLETE',name,last,100*stages[last]['error'],reason,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow',type=Path,required=True)
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--variant',default='soap_314k')
    parser.add_argument('--near-inlet',action='store_true')
    parser.add_argument('--extend-near',action='store_true')
    parser.add_argument('--parity',action='store_true')
    parser.add_argument('profiles',nargs='*')
    options = parser.parse_args()
    if not set(options.profiles)<=set(NAMES):
        parser.error('Unrecognized profile')
    if options.extend_near and (not options.near_inlet or options.profiles!=['nine30']):
        parser.error('--extend-near requires --near-inlet and nine30')
    run(options)
