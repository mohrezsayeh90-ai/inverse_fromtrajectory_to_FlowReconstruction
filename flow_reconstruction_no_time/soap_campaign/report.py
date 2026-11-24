"""Audit the matched geometry-only study and render its vector report."""
from collections import Counter
import csv
from dataclasses import asdict
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from flow_operator.profiles import Inlet
from .fit import BASE, CHECKPOINT, COUNTS, FLOW, ITERATIONS, KNOTS, OUT, POINTS, PROFILES, ROOT, SEEDS, SHA, digest, pool, write_json
from ..geometry import curve_metrics, resample_curve

DESIGNS = ('random','hybrid')
COLORS = dict(random='#b06b32',hybrid='#275f83')
SYMMETRIC = 'symmetric_distance_rmse_H'


def read(path):
    return json.loads(path.read_text())


def fit_dir(seed, profile, design, n):
    return OUT / f'seed_{seed}' / profile / design / f'N{n}'


def audit():
    assert digest(CHECKPOINT)==SHA
    stages, replays = [], []
    for seed in SEEDS:
        for profile in PROFILES:
            archive = BASE/f'seed_{seed}'/profile
            reference = ROOT/'results/particle_count/reference'/profile/'reference.npz'
            reference_sha = digest(reference)
            with np.load(reference,allow_pickle=False) as data:
                target = Inlet.from_samples(data['inlet_y'],data['inlet_values'])
            for design in DESIGNS:
                for n in COUNTS:
                    folder = fit_dir(seed,profile,design,n)
                    record = read(folder/'report.json')
                    assert (record['profile'],record['seed'],record['design'],record['count']) == (profile,seed,design,n)
                    assert record['checkpoint_sha256']==SHA
                    assert record['integrator_parity']<1e-12 and record['reference_sha256']==reference_sha
                    original = read(archive/design/f'N{n}'/'report.json')
                    np.testing.assert_allclose(record['timed_error'],original['error'],rtol=0,atol=0)
                    observations, velocities, sources = pool(archive/'observations',design,n)
                    assert len(observations)==n and sources==record['sources']
                    with np.load(folder/'arrays.npz',allow_pickle=False) as data:
                        np.testing.assert_array_equal(data['launch_velocity'],velocities)
                        np.testing.assert_array_equal(data['curves'], [resample_curve(q,POINTS)[0] for q in observations])
                        np.testing.assert_allclose(data['truth'],target.spline(data['y']),rtol=1e-12,atol=1e-13)
                        assert np.all(data['clock'][:,0]==0) and np.all(np.diff(data['clock'],axis=1)>0)
                        assert np.isfinite(data['inlet']).all()
                        error = float(np.linalg.norm(data['inlet']-data['truth'])/np.linalg.norm(data['truth']))
                        np.testing.assert_allclose(error,record['error'],rtol=1e-13)
                    selected = record['selected']
                    assert selected['valid'] and len(record['candidates'])==2
                    assert selected['value']==min(c['value'] for c in record['candidates'] if c['valid'])
                    stages.append(dict(seed=seed,profile=profile,design=design,N=n,
                                       error_percent=100*error,timed_percent=100*original['error'],
                                       converged=selected['converged'],iterations=selected['iterations'],
                                       stop=selected['stop']))
                folder = fit_dir(seed,profile,design,48)
                report = read(folder/'replay.json')
                assert report['geometry_fit_sha256']==digest(folder/'arrays.npz')
                assert len(report['tracks'])==56
                with np.load(folder/'replays.npz',allow_pickle=False) as data:
                    for i, entry in enumerate(report['tracks'],1):
                        source=ROOT/entry['source']
                        assert digest(source)==entry['source_sha256']
                        assert entry['subset']==('fit' if i<=48 else 'test')
                        observed,replay=data[f'observed_{i}'],data[f'replay_{i}']
                        np.testing.assert_array_equal(observed, np.loadtxt(source,delimiter=',',skiprows=1,usecols=(1,2)))
                        np.testing.assert_array_equal(replay[0,1:],observed[0])
                        assert entry['completed']==(len(replay)==129 and entry['boundary_exit'] is None)
                        if entry['completed']:
                            np.testing.assert_allclose(replay[-1,0],entry['horizon'],rtol=1e-13)
                        if len(replay)>=4:
                            values=curve_metrics(observed,replay[:,1:],count=129)
                            for key,value in values.items():
                                np.testing.assert_allclose(value,entry[key],rtol=1e-12,atol=1e-15)
                        replays.append(dict(seed=seed,profile=profile,design=design,**entry))
            with np.load(fit_dir(seed,profile,'random',2)/'arrays.npz') as a, np.load(fit_dir(seed,profile,'hybrid',2)/'arrays.npz') as b:
                for key in ('curves','clock','theta'):
                    np.testing.assert_array_equal(a[key],b[key])
    return stages,replays


def table(path,rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def study_figures(stages,replays,stats):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    cases = {(r['seed'],r['profile'],r['design'],r['N']):r for r in stages}
    with PdfPages(OUT/'geometry_only_report.pdf') as pdf:
        fig, axes = plt.subplots(1,2,figsize=(10,5),sharex=True,layout='constrained')
        for ax,seed in zip(axes,SEEDS,strict=True):
            for i,design in enumerate(DESIGNS):
                new = [cases[seed,p,design,48]['error_percent'] for p in PROFILES]
                old = [cases[seed,p,design,48]['timed_percent'] for p in PROFILES]
                u=np.arange(len(PROFILES)) + (-.14 if i==0 else .14)
                ax.scatter(new,u,color=COLORS[design],label=f'{design}: geometry')
                ax.scatter(old,u,facecolors='none',edgecolors=COLORS[design],label=f'{design}: timed')
            ax.set(xlabel='Inlet relative L2 error (%)',yticks=np.arange(len(PROFILES)),yticklabels=PROFILES,title=f'Seed {seed}')
            ax.axvline(5,color='.5',ls=':',lw=.8)
            ax.grid(alpha=.2)
        handles,labels=axes[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle('Matched release pools · 48 trajectories · frozen SOAP 314,000')
        pdf.savefig(fig);plt.close(fig)

        for profile in PROFILES:
            fig,axes=plt.subplots(2,3,figsize=(10,6.2),layout='constrained')
            for j,seed in enumerate(SEEDS):
                for i,design in enumerate(DESIGNS):
                    ax=axes[j,i]
                    folder=fit_dir(seed,profile,design,48)
                    with np.load(folder/'arrays.npz') as data:
                        ax.plot(data['y'],data['truth'],'k--',label='Archived inlet')
                        ax.plot(data['y'],data['inlet'],color=COLORS[design],label='Geometry-only')
                    t=np.genfromtxt(BASE/f'seed_{seed}'/profile/design/'N48/inlet.csv',delimiter=',',names=True)
                    ax.plot(t['y'],t['estimated'],color='.5',ls=':',label='Timed baseline')
                    ax.set(xlabel='y/H',ylabel='u / Uref',title=f'{seed} · {design} · 48 tracks')
                    if i==0 and j==0:
                        ax.legend(fontsize=7)
                    error=axes[j,2]
                    for key,style in (('error_percent','-'),('timed_percent','--')):
                        error.plot(COUNTS,[cases[seed,profile,design,n][key] for n in COUNTS],
                                   style,marker='o',ms=3,color=COLORS[design])
                error.set(xscale='log',xticks=COUNTS,xticklabels=COUNTS,xlabel='Trajectories',
                          ylabel='Relative inlet error (%)',title='Solid: geometry · dashed: timed')
                for ax in axes[j]:
                    ax.grid(alpha=.2)
                lim=max(ax.get_ylim()[1] for ax in axes[j,:2])
                for ax in axes[j,:2]:
                    ax.set_ylim(0,lim)
            fig.suptitle(profile)
            pdf.savefig(fig);plt.close(fig)

        fig, axes=plt.subplots(1,2,figsize=(10,5.5),layout='constrained')
        for seed,marker in zip(SEEDS,('o','s'),strict=True):
            for design in DESIGNS:
                entries=[r for r in replays if r['seed']==seed and r['design']==design and r['subset']=='fit']
                signed=[100*(r['horizon']/r['observed_duration']-1) for r in entries]
                axes[0].scatter(np.arange(len(signed)),signed,s=7,alpha=.35,marker=marker,
                                color=COLORS[design],label=f'{seed} · {design}')
                test=[r for r in replays if r['seed']==seed and r['design']==design and r['subset']=='test' and r['completed']]
                axes[1].scatter(np.arange(len(test)),[100*r[SYMMETRIC] for r in test],s=9,alpha=.45,
                                marker=marker,color=COLORS[design])
        axes[0].axhline(0,color='.4',lw=.7)
        axes[0].set(xlabel='Fit trajectory index per arm',ylabel='Inferred duration difference (%)',yscale='symlog')
        axes[0].legend(fontsize=7)
        axes[1].set(xlabel='Completed held-out trajectory index per arm',ylabel='Symmetric curve distance (% H)',yscale='log')
        fig.suptitle('Independent replay and post-fit clock diagnostics')
        pdf.savefig(fig);plt.close(fig)

        fig,ax=plt.subplots(figsize=(10,5.5));ax.axis('off')
        lines=['Matched protocol and numerical audit',
               '',
               'Geometry input: ordered x/y, launch velocity and Stokes time 0.42.',
               'A positive latent clock is fitted independently for each curve.',
               'SOAP checkpoint 314,000 is frozen; FEM provides post-fit profile scoring.',
               '65 arclength points, 17 clock knots, two starts, 250 iterations per start.',
               'Held-out replay starts from unseen releases and uses the reference horizon only after fitting.',
               'Full duration diagnostics for fitted trajectories use the inferred clock.',
               '',
               f"Stage records: {stats['stage_records']} · unique fits: {stats['unique_fits']}",
               f"Optimizer convergence among unique selected fits: {stats['converged_unique']}",
               f"Fit replays completed: {stats['fit_complete']} / {stats['fit_total']}",
               f"Held-out replays completed: {stats['test_complete']} / {stats['test_total']}",
               'A smaller geometric mismatch does not establish better inlet recovery.']
        for i,line in enumerate(lines):
            ax.text(.03,.95-i*.062,line,fontsize=15 if i==0 else 10,transform=ax.transAxes)
        pdf.savefig(fig);plt.close(fig)


def main():
    stages,replays = audit()
    last=[r for r in stages if r['N']==48]
    unique=[r for r in stages if r['N']!=2 or r['design']=='random']
    pairs={(r['seed'],r['profile']):r for r in last if r['design']=='random'}
    stats = dict(source_commit='a9984f07e91680eb22e2a236e74116190d0cb210',checkpoint_sha256=SHA,
                 stages_per_arm=COUNTS,stage_records=len(stages),unique_fits=len(unique),
                 converged_unique=sum(r['converged'] for r in unique),
                 stops=dict(Counter(r['stop'] for r in unique)),flow_parameterization=asdict(FLOW),
                 geometry_config=dict(points=POINTS,clock_knots=KNOTS,starts=2,max_iterations=ITERATIONS),
                 geometry_less_than_timed=sum(r['error_percent']<r['timed_percent'] for r in last),
                 hybrid_less_than_random=sum(r['error_percent']<pairs[r['seed'],r['profile']]['error_percent']
                                             for r in last if r['design']=='hybrid'),
                 below_five_percent={d:sum(r['error_percent']<=5 for r in last if r['design']==d) for d in DESIGNS},
                 median_N48={d:float(np.median([r['error_percent'] for r in last if r['design']==d])) for d in DESIGNS})
    for subset in ('fit','test'):
        rows=[r for r in replays if r['subset']==subset]
        stats[f'{subset}_total']=len(rows)
        done=[r for r in rows if r['completed']]
        stats[f'{subset}_complete']=len(done)
        stats[f'{subset}_median_symmetric_distance_H']=float(np.median([r[SYMMETRIC] for r in done]))
    differences=[100*(r['horizon']/r['observed_duration']-1) for r in replays if r['subset']=='fit']
    stats['clock_signed_percentiles_5_50_95']=[float(v) for v in np.percentile(differences,[5,50,95])]
    table(OUT/'stage_metrics.csv',stages)
    table(OUT/'replay_metrics.csv',replays)
    study_figures(stages,replays,stats)
    stats['pdf_sha256']=digest(OUT/'geometry_only_report.pdf')
    write_json(OUT/'audit.json',stats)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
