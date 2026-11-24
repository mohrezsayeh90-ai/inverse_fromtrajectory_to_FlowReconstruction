"""Render and audit the exact thesis sample refits separately from the two-seed study."""
from collections import defaultdict
import argparse
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .archive import ADAPTIVE, cases
from .fit import BASE, OUT, ROOT, SEEDS, SHA, digest, write_json
from .report import read, table


def audit(study):
    rows=[]
    for profile,n,recorded,timed,expected in cases(study):
        folder=OUT/'thesis_samples'/study/profile/f'N{n}'
        saved=read(folder/'report.json')
        assert (saved['study'],saved['profile'],saved['count'])==(study,profile,n)
        assert saved['checkpoint_sha256']==SHA
        assert saved['timed_report_sha256']==digest(timed)
        assert saved['timed_error']==read(timed)['inlet_relative_l2']
        assert len(saved['observations'])==n
        for source in saved['observations']:
            path=ROOT/source['track']
            assert digest(path)==source['sha256']
            if expected is not None:
                assert expected[path.name]==source['sha256']
        with np.load(folder/'arrays.npz',allow_pickle=False) as data:
            error=float(np.linalg.norm(data['inlet']-data['truth'])/np.linalg.norm(data['truth']))
            np.testing.assert_allclose(error,saved['error'],rtol=1e-13)
            assert np.all(np.diff(data['clock'],axis=1)>0)
            assert data['curves'].shape==(n,65,2)
        chosen=saved['selected']
        assert chosen['valid'] and chosen['value']==min(c['value'] for c in saved['candidates'] if c['valid'])
        row=dict(study=study,profile=profile,N=n,geometry_percent=100*error,
                 timed_percent=100*saved['timed_error'],iterations=chosen['iterations'],
                 converged=chosen['converged'],stop=chosen['stop'],timed_source=saved['timed_report'])
        historical=ADAPTIVE/'near_inlet/fits'/profile/f'N{n}'/'report.json'
        if study=='near' and historical.is_file():
            legacy=ADAPTIVE/'near_inlet/observations'/profile
            for source in saved['observations']:
                assert digest(legacy/(ROOT/source['track']).name)==source['sha256']
            past=read(legacy/'launches.json')
            current=read(recorded/'launches.json')
            assert past['St']==current['St']
            assert [r['velocity'] for r in past['launches'][:n]]==[r['velocity'] for r in current['launches'][:n]]
            row['legacy_timed_percent']=100*read(historical)['inlet_relative_l2']
            row['legacy_timed_sha256']=digest(historical)
        rows.append(row)
    return rows


def draw(pdf, rows):
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.2),layout='constrained')
    fig.get_layout_engine().set(w_pad=.14,h_pad=.10)
    ordered=sorted(rows,key=lambda r:r['N'])
    counts=[row['N'] for row in ordered]
    axes[0].plot(counts,[r['geometry_percent'] for r in ordered],'-o',color='#275f83',label='Geometry-only')
    axes[0].plot(counts,[r['timed_percent'] for r in ordered],'--s',color='.45',label='Archived timed fit')
    historical=[r for r in ordered if 'legacy_timed_percent' in r]
    if historical:
        axes[0].plot([r['N'] for r in historical],[r['legacy_timed_percent'] for r in historical],
                     '-.^',color='#af6938',label='Timed baseline 94,000')
    axes[0].axhline(5,color='.6',ls=':',lw=.8)
    ticks=[n for n in counts if n!=23 or 24 not in counts]
    axes[0].set(xlabel='Trajectories',ylabel='Inlet relative L2 error (%)',xticks=ticks)
    last=ordered[-1]
    root=OUT/'thesis_samples'/last['study']/last['profile']/f"N{last['N']}"
    with np.load(root/'arrays.npz') as data:
        axes[1].plot(data['y'],data['truth'],'k--',label='Archived reference')
        axes[1].plot(data['y'],data['inlet'],color='#275f83',label='Geometry-only')
    timed=np.genfromtxt((ROOT/last['timed_source']).with_name('inlet.csv'),names=True,delimiter=',')
    axes[1].plot(timed['y'],timed['estimated'],':',color='.45',label='Archived timed fit')
    axes[1].set(xlabel='y/H',ylabel='u / Uref',title=f"Last displayed stage · N={last['N']}")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f"{last['study']} · {last['profile']} · same archived trajectories")
    pdf.savefig(fig);plt.close(fig)


def spatial_page(pdf):
    from .spatial_prefix import COUNTS, EDGE, inputs

    fig,axes=plt.subplots(2,2,figsize=(9.6,6.3),layout='constrained')
    fig.get_layout_engine().set(w_pad=.14,h_pad=.10)
    output=OUT/'thesis_samples/spatial_prefix'
    rows=[]
    for j,seed in enumerate(SEEDS):
        original=BASE/f'seed_{seed}'/'nine30'
        _,_,sources=inputs(original/'observations')
        from .replay import held_out
        from ..geometry import curve_metrics

        final=output/f'seed_{seed}/N96'
        tested=read(final/'replay.json')
        assert tested['timing_used_only_after_fit'] and len(tested['tracks'])==8
        with np.load(final/'replays.npz',allow_pickle=False) as arrays:
            for i,(row,(q,_,horizon,path)) in enumerate(zip(
                    tested['tracks'],held_out(original/'observations'),strict=True),1):
                assert row['source']==str(path.relative_to(ROOT)) and row['horizon']==horizon
                replay=arrays[f'replay_{i}']
                np.testing.assert_array_equal(arrays[f'observed_{i}'],q)
                np.testing.assert_array_equal(replay[0,1:],q[0])
                if row['completed']:
                    assert len(replay)==129 and replay[-1,0]==horizon
                if len(replay)>=4:
                    for key,expected in curve_metrics(q,replay[:,1:],count=129).items():
                        np.testing.assert_allclose(row[key],expected,rtol=1e-12,atol=1e-15)
        completed=sum(r['completed'] for r in tested['tracks'])
        for n in COUNTS:
            folder=output/f'seed_{seed}'/f'N{n}'
            data=read(folder/'report.json')
            timed=original/'local_extended_fine'/f'N{n}'/'report.json'
            assert (data['seed'],data['N'],data['edge'])==(seed,n,EDGE)
            assert data['checkpoint_sha256']==SHA and data['sources']==sources[:n]
            assert data['timed_sha256']==digest(timed)
            assert data['timed_error']==read(timed)['error']
            with np.load(folder/'arrays.npz',allow_pickle=False) as a:
                assert np.all(np.diff(a['clock'],axis=1)>0)
                assert a['curves'][...,0].max()<=EDGE
                error=float(np.linalg.norm(a['inlet']-a['truth'])/np.linalg.norm(a['truth']))
                np.testing.assert_allclose(error,data['error'],rtol=1e-13)
                if n==96:
                    axes[j,1].plot(a['y'],a['truth'],'k--',label='Reference')
                    axes[j,1].plot(a['y'],a['inlet'],color='#275f83',label='Geometry + SOAP')
                    old=np.genfromtxt(timed.with_name('inlet.csv'),delimiter=',',names=True)
                    axes[j,1].plot(old['y'],old['estimated'],':',color='.45',label='Timed local spline')
            rows.append(dict(seed=seed,N=n,error_percent=100*error,timed_percent=100*data['timed_error'],
                             converged=data['selected']['converged'],iterations=data['selected']['iterations'],
                             complete_test_replay_N96=completed if n==96 else ''))
        subset=[r for r in rows if r['seed']==seed]
        axes[j,0].plot(COUNTS,[r['error_percent'] for r in subset],'-o',color='#275f83',label='Geometry + SOAP')
        axes[j,0].plot(COUNTS,[r['timed_percent'] for r in subset],'--s',color='.45',label='Timed local spline')
        axes[j,0].set(xlabel='Near-inlet traces',ylabel='Relative error (%)',title=f'Seed {seed}',xscale='log',xticks=COUNTS,xticklabels=COUNTS)
        axes[j,1].set(xlabel='y/H',ylabel='u / Uref',title=f'Seed {seed} · N=96')
        for ax in axes[j]:
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Nine-cycle spatial prefix · x/H ≤ 0.04 · page 130 sample')
    pdf.savefig(fig);plt.close(fig)
    table(output/'stage_metrics.csv',rows)
    return rows


def sensitivity_page(pdf):
    from .sensitivity import PROFILES, BUDGET

    rows=[]
    for seed in SEEDS:
        for name in PROFILES:
            for design in ('random','hybrid'):
                folder=OUT/'sensitivity'/f'seed_{seed}'/name/design
                r=read(folder/'report.json')
                source=OUT/f'seed_{seed}'/name/design/'N48'
                assert r['source_fit_sha256']==digest(source/'arrays.npz')
                assert r['checkpoint_sha256']==SHA and r['budget']==BUDGET
                baseline=read(source/'report.json')
                assert baseline['error']==r['baseline_error']
                with np.load(folder/'arrays.npz',allow_pickle=False) as a:
                    error=float(np.linalg.norm(a['inlet']-a['truth'])/np.linalg.norm(a['truth']))
                    np.testing.assert_allclose(error,r['error'],rtol=1e-13)
                    assert np.all(np.diff(a['clock'],axis=1)>0)
                rows.append(dict(seed=seed,profile=name,design=design,baseline_percent=100*r['baseline_error'],
                                 alternative_percent=100*error,base_cost=r['baseline_cost'],
                                 alternative_cost=r['selected']['value'],converged=r['selected']['converged']))
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.2),sharex=True,layout='constrained')
    fig.get_layout_engine().set(w_pad=.14,h_pad=.10)
    for ax,seed in zip(axes,SEEDS,strict=True):
        subset=[r for r in rows if r['seed']==seed]
        markers=np.arange(len(subset))
        a=[r['baseline_percent'] for r in subset]
        b=[r['alternative_percent'] for r in subset]
        ax.hlines(markers,a,b,color='.72')
        ax.plot(a,markers,'o',color='.45',label='Standard clock seed, 250')
        ax.plot(b,markers,'s',color='#275f83',label='Release clock seed, 700')
        ax.set(xlabel='Relative inlet error (%)',yticks=markers,
               yticklabels=[f"{r['profile']} · {r['design']}" for r in subset],title=f'Seed {seed}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Sensitivity to clock initialization and solver budget')
    pdf.savefig(fig);plt.close(fig)
    table(OUT/'sensitivity/stage_metrics.csv',rows)
    return rows


def main(studies):
    rows=[row for study in studies for row in audit(study)]
    destination=OUT/'thesis_samples'
    table(destination/'stage_metrics.csv',rows)
    plot=destination/'matched_thesis_samples.pdf'
    by_case=defaultdict(list)
    for row in rows:
        by_case[row['study'],row['profile']].append(row)
    with PdfPages(plot) as pdf:
        for group in by_case.values():
            draw(pdf,group)
        spatial=spatial_page(pdf)
        sensitivity=sensitivity_page(pdf)
    info={study:dict(stages=sum(row['study']==study for row in rows),
                     profiles=sum(key[0]==study for key in by_case))
          for study in studies}
    write_json(destination/'audit.json',dict(studies=info,spatial_prefix_stages=len(spatial),
               sensitivity_trials=len(sensitivity),pdf_sha256=digest(plot),checkpoint_sha256=SHA))
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies',nargs='+',choices=('adaptive','near'))
    main(parser.parse_args().studies)
