"""Audit the inverse redesign and render paired six-panel profile comparisons."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/inverse-rewrite-mpl')
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run import ARCHIVE, CHECKPOINT_SHA, COUNTS, NAMES, OUT, ROOT

PRIMARY = 39421
REPLICATE = 53129
LABELS = ['Narrow beta','Asymmetric','Near-flat','Twin peaks','Shoulder',
          'Mixed scale','Three cycles','Four cycles','Nine cycles']
DESIGN_LABELS = {'recorded':'Recorded launches','random':'Random launches','hybrid':'Random + near inlet'}
COLORS = {'random':'#8461a0','hybrid':'#007c75'}


def load(name,seed,design):
    folder = OUT/f'seed_{seed}'/name/design
    example = json.loads((folder/'summary.json').read_text())
    assert example['checkpoint']==CHECKPOINT_SHA
    assert tuple(r['N'] for r in example['stages'])==COUNTS[design]
    for replay in example['final_replay']:
        dat = np.genfromtxt(folder/f"replay_{replay['index']:02d}.csv",delimiter=',',names=True)
        diff = np.column_stack([dat['x']-dat['x_observed'],dat['y']-dat['y_observed']])
        np.testing.assert_allclose(np.sqrt(np.mean(diff**2)),replay['coordinate_rmse'],rtol=1e-10)
        assert len(dat)==replay['sample_count']
    return folder,example


def curve(folder,n,report):
    p = folder/f'N{n}'/'inlet.csv'
    data = np.genfromtxt(p,delimiter=',',names=True)
    np.testing.assert_allclose(np.linalg.norm(data['estimated']-data['truth'])/np.linalg.norm(data['truth']),
                               report['error'],rtol=1e-11)
    assert np.min(data['estimated'])>=-1e-8
    eligible = [c for c in report['candidates'] if c['positive']]
    assert report['selected']['validation_score']==min(c['validation_score'] for c in eligible)
    return data


def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render():
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    results,paired,rollout_rows = [],[],[]
    fit_count = 0
    with fitz.open() as book:
        for name,title in zip(NAMES,LABELS):
            compatible = {}
            for seed in (PRIMARY,REPLICATE):
                for design in COUNTS:
                    if not (OUT/f'seed_{seed}'/name/design/'summary.json').exists():
                        continue
                    folder,data = load(name,seed,design)
                    final = data['final_replay']
                    test = [v for v in final if v['subset']=='test']
                    rollout_rows.append(dict(profile=name,seed=seed,design=design,
                        total=len(final),complete=sum(v['complete'] for v in final),test_complete=sum(v['complete'] for v in test),
                        test_rmse=np.sqrt(np.mean([v['coordinate_rmse']**2 for v in test]))))
                    for row in data['stages']:
                        n = row['N']
                        values = curve(folder,n,row)
                        if seed==PRIMARY:
                            compatible[design,n] = values
                        results.append(dict(profile=name,seed=seed,design=design,N=n,
                            error_percent=100*row['error'],validation_score=row['selected']['validation_score'],
                            selected=row['selected']['label'],converged=row['selected']['converged']))
                        if design=='recorded':
                            old_path = ARCHIVE/f'soap_314k/fits/{name}/N{n}/report.json'
                            if old_path.is_file():
                                old = json.loads(old_path.read_text())
                                assert old['checkpoint_sha256']==CHECKPOINT_SHA
                                paired.append(dict(profile=name,N=n,old_error_percent=100*old['inlet_relative_l2'],
                                                   rewritten_error_percent=100*row['error']))
                        fit_count += 1
            fig,axs = plt.subplots(2,3,figsize=(6.27,4.4),sharex=True,sharey=True)
            for n,ax in zip(COUNTS['random'],axs.flat):
                data = compatible['random',n]
                ax.plot(data['truth'],data['y'],'k-',lw=1)
                errors = {}
                for design in ('random','hybrid'):
                    d = compatible[design,n]
                    np.testing.assert_array_equal(d['truth'],data['truth'])
                    errors[design] = 100*np.linalg.norm(d['estimated']-d['truth'])/np.linalg.norm(d['truth'])
                    ax.plot(d['estimated'],d['y'],color=COLORS[design],ls='--' if design=='random' else '-.',lw=1.2)
                if n==2:
                    np.testing.assert_allclose(compatible['random',n]['estimated'],compatible['hybrid',n]['estimated'],
                                               atol=1e-12,rtol=1e-12)
                ax.set(xlabel=r'$u/U_{ref}$',ylabel='$y/H$',title=(
                    f"N={n} | {errors['random']:.1f} / {errors['hybrid']:.1f}%"),ylim=(0,1))
                ax.grid(alpha=.15)
            fig.legend(handles=[Line2D([],[],color='k',label='FEM reference'),
                        Line2D([],[],color=COLORS['random'],ls='--',label='Random'),
                        Line2D([],[],color=COLORS['hybrid'],ls='-.',label='Random + near inlet')],
                       loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
            fig.suptitle(title+' | random / mixed error',fontsize=10)
            fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.86,wspace=.29,hspace=.65)
            output = figures/f'{name}.pdf'
            fig.savefig(output,metadata={'CreationDate':None,'ModDate':None})
            plt.close(fig)
            with fitz.open(output) as page:
                book.insert_pdf(page)
        temp = figures/'all_profiles.tmp.pdf'
        book.save(temp)
        temp.replace(figures/'all_profiles.pdf')
    write_csv('stage_metrics.csv',results)
    write_csv('same_tracks_comparison.csv',paired)
    write_csv('independent_replay_metrics.csv',rollout_rows)
    print('Audited fits',fit_count,'independent replays',sum(r['total'] for r in rollout_rows))
    return results,paired,rollout_rows


if __name__=='__main__':
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    render()
