"""Check saved SOAP reconstructions and produce plots, comparisons and thesis text."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/soap-particle-mpl')
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run import BASE, NAMES, ROOT, REVISION, SHA

NICE_NAMES = ['Narrow beta','Asymmetric','Near-flat','Twin peaks','Shoulder',
              'Mixed scale','Three-cycle','Four-cycle','Nine-cycle']
PERSIAN = ['بتای باریک','نامتقارن','تقریباً تخت','دوقله‌ای','شانه‌دار',
           'چندمقیاسی','سه‌نوسانی','چهارنوسانی','نه‌نوسانی']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deliver(variant, names, control, titles):
    out = BASE/variant
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    rows, manifest, paired = [], [], []
    checked = dict(profiles=0,fits=0,replays=0,partial_replays=0)
    with fitz.open() as pack:
        for name,title in zip(names,titles):
            p = out/'progress'/f'{name}.json'
            progress = json.loads(p.read_text())
            assert progress['phase']=='completed' and progress['checkpoint_sha256']==SHA
            obs = out/'observations'/name
            for file,digest in progress['observation_sha256'].items():
                assert sha(obs/file)==digest
            evaluated = {v['N']:v for v in progress['evaluated']}
            curves = {}
            for n,stage in evaluated.items():
                folder = out/'fits'/name/f'N{n}'
                report = json.loads((folder/'report.json').read_text())
                assert report['checkpoint_sha256']==SHA
                source = report['initialization_source_N']
                assert source is None or source<n
                valid = [c for c in report['candidates'] if c['valid_inlet']]
                assert report['candidates'][report['selected_start']]['objective']==min(c['objective'] for c in valid)
                path = folder/'inlet.csv'
                curve = np.genfromtxt(path,delimiter=',',names=True)
                error = np.linalg.norm(curve['estimated']-curve['truth'])/np.linalg.norm(curve['truth'])
                np.testing.assert_allclose(error,stage['error'],rtol=1e-12)
                assert len(report['trajectories'])==n
                for k,track in enumerate(report['trajectories'],1):
                    replay = np.genfromtxt(folder/f'replay_{k}.csv',delimiter=',',names=True)
                    observed = np.genfromtxt(obs/f'track_{k}.csv',delimiter=',',names=True)[:len(replay)]
                    np.testing.assert_array_equal(replay['t'],observed['t'])
                    residual = np.column_stack([replay[c]-observed[c] for c in ['x','y']])
                    np.testing.assert_allclose(np.sqrt(np.mean(residual**2)),track['position_rmse_observed'],rtol=1e-10)
                    checked['replays']+=1
                    checked['partial_replays']+=not track['completed']
                assert stage['complete']==all(t['completed'] for t in report['trajectories'])
                checked['fits']+=1
                curves[n]=(curve,path)
                old = BASE/control/'fits'/name/f'N{n}/report.json'
                if old.exists():
                    previous = json.loads(old.read_text())
                    paired.append(dict(profile=name,N=n,old_percent=100*previous['inlet_relative_l2'],
                                       soap_percent=100*error,
                                       old_complete=previous['status']=='completed',soap_complete=stage['complete']))
            checked['profiles']+=1
            counts = progress['display_counts']
            end = progress['final_count']
            assert counts[0]==2 and counts[-1]==end
            fig,axes = plt.subplots(2,3,figsize=(6.27,4.45),sharex=True,sharey=True)
            inputs = {}
            for ax,n in zip(axes.flat,counts):
                d,path = curves[n]
                ax.plot(d['truth'],d['y'],'k-',lw=1.15)
                inputs[str(path.relative_to(ROOT))]=sha(path)
                old = BASE/control/'fits'/name/f'N{n}/inlet.csv'
                if old.exists():
                    previous = np.genfromtxt(old,delimiter=',',names=True)
                    np.testing.assert_allclose(previous['truth'],d['truth'],rtol=1e-12,atol=1e-12)
                    ax.plot(previous['estimated'],d['y'],color='#173f59',ls=':',lw=1.15)
                    inputs[str(old.relative_to(ROOT))]=sha(old)
                ax.plot(d['estimated'],d['y'],color='#bb2888',ls='--',lw=1.2)
                suffix = '*' if not evaluated[n]['complete'] else ''
                ax.set(xlabel=r'$u/U_{ref}$',ylabel='$y/H$',ylim=(0,1),
                       title=f"N={n} | SOAP {100*evaluated[n]['error']:.2f}%{suffix}")
                ax.grid(alpha=.15)
            for ax in list(axes.flat)[len(counts):]:
                ax.axis('off')
            fig.legend(handles=[Line2D([],[],color='k',label='Reference'),
                                Line2D([],[],color='#173f59',ls=':',label='94k when available'),
                                Line2D([],[],color='#bb2888',ls='--',label='314k SOAP')],
                       loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025))
            fig.suptitle(title,fontsize=10)
            fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.86,wspace=.28,hspace=.65)
            if any(not evaluated[n]['complete'] for n in counts):
                fig.text(.5,.016,'* At least one independent replay ended early',ha='center',fontsize=7)
            target = figures/f'{name}.pdf'
            fig.savefig(target,metadata={'CreationDate':None,'ModDate':None})
            plt.close(fig)
            with fitz.open(target) as page:
                pack.insert_pdf(page)
            found = [n for n,s in evaluated.items() if s['complete'] and s['error']<=.05]
            rows.append(dict(profile=name,display_N=','.join(map(str,counts)),search_stop_N=progress['search_stop_N'],
                             final_N=end,final_error_percent=100*evaluated[end]['error'],
                             first_tested_below_5_N=min(found,default=None),
                             final_replay_complete=evaluated[end]['complete'],reason=progress['stopping_reason']))
            manifest.append(dict(profile=name,sha256=sha(target),source_revision=REVISION,checkpoint_sha256=SHA,
                                 observation_sha256=progress['observation_sha256'],inputs_sha256=inputs))
        target = figures/'all_profiles.pdf'
        temporary = figures/'all_profiles.tmp.pdf'
        pack.save(temporary)
        temporary.replace(target)
    for filename,data in [('summary.csv',rows),('same_count_comparison.csv',paired)]:
        if data:
            with (out/filename).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    (out/'audit.json').write_text(json.dumps({'passed':True,**checked},indent=2)+'\n')
    (out/'figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(variant,checked)
    return rows,paired,checked


def thesis(rows, focused=None, focus_variant='soap_314k_near'):
    text = r'''\FloatBarrier
\par\medskip
\noindent\textbf{تکرار مطالعه با وزن جدید اپراتور جریان.}\par\smallskip
اپراتور بازآموزی‌شده با وزن گام $314000$ در برازش استفاده شد.
هر نه پروفایل و مسیرهای مشاهده‌شده با آزمون قبلی مشترک بودند.
رهاسازی‌ها شامل مکان، تندی و جهت‌های متنوع و تصادفی بودند.
تعداد ذرات از دو شروع شد و تطبیقی افزایش یافت.
نمودارها پنج یا شش برازش واقعی را نشان می‌دهند.
منحنی نقطه‌چین فقط برای تعدادهای مشترکِ آزموده‌شده رسم شده است.
آستانهٔ خطای پروفایل $5\%$ و سقف پیمایش $24$ ذره بود.
برای توقف زودهنگام، چند مرحلهٔ بیشتر به نمایش افزوده شد.
حقیقت ورودی برای ارزیابی و انتخاب تعداد استفاده شد.
پارامترهای اولیهٔ ذرات در برازش معلوم بودند.

'''
    for row,label in zip(rows,PERSIAN):
        name = row['profile']
        text += f"شکل~\\ref{{fig:soap-particle-{name}}} بازسازی ورودی {label} را نشان می‌دهد.\n"
        text += f"خطای مرحلهٔ پایانی با ${row['final_N']}$ ذره، ${row['final_error_percent']:.2f}\\%$ بود.\n"
        hit = row['first_tested_below_5_N']
        if hit is not None and hit != row['final_N']:
            text += f"نخستین تعداد موفق آزموده‌شده ${hit}$ ذره بود.\n"
        if hit is None:
            text += 'در تعدادهای آزموده‌شده، خطا به آستانهٔ پنج درصد نرسید.\n'
        if not row['final_replay_complete']:
            text += 'در مرحلهٔ پایانی دست‌کم یک بازپخش مستقل زودتر پایان یافت.\n'
        text += figure('soap_314k',name,'soap-particle-'+name,
                       f'بازسازی ورودی {label} با وزن جدید و افزایش تعداد ذرات.')
    if focused is not None:
        text += r'''\par\medskip
\noindent\textbf{تکرار آزمون رهاسازی نزدیک ورودی.}\par\smallskip
شرایط رهاسازی و زمان‌های مشاهده با آزمایش هدفمند قبلی مشترک بودند.
ضریب زبری و سقف تکرار همان تنظیمات آزمون هدفمند قبلی بودند.
شکل~\ref{fig:soap-particle-near} روند بازسازی پروفایل نه‌نوسانی را نشان می‌دهد.
'''
        text += f"خطای مرحلهٔ پایانی با ${focused['final_N']}$ ذره، ${focused['final_error_percent']:.2f}\\%$ بود.\n"
        if focus_variant.endswith('_extended'):
            text += 'در ادامه، رهاسازی‌های بیشتری با همان الگوی نزدیک ورودی افزوده شدند.\n'
            text += 'مسیرهای ۲۴ ذرهٔ نخست در این گسترش یکسان ماندند.\n'
        text += figure(focus_variant,'nine30','soap-particle-near',
                       'بازسازی ورودی نه‌نوسانی با رهاسازی هدفمند و وزن جدید.')
    path = ROOT/'chapters/soap-particle-results.tex'
    path.write_text(text)
    body = ROOT/'chapters/results.tex'
    source = body.read_text()
    marker = r'\input{chapters/soap-particle-results}'
    if marker not in source:
        anchor = r'\input{chapters/adaptive-particle-results}'
        assert source.count(anchor)==1
        body.write_text(source.replace(anchor,anchor+'\n'+marker))


def figure(variant,name,label,caption):
    return (r'\begin{figure}[htbp]'+'\n'+r'\centering'+'\n'
            +r'\includegraphics[width=\textwidth]{'+f'New_Adaptive_Particle_Results/{variant}/figures/{name}.pdf'+'}\n'
            +r'\caption{'+caption+'}\n'+r'\label{fig:'+label+'}\n'
            +r'\end{figure}'+'\n'+r'\FloatBarrier'+'\n')


if __name__=='__main__':
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    rows,paired,audit = deliver('soap_314k',NAMES,'continuation',NICE_NAMES)
    focused = None
    progress = BASE/'soap_314k_near/progress/nine30.json'
    if progress.exists() and json.loads(progress.read_text())['phase']=='completed':
        focused = deliver('soap_314k_near',['nine30'],'near_inlet',['Nine-cycle, near-inlet launches'])[0][0]
    variant = 'soap_314k_near_extended'
    extended = BASE/variant/'progress/nine30.json'
    if extended.exists() and json.loads(extended.read_text())['phase']=='completed':
        focused = deliver(variant,['nine30'],'near_inlet',['Nine-cycle, extended near-inlet launches'])[0][0]
        thesis(rows,focused,focus_variant=variant)
    else:
        thesis(rows,focused)
