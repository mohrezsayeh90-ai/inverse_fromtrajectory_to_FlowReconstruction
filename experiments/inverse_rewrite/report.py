"""Render the local inlet result and add the experiment to the thesis."""
import csv
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/inverse-rewrite-mpl')

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from deliver import PRIMARY, REPLICATE
from run import NAMES, OUT, ROOT

FA = ['بتای باریک','نامتقارن','تقریباً تخت','دوقله‌ای','شانه‌دار',
      'چندمقیاسی','سه‌نوسانی','چهارنوسانی','نه‌نوسانی']
D = '$'


def rows(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def local_figure():
    results = {}
    for seed in (PRIMARY,REPLICATE):
        folder = OUT/f'seed_{seed}/nine30/local_extended_fine'
        data = json.loads((folder/'summary.json').read_text())
        results[seed] = {item['N']:item for item in data['stages']}
        for row in data['stages']:
            x = np.genfromtxt(folder/f"N{row['N']}/inlet.csv",delimiter=',',names=True)
            np.testing.assert_allclose(np.linalg.norm(x['estimated']-x['truth'])/np.linalg.norm(x['truth']),row['error'])
            assert row['selected']['validation_rmse']==min(v['validation_rmse'] for v in row['candidates'])
            assert np.min(x['estimated'])>=-1e-10
        for rep in data['full_domain_replay']:
            x = np.genfromtxt(folder/f"replay_{rep['test_track']:02d}.csv",delimiter=',',names=True)
            diff = np.column_stack([x[k]-x[k+'_observed'] for k in ('x','y')])
            np.testing.assert_allclose(np.sqrt(np.mean(diff**2)),rep['coordinate_rmse'])
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,3,figsize=(6.27,4.4),sharex=True,sharey=True)
    for n,ax in zip([2,4,20,31,64,96],axes.flat):
        errors = []
        for seed,color,style in [(PRIMARY,'#007c75','-.'),(REPLICATE,'#a04679','--')]:
            path = OUT/f'seed_{seed}/nine30/local_extended_fine/N{n}/inlet.csv'
            d = np.genfromtxt(path,delimiter=',',names=True)
            if not errors:
                ax.plot(d['truth'],d['y'],'k-',lw=1,label='FEM reference')
            ax.plot(d['estimated'],d['y'],style,color=color,lw=1.1,label=f'Seed {seed}')
            errors.append(100*results[seed][n]['error'])
        ax.set(xlabel=r'$u/U_{ref}$',ylabel='$y/H$',ylim=(0,1),title=f'N={n} | {errors[0]:.1f} / {errors[1]:.1f}%')
        ax.grid(alpha=.15)
    fig.suptitle('Nine cycles | early motion at the inlet',fontsize=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.86,wspace=.28,hspace=.65)
    fig.savefig(OUT/'figures/nine30_local_inlet.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    return results


def figure(file,label,caption):
    return (f'\\begin{{figure}}[htbp]\n\\centering\n'
            f'\\includegraphics[width=\\textwidth]{{New_Adaptive_Particle_Results/inverse_rewrite/figures/{file}.pdf}}\n'
            f'\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n\\FloatBarrier\n')


def thesis(final, local):
    text = r'''\FloatBarrier
\par\medskip
\noindent\textbf{بازنویسی وارون و مقایسهٔ طراحی رهاسازی.}\par\smallskip
برازش بهتر مسیر الزاماً خطای پروفایل ورودی را کاهش نمی‌دهد.
در نمونهٔ سه‌نوسانی، این تفاوت روی همان شش مسیر مشاهده شد.
وزن اپراتور جریان در این مقایسه ثابت ماند.
نمایش پروفایل، هموارسازی و وزن داده‌های نزدیک ورودی بازنویسی شدند.
نامزدها با شانزده مسیر اعتبارسنجی جداگانه رتبه‌بندی شدند.
مسیرهای آزمون نهایی نیز مستقل بودند.
دو طراحی با دو بذر تصادفی بررسی شدند.
دو رهاسازی نخست در هر دو طراحی یکسان بودند.
طراحی ترکیبی دو رهاسازی نزدیک ورودی به‌ازای هر رهاسازی تصادفی داشت.
ارتفاع رهاسازی‌های نزدیک ورودی در عرض کانال پخش شد.
شکل‌ها برای هر طراحی شش تعداد مشترک و واقعاً برازش‌شده دارند.
نتایج دو بذر در گزارش تکمیلی ثبت شده‌اند.

'''
    for name,label in zip(NAMES,FA):
        r,h = (final[PRIMARY,name,d] for d in ('random','hybrid'))
        text += f'شکل~\\ref{{fig:inverse-rewrite-{name}}} بازسازی ورودی {label} را مقایسه می‌کند.\n'
        text += f'در $48$ ذره، خطاهای تصادفی و ترکیبی {D}{r:.2f}\\%{D} و {D}{h:.2f}\\%{D} بودند.\n'
        text += figure(name,'inverse-rewrite-'+name,f'مقایسهٔ رهاسازی تصادفی و ترکیبی برای ورودی {label}.')
    text += r'''\par\medskip
\noindent\textbf{برآورد مستقیم ورودی از حرکت آغازین.}\par\smallskip
برای ورودی نه‌نوسانی، تنها بخش آغازین مسیرهای نزدیک ورودی بررسی شد.
معادلهٔ انتگرالی ذره برای برازش اسپلاین نامنفی به کار رفت.
تغییر سرعت در امتداد کانال با یک جملهٔ درجه‌دو تقریب زده شد.
انتخاب مدل با مسیرهای اعتبارسنجی و در پنجرهٔ مکانی مشترک انجام شد.
برای بررسی وضوح، اسپلاین‌های $45$ و $53$ ضریبی نیز آزموده شدند.
شکل~\ref{fig:inverse-rewrite-local} برآورد مستقیم ورودی را در دو تکرار نشان می‌دهد.
'''
    for n in (31,96):
        a,b = [100*local[s][n]['error'] for s in (PRIMARY,REPLICATE)]
        text += f'در {D}{n}{D} ذره، خطاهای دو تکرار {D}{a:.2f}\\%{D} و {D}{b:.2f}\\%{D} بودند.\n'
    text += 'بازپخش کامل کانال با اپراتور ثابت جداگانه ارزیابی شد.\n'
    text += figure('nine30_local_inlet','inverse-rewrite-local','بازسازی موضعی ورودی نه‌نوسانی با مسیرهای نزدیک ورودی.')
    (ROOT/'chapters/rewritten-inverse-results.tex').write_text(text)
    chapter = ROOT/'chapters/results.tex'
    content = chapter.read_text()
    marker = r'\input{chapters/rewritten-inverse-results}'
    anchor = r'\input{chapters/soap-particle-results}'
    if marker not in content:
        assert content.count(anchor)==1
        chapter.write_text(content.replace(anchor,anchor+'\n'+marker))


def report(final, local, metrics, paired, rollouts):
    cases = [(s,p) for s in (PRIMARY,REPLICATE) for p in NAMES]
    wins = sum(float(r['rewritten_error_percent'])<float(r['old_error_percent']) for r in paired)
    design_wins = sum(final[s,p,'hybrid']<final[s,p,'random'] for s,p in cases)
    hybrid_hits = sum(final[s,p,'hybrid']<=5 for s,p in cases)
    random_hits = sum(final[s,p,'random']<=5 for s,p in cases)
    lines = [
        '# بازنویسی وارون و طراحی رهاسازی نزدیک ورودی','',
        'جواب دو تعداد مختلف روی مجموعهٔ بزرگ‌تر و یکسان مسیرها با تابع هدف قبلی ارزیابی شد.','',
        '| ورودی | تعداد قبلی → بعدی | هزینهٔ قبلی → بعدی روی دادهٔ بزرگ‌تر | خطای ورودی قبلی → بعدی (%) |',
        '| --- | --- | --- | --- |',
    ]
    for case in json.loads((OUT/'old_regressions.json').read_text()):
        a,b = case['before'],case['after']
        lines.append(f"| {case['profile']} | {a['fitted_N']} → {b['fitted_N']} | {a['objective']:.3f} → {b['objective']:.3f} | {a['inlet_l2_percent']:.2f} → {b['inlet_l2_percent']:.2f} |")
    lines += [
        '',
        'در چهار نمونهٔ بررسی‌شده، کاهش هزینهٔ مسیر هم‌زمان با افزایش خطای پروفایل رخ داد. هر چهار جفت پاسخ نیز به سقف تکرار رسیده بودند.',
        'در بازنویسی، مدل‌های نرم و پرجزئیات براساس مسیرهای اعتبارسنجی رتبه‌بندی شدند. وزن SOAP در تمام برازش‌های سراسری ثابت ماند.',
        f'با همان داده‌های ثبت‌شدهٔ قبلی، بازنویسی در {wins} مورد از {len(paired)} مقایسهٔ هم‌تعداد خطای کمتری داشت. اثر بازنویسی به‌تنهایی یکسان نبود.',
        '',
        f'طراحی ترکیبی در {design_wins} مورد از ۱۸ مقایسهٔ ۴۸ذره‌ای خطای کمتری داشت. {hybrid_hits} آزمون ترکیبی و {random_hits} آزمون تصادفی به آستانهٔ ۵٪ رسیدند.',
        'دو ذرهٔ اول مشترک‌اند. سپس در طراحی ترکیبی دو مسیر نزدیک ورودی به‌ازای هر مسیر تصادفی افزوده شدند. شمارهای مشترک ۲، ۴، ۸، ۱۶، ۳۲ و ۴۸ بودند.',
        'طراحی نزدیک ورودی، فاصله از ورودی، پخش ارتفاع و سرعت اولیه را با هم تغییر می‌دهد. نتایج به دادهٔ مصنوعی، دو بذر و نویز این آزمایش مربوط‌اند.',
        '',
        '| ورودی | تصادفی بذر اول (%) | ترکیبی بذر اول (%) | تصادفی بذر دوم (%) | ترکیبی بذر دوم (%) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for p in NAMES:
        cells = [f'{final[s,p,d]:.2f}' for s in (PRIMARY,REPLICATE) for d in ('random','hybrid')]
        lines.append('| '+' | '.join([p]+cells)+' |')
    lines += [
        '', '## تخمین موضعی ورودی نه‌نوسانی', '',
        'اسپلاین نامنفی با حرکت آغازین ذره و تصحیح اختیاری درجه‌دو برازش شد. وزن SOAP در تخمین موضعی وارد نمی‌شود. بازپخش کل کانال با اپراتور ثابت جداگانه اجرا شد.',
        'نامزدها با خطای محوریِ مسیرهای اعتبارسنجی در پنجرهٔ مکانی مشترک انتخاب شدند. شکل تکمیلی شامل آزمون وضوح بیشتر با ۴۵ و ۵۳ ضریب است.',
        '', '| ذرات نزدیک ورودی | بذر اول (%) | بذر دوم (%) |', '| ---: | ---: | ---: |',
    ]
    for n in sorted(set(local[PRIMARY])&set(local[REPLICATE])):
        lines.append(f"| {n} | {100*local[PRIMARY][n]['error']:.2f} | {100*local[REPLICATE][n]['error']:.2f} |")
    lines += ['', '| حساسیت وضوح در ۹۶ ذره | بذر اول (%) | بذر دوم (%) |',
              '| --- | ---: | ---: |']
    for variant,title in [('local_extended','تنظیم اولیه تا ۳۷ ضریب'),('local_extended_fine','با نامزدهای ۴۵ و ۵۳ ضریبی')]:
        values = []
        for seed in (PRIMARY,REPLICATE):
            experiment = json.loads((OUT/f'seed_{seed}/nine30/{variant}/summary.json').read_text())
            endpoint = next(row for row in experiment['stages'] if row['N']==96)
            values.append(f"{100*endpoint['error']:.2f}")
        lines.append('| '+' | '.join([title]+values)+' |')
    with fitz.open(ROOT/'thesis.pdf') as doc:
        pages = len(doc)
    lines += [
        '',
        'افزایش تعداد، کاهش یکنواخت خطا را تضمین نکرد. پوشش مسیر، نویز، خطای مدل و تنظیم هموارسازی بر نتیجه اثر دارند.',
        f'ممیزی عددی شامل {len(metrics)} برازش سراسری، {sum(int(r["total"]) for r in rollouts)} بازپخش انتهایی و {sum(len(r) for r in local.values())} برازش موضعی بود.',
        f'پایان‌نامه با XeLaTeX و BibTeX ساخته شد و {pages} صفحه دارد. نسخه و نتایج پیشین حفظ شده‌اند.',
        '',
        'داده‌ها و مقایسه‌ها در پوشهٔ inverse_rewrite، در فایل‌های stage_metrics.csv، same_tracks_comparison.csv و independent_replay_metrics.csv قرار دارند.',
    ]
    (ROOT/'docs/INVERSE_REWRITE_FA.md').write_text('\n'.join(lines)+'\n')
    print('Same tracks wins',wins,len(paired),'mixed wins and below 5',design_wins,hybrid_hits,random_hits)
    print('Local',[(n,*[round(100*local[s][n]['error'],3) for s in (PRIMARY,REPLICATE)]) for n in [20,31,48,64,96]])


if __name__=='__main__':
    metrics,paired,rollouts = [rows(f) for f in ['stage_metrics.csv','same_tracks_comparison.csv','independent_replay_metrics.csv']]
    final = {(int(r['seed']),r['profile'],r['design']):float(r['error_percent'])
             for r in metrics if int(r['N'])==48 and r['design'] in ['random','hybrid']}
    assert len(final)==36
    local = local_figure()
    if '--thesis' in sys.argv:
        thesis(final,local)
    else:
        report(final,local,metrics,paired,rollouts)
