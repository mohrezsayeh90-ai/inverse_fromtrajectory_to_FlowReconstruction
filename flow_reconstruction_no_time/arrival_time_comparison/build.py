"""Post-fit duration comparison; no optimization and no old artifact changes."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
NAMES={'corners_beta4':'گوشه‌ای B','corners_beta5':'گوشه‌ای A','oblique_beta5':'پرتاب مایل'}

def main():
    cases=json.loads((BASE/'comparison_metrics.json').read_text())['cases'];rows=[];gaps=[]
    for case in cases:
        durations=[]
        for p in case['paths']:
            old=p['true_duration_for_posthoc_comparison_only'];new=p['inferred_duration']
            # Independently confirm actual elapsed duration in both saved ODEs.
            a=np.genfromtxt(BASE/'comparisons'/case['case']/f"timed_track_{p['particle']}_dense.csv",delimiter=',',names=True)
            b=np.genfromtxt(BASE/'experiments'/case['case']/'result'/f"track_{p['particle']}_forward.csv",delimiter=',',names=True)
            np.testing.assert_allclose([a['t'][-1]-a['t'][0],b['inferred_t'][-1]-b['inferred_t'][0]],[old,new],atol=1e-12)
            rows.append(dict(case=case['case'],particle=p['particle'],timed_duration=old,no_time_inferred_duration=new,
                             signed_difference=new-old,absolute_difference=abs(new-old),relative_difference_percent=100*(new-old)/old,
                             no_time_endpoint_mismatch_H=p['without_time']['endpoint_error_H']))
            durations.append((old,new))
        gaps.append(dict(case=case['case'],timed_particle2_minus_particle1=durations[1][0]-durations[0][0],no_time_particle2_minus_particle1=durations[1][1]-durations[0][1]))
    with (HERE/'durations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (HERE/'metrics.json').write_text(json.dumps(dict(time_unit='H/U_ref',durations=rows,interparticle_gaps=gaps,
        meaning='Elapsed duration of fitted observed curve; not an exact endpoint first-passage event. Timed duration is supplied, not independently inferred.'),indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained');x=np.arange(6);width=.36
    labels=[r['case'].replace('corners_beta4','Corner B').replace('corners_beta5','Corner A').replace('oblique_beta5','Oblique')+f" / P{r['particle']}" for r in rows]
    a=axes[0].bar(x-width/2,[r['timed_duration'] for r in rows],width,color='#ca7026',label='With time: supplied duration')
    b=axes[0].bar(x+width/2,[r['no_time_inferred_duration'] for r in rows],width,color='#11687c',label='Without time: inferred duration')
    for bars,padding in [(a,3),(b,14)]:axes[0].bar_label(bars,fmt='%.4f',fontsize=8,padding=padding)
    axes[0].set(title='Start-to-end duration',ylabel='Duration [H/U_ref]',xticks=x,xticklabels=labels);axes[0].tick_params(axis='x',rotation=40);axes[0].legend(fontsize=8);axes[0].margins(y=.25)
    values=[r['relative_difference_percent'] for r in rows]
    bars=axes[1].bar(x,values,color=['#11687c' if v<0 else '#ca7026' for v in values]);axes[1].bar_label(bars,fmt='%+.2f%%',fontsize=9,padding=4)
    axes[1].axhline(0,color='#555',lw=.8);axes[1].set(title='Duration difference between methods',ylabel='100 (T_without - T_with) / T_with [%]',xticks=x,xticklabels=labels);axes[1].tick_params(axis='x',rotation=40);axes[1].margins(y=.3)
    values=[r['no_time_particle2_minus_particle1'] for r in gaps];bars=axes[2].bar(np.arange(3),values,color='#11687c',width=.55,label='Without measured time')
    axes[2].bar_label(bars,fmt='%+.5f',fontsize=10,padding=4);axes[2].axhline(0,color='#ca7026',lw=1.5,ls='--',label='With time: gap = 0 in these cases')
    axes[2].set(title='Difference between the two particles',ylabel='T_particle2 - T_particle1 [H/U_ref]',xticks=np.arange(3),xticklabels=['Corner B','Corner A','Oblique']);axes[2].margins(y=.3);axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('PI-CFO | Motion duration with and without measured timestamps',fontsize=18,fontweight='bold')
    fig.savefig(HERE/'duration_comparison.png',dpi=320);fig.savefig(HERE/'duration_comparison.pdf');plt.close(fig)
    text=['# مقایسهٔ مدت حرکت دو روش','',
      'این بررسی فقط از نتایج ذخیره‌شده خوانده شده است؛ هیچ برازش مجدد یا تغییری در نتایج قبلی انجام نشده است.','',
      '**زمان‌ها بی‌بعد و برحسب H/U_ref هستند، نه ثانیه.** برای زمان فیزیکی باید مقدار جدول در H/U_ref فیزیکی ضرب شود.','',
      'در روش زمان‌دار، مدت از داده معلوم بوده و مستقلاً پیش‌بینی نشده است. در روش بدون زمان، مدت از ساعت پنهان به دست آمده است. منظور از رسیدن، پایان مسیر برازش‌شده است؛ ذره در حل مستقل الزاماً دقیقاً از نقطهٔ انتهایی مشاهده‌شده عبور نمی‌کند. این اعداد زمان اولین برخورد دقیق با مقصد نیستند. اختلاف مکانی انتها نیز در جدول آمده است.','',
      '![مقایسه زمان](duration_comparison.png)','', '[PDF برداری](duration_comparison.pdf) · [CSV اعداد](durations.csv)','',
      '| مثال | ذره | مدت روش با زمان | مدت روش بدون زمان | اختلاف علامت‌دار | اختلاف نسبی | اختلاف مکانی انتها، درصد H |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {NAMES[r['case']]} | {r['particle']} | {r['timed_duration']:.6f} | {r['no_time_inferred_duration']:.6f} | {r['signed_difference']:+.6f} | {r['relative_difference_percent']:+.3f}٪ | {100*r['no_time_endpoint_mismatch_H']:.4f} |")
    text+=['','اختلاف منفی یعنی مدت تخمینی روش بدون زمان کوتاه‌تر بوده است.','',
      '## اختلاف مدت میان دو ذره','',
      'در این مثال‌ها زمان شروع هر دو ذره صفر است. هر ذره به انتهای مسیر خودش می‌رود؛ مقصد مشترک یا برخورد دو ذره مدنظر نیست. اختلاف زیر برابر مدت ذرهٔ دوم منهای ذرهٔ اول است.','',
      '| مثال | روش با زمان | روش بدون زمان |','|---|---:|---:|']
    for g in gaps:text.append(f"| {NAMES[g['case']]} | {g['timed_particle2_minus_particle1']:+.6f} | {g['no_time_particle2_minus_particle1']:+.6f} |")
    text+=['', 'در روش زمان‌دار، صفر بودن اختلاف دو ذره از پنجرهٔ مشاهدهٔ یکسان می‌آید و به معنی رسیدن واقعی هم‌زمان به مقصد مشترک نیست.','',
      'بازتولید از ریشهٔ مخزن: `python flow_reconstruction_no_time/arrival_time_comparison/build.py`']
    (HERE/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows,indent=2));print(json.dumps(gaps,indent=2))

if __name__=='__main__':main()
