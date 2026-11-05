"""Summarize actual completed experiments; never changes acceptance thresholds."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LABELS={'gaussian':'Gaussian / pooled','oseen':'Oseen / pooled','moments':'Oseen / moments','sensor':'Direct sensor branch','pressure':'Separate pressure/stream operators'}
COLORS={'gaussian':'#9b778c','oseen':'#527ca8','moments':'#e39432','sensor':'#008b8b','pressure':'#542b9e'}


def main():
    out=ROOT/'reports/experiments';out.mkdir(parents=True,exist_ok=True)
    summaries=[];histories={}
    for name in LABELS:
        p=out/name
        if (p/'training.jsonl').exists():
            records={}
            for line in (p/'training.jsonl').read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if 'step' in r:records[r['step']]=r
            histories[name]=records
        if (p/'metrics.json').exists():
            m=json.loads((p/'metrics.json').read_text())
            summaries.append(dict(name=name,step=m['step'],max_velocity_pct=100*max(r['velocity_relative_l2'] for r in m['cases']),
                max_pressure_pct=100*max(r['pressure_relative_l2'] for r in m['cases']),
                max_momentum_rms=max(r['momentum_rms'] for r in m['cases']),
                accepted=m['all_gates_passed'],checkpoint_sha256=m['checkpoint_sha256'],gates=m['gates']))
    names=['asymmetric','beta','flat','nine']
    fig,axes=plt.subplots(4,2,figsize=(14,14),layout='constrained')
    for row,case in enumerate(names):
        for col,(key,label,limit) in enumerate([('velocity_relative_l2','Velocity vector relative L2 (%)',5),('pressure_relative_l2','Pressure relative L2 (%)',10)]):
            ax=axes[row,col]
            for name,history in histories.items():
                points=[]
                for step,r in sorted(history.items()):
                    for c in r.get('reference_validation',{}).get('cases',[]):
                        if c['name']==case:points.append((step,100*c[key]))
                if points:
                    x,y=np.array(points).T;ax.plot(x,y,label=LABELS[name],color=COLORS[name],lw=1.3,alpha=.85)
            ax.axhline(limit,color='#b43d3d',ls='--',lw=1)
            ax.set(title=case,xlabel='Global optimizer step',ylabel=label,xscale='symlog');ax.set_xscale('symlog',linthresh=1000);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Measured reference errors during physics-only training\nRepeated validation; no CFD response labels in the objective',fontsize=15,fontweight='bold')
    fig.savefig(out/'reference_learning_curves.png',dpi=150);fig.savefig(out/'reference_learning_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,case in zip(axes.flat,['poiseuille','nine_oscillations','beta_5_5','flat']):
        for name,history in histories.items():
            points=[(s,c['momentum_rms']) for s,r in sorted(history.items()) for c in r.get('cases',[]) if c['name']==case]
            if points:
                x,y=np.array(points).T;ax.plot(x,np.maximum(y,1e-10),label=LABELS[name],color=COLORS[name],lw=1.2,alpha=.8)
        ax.set(title=case,xlabel='Global optimizer step',ylabel='Unscaled NS momentum RMS',yscale='log')
        ax.set_xscale('symlog',linthresh=1000);ax.axhline(.05,color='#b43d3d',ls='--');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Independent physical diagnostics: successful execution is not acceptance',fontsize=15)
    fig.savefig(out/'momentum_learning_curves.png',dpi=150);plt.close(fig)
    with (out/'summary.csv').open('w',newline='') as f:
        columns=['name','step','max_velocity_pct','max_pressure_pct','max_momentum_rms','accepted','checkpoint_sha256']
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for s in summaries:writer.writerow({k:s[k] for k in columns})
    text=['# نتایج واقعی آزمایش‌های اپراتور جدید','','تمام اعداد این صفحه از checkpoint واقعی و مرجع مستقل آمده‌اند. این‌ها validation تکرارشونده‌اند؛ آزمون کور نهایی نیستند. پاسخ CFD وارد objective آموزش نشده است.','',
          '| آزمایش | گام | بیشینهٔ خطای سرعت ٪ | بیشینهٔ خطای فشار ٪ | بیشینهٔ RMS مومنتوم | تمام معیارها پاس؟ |','|---|---:|---:|---:|---:|---|']
    for s in summaries:text.append(f"| [{LABELS[s['name']]}]({s['name']}/metrics.json) | {s['step']} | {s['max_velocity_pct']:.3f} | {s['max_pressure_pct']:.3f} | {s['max_momentum_rms']:.4f} | {s['accepted']} |")
    text+=['','![خطای مرجع در آموزش](reference_learning_curves.png)','','![باقی‌ماندهٔ معادلات](momentum_learning_curves.png)','',
           'آزمایش‌ها فقط در یک عامل تغییر نکرده‌اند؛ از این جدول نمی‌توان سهم مستقل هر تغییر معماری، batch یا مقیاس‌بندی را استنتاج کرد. قیود ورودی و جرم در همهٔ نسخه‌های جدید برقرارند، اما این به‌تنهایی دقت میدان را تضمین نمی‌کند.',
           '', 'برای هر آزمایش، `training.jsonl`، متادیتای ورودی‌ها، SHA256 سورس و معیارهای رد/قبول نگه داشته شده‌اند. وزن‌های آزمایشی در `models/candidates/` قرار دارند؛ نامزدِ ردشده، مدل آمادهٔ استفاده نیست.']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
