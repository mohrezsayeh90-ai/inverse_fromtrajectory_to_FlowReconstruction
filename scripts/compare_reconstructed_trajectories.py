"""Compare input paths with fresh ODE paths in saved inferred flows; no refitting."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flow_reconstruction.io import load_fit,read_problem

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'flow_reconstruction/trajectory_comparisons'
CASES=[('development/trajectory_only_initial','Oblique launches / initial fit','پرتاب مایل، جواب اولیه'),
       ('fem_beta','Oblique launches / physics-regularized fit','پرتاب مایل، جواب با قید فیزیک'),
       ('corner_launch','Corner launches / original coordinates','پرتاب گوشه‌ای، جواب قبلی'),
       ('corner_launch_preconditioned','Corner launches / preconditioned fit','پرتاب گوشه‌ای، جواب پیش‌شرط‌سازی‌شده')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Protect every pre-existing tracked file, not just the trained weights.
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    protected={name:sha(ROOT/name) for name in files if name and not (ROOT/name).is_relative_to(OUT)}
    OUT.mkdir(parents=True,exist_ok=True);records=[];case_reports=[]
    for case_index,(source,title,label) in enumerate(CASES,1):
        directory=ROOT/'flow_reconstruction/examples'/source/'result'
        tracks,_,_=read_problem(directory/'problem.json')
        model,theta=load_fit(directory/'fitted_model.json',ROOT/'models/candidates/pressure_long/checkpoint.npz',allow_unvalidated=True)
        name=f'case_{case_index}';output=OUT/name;output.mkdir(exist_ok=True)
        figure=plt.figure(figsize=(14,12),layout='constrained');layout=figure.add_gridspec(3,2,height_ratios=[1,1,1])
        overview=figure.add_subplot(layout[0,:]);zooms=[figure.add_subplot(layout[1,j]) for j in range(2)]
        errors=[figure.add_subplot(layout[2,j]) for j in range(2)]
        component_fig,component_axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        local=[]
        for i,track in enumerate(tracks):
            # Start from the known launch state and query the saved inferred flow
            # along the particle's OWN evolving position, not the measured path.
            sim=model.simulate(theta,track);n=len(sim['time'])
            np.testing.assert_allclose(sim['time'],track.time[:n],atol=1e-12,rtol=0)
            observed=track.position[:n];prediction=sim['position'];difference=prediction-observed
            distance=np.linalg.norm(difference,axis=1)
            stored=np.genfromtxt(directory/f'track_{i+1}_forward.csv',delimiter=',',names=True)
            assert len(stored)==n,'Replayed simulation has different coverage'
            replay_difference=float(np.max(abs(prediction-np.column_stack([stored['x'],stored['y']]))))
            assert replay_difference<1e-8,'Saved inferred flow no longer reproduces its stored trajectory'
            row=dict(case=name,particle=track.name,observed_samples=len(track.time),predicted_samples=n,
                     complete_time_coverage=bool(sim['completed']),
                     coordinate_rmse_H=float(np.sqrt(np.mean(difference**2))),
                     position_2d_rmse_H=float(np.sqrt(np.mean(distance**2))),
                     position_2d_rmse_percent_H=float(100*np.sqrt(np.mean(distance**2))),
                     mean_distance_H=float(distance.mean()),max_distance_H=float(distance.max()),
                     final_distance_H=float(distance[-1]),replay_max_difference_H=replay_difference)
            local.append(row);records.append(row)
            np.savetxt(output/f'particle_{i+1}.csv',np.column_stack([sim['time'],observed,prediction,difference,distance,sim['velocity']]),
                       delimiter=',',header='t,x_input,y_input,x_ODE,y_ODE,dx,dy,distance,vx_ODE,vy_ODE',comments='')
            color=['#176ca4','#b9462f'][i]
            for axis in [overview,zooms[i]]:
                axis.plot(track.position[:,0],track.position[:,1],'.',color=color,ms=4,alpha=.65,label=track.name+' input')
                axis.plot(prediction[:,0],prediction[:,1],'-',color='#ee9c19',lw=1.6,label=track.name+' inferred-flow ODE')
                axis.scatter(*track.position[0],marker='s',color=color,s=30,zorder=5)
            worst=int(np.argmax(distance));center=(observed[worst]+prediction[worst])/2
            radius=max(.003,4*float(distance[worst]))
            zooms[i].set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                         title=f'{track.name}: zoom at largest mismatch, t={track.time[worst]:.3g}',xlabel='x/H',ylabel='y/H',aspect='equal')
            zooms[i].plot([observed[worst,0],prediction[worst,0]],[observed[worst,1],prediction[worst,1]],'k-',lw=2)
            zooms[i].scatter(*observed[worst],marker='o',facecolors='none',edgecolors='black',s=65,zorder=7)
            zooms[i].scatter(*prediction[worst],marker='x',color='black',s=55,zorder=7)
            errors[i].plot(sim['time'],distance,color=color,label='Euclidean distance')
            errors[i].axhline(row['position_2d_rmse_H'],color='black',ls='--',lw=1,label='2D position RMSE')
            errors[i].set(title=f"{track.name}: 2D RMSE={row['position_2d_rmse_H']:.6g} H",xlabel='t / (H/U_ref)',ylabel='position mismatch / H')
            errors[i].legend(fontsize=8);errors[i].grid(alpha=.2)
            for axis_index,coordinate in enumerate(['x','y']):
                axis=component_axes[axis_index,i]
                axis.plot(track.time,track.position[:,axis_index],'.',color=color,ms=3,label='Input positions')
                axis.plot(sim['time'],prediction[:,axis_index],color='#ee9c19',lw=1.4,label='Independent ODE in inferred flow')
                axis.set(title=track.name+': '+coordinate+'(t)',xlabel='t / (H/U_ref)',ylabel=coordinate+'/H');axis.legend(fontsize=8);axis.grid(alpha=.2)
        overview.set(xlim=(-.03,model.operator.config.length+.03),ylim=(-.03,1.03),aspect='equal',xlabel='x/H',ylabel='y/H',
                     title='Input trajectories overlaid with independently simulated trajectories')
        overview.legend(ncol=2,fontsize=9)
        figure.suptitle(title+'\nSaved inlet and flow unchanged; same known launches and Stokes drag',fontsize=14)
        figure.savefig(output/'overlay_and_error.png',dpi=170);figure.savefig(output/'overlay_and_error.pdf');plt.close(figure)
        component_fig.suptitle(title+' | input versus predicted positions at the same times')
        component_fig.savefig(output/'position_time_series.png',dpi=160);plt.close(component_fig)
        z,mean_speed,_=model.decode(theta);inlet=model.builder.inlet(z);ys=np.linspace(0,1,513)
        inlet_values=inlet.spline(ys)*float(mean_speed)
        np.savetxt(output/'inlet_used.csv',np.column_stack([ys,inlet_values]),delimiter=',',header='y,u_over_Uref',comments='')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.plot(ys,inlet_values,color='#176ca4');ax.set(xlabel='y/H',ylabel='inlet velocity / U_ref',
            title='Saved inferred inlet used for the particle simulation')
        ax.grid(alpha=.2);fig.savefig(output/'inlet_used.png',dpi=160);plt.close(fig)
        report=dict(source_fit=str((directory/'fitted_model.json').relative_to(ROOT)),
                    saved_fit_sha256=sha(directory/'fitted_model.json'),checkpoint_sha256=model.operator.sha256,
                    normalized_inlet_sha256=sha(directory/'inlet_normalized.csv'),mean_speed=float(model.decode(theta)[1]),
                    reynolds=float(model.decode(theta)[2]),refitting=False,forward_weights_modified=False,
                    comparison='Only observed/input positions versus independently integrated positions in the saved inferred flow',particles=local)
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        case_reports.append(dict(case=name,label=label,source=source,particles=local))
        print(json.dumps(dict(case=name,particles=local)),flush=True)
    changed=[name for name,digest in protected.items() if sha(ROOT/name)!=digest]
    assert not changed,changed
    (OUT/'preservation_check.json').write_text(json.dumps(dict(preexisting_tracked_files_checked=len(protected),
        preexisting_tracked_files_unchanged=True,changed_files=changed,model_refitted=False,reference_fields_read=False),indent=2)+'\n')
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    (OUT/'summary.json').write_text(json.dumps(case_reports,ensure_ascii=False,indent=2)+'\n')
    text=['# تطبیق مسیر ورودی با حرکت در جریان تخمین‌زده‌شده','','در این بخش فقط می‌سنجیم آیا جریانِ تخمین‌زده‌شده، ذره را تقریباً در همان مسیر داده‌شده حرکت می‌دهد. هیچ مقایسهٔ میدان یا ورودی با مرجع عددی در این بخش انجام نشده است.',
          '', 'جواب‌های ذخیره‌شده بدون بهینه‌سازی دوباره بارگذاری شده‌اند. برای هر ذره، معادلهٔ حرکت با همان مکان و سرعت اولیه و همان زمان پاسخ استوکس، مستقل دوباره حل شده است. سرعت سیال در مکانِ پیش‌بینی‌شدهٔ خود ذره ارزیابی می‌شود؛ مسیر مشاهده‌شده به حل ODE تحمیل نمی‌شود.',
          '', '**چهار جواب قبلی، شامل هشت مسیر ذره، مقایسه شده‌اند.** این‌ها چهار جواب برای دو مجموعهٔ ورودی‌اند؛ هشت آزمایش مستقل نیستند. مسیر ورودی شامل همان نویز موجود در دادهٔ ذخیره‌شده است.',
          '', '| جواب ذخیره‌شده | ذره | RMSE دوبعدی / H | RMSE دوبعدی، درصد H | بیشینهٔ فاصله / H | تمام زمان‌ها پوشش داده شد؟ |',
          '|---|---|---:|---:|---:|---|']
    for case in case_reports:
        for row in case['particles']:
            text.append(f"| [{case['label']}]({case['case']}/overlay_and_error.png) | {row['particle']} | {row['position_2d_rmse_H']:.7f} | {row['position_2d_rmse_percent_H']:.4f}٪ | {row['max_distance_H']:.7f} | {'بله' if row['complete_time_coverage'] else 'خیر'} |")
    text+=['','RMSE دوبعدی برابر `sqrt(mean(dx²+dy²))` است. در گزارش‌های قبلی RMSE مختصاتی `sqrt(mean([dx²,dy²]))` آمده بود؛ RMSE دوبعدی این جدول √۲ برابر آن است. هر دو تعریف در CSV ذخیره شده‌اند. درصدها نسبت به ارتفاع کانال H هستند، نه درصد فاصلهٔ طی‌شده.',
           '', 'خطوط نارنجی مسیر خروجی و نقطه‌ها مسیر ورودی‌اند. بزرگ‌نمایی در زمان بیشینهٔ اختلاف، خطای کوچک را قابل دیدن می‌کند. نمودار فاصله و نمودار x(t)،y(t) مقایسه در زمان یکسان را نشان می‌دهند.',
           '', 'این نتیجه میزان سازگاری همین جریانِ تخمینی با همین مسیرهای داده‌شده را نشان می‌دهد؛ تأیید یکتایی جریان یا دقت آن در همهٔ نقاط کانال نیست. مسیرها برای استنتاج استفاده شده‌اند و این مقایسه آزمون مسیرِ دیده‌نشده محسوب نمی‌شود.']
    for case in case_reports:
        name=case['case'];text+=['',f"## {case['label']}",'',f'![مسیرهای روی‌هم‌افتاده، بزرگ‌نمایی و خطای زمانی]({name}/overlay_and_error.png)','',
                                f'[مقایسهٔ x(t) و y(t)]({name}/position_time_series.png) · [پروفایل تخمینی استفاده‌شده]({name}/inlet_used.png) · [PDF]({name}/overlay_and_error.pdf) · [اعداد کامل]({name}/metrics.json)']
    text+=['', 'بازتولید از ریشهٔ مخزن:','','```bash','python -m scripts.compare_reconstructed_trajectories','```','',
           '[شاهد تغییرنکردن فایل‌های قبلی](preservation_check.json) · [CSV تمام مقایسه‌ها](summary.csv)']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
