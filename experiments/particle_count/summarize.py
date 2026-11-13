"""Audit, aggregate and plot all fifty completed particle-count experiments."""
from pathlib import Path
import sys,json,csv,hashlib,importlib.metadata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.particle_count.campaign import OUT,NAMES,LABELS,LAUNCHES,SHA,CP,dump,sha,profile

def main():
 metadata={n:json.loads((OUT/'reference'/n/'metadata.json').read_text()) for n in NAMES}
 ordered=sorted(NAMES,key=lambda n:metadata[n]['roughness']);allrows=[];matrix=[]
 for name in ordered:
  errors=[]
  for n in range(2,7):
   folder=OUT/'fits'/name/f'N{n}';r=json.loads((folder/'report.json').read_text())
   assert r['status'] in ['completed','partial_replay'],r
   assert r['checkpoint_sha256']==SHA and r['checkpoint_unchanged']
   assert r['config']['starts']==2 and r['config']['max_iterations']==200 and r['config']['profile_points']==65
   valid=[(i,c) for i,c in enumerate(r['candidates']) if c['valid_inlet']]
   assert r['selected_start']==min(valid,key=lambda v:v[1]['objective'])[0]
   a=np.genfromtxt(folder/'inlet.csv',delimiter=',',names=True)
   np.testing.assert_allclose(a['truth'],profile(name).spline(a['y']),atol=1e-12)
   error=np.linalg.norm(a['estimated']-a['truth'])/np.linalg.norm(a['truth'])
   np.testing.assert_allclose(error,r['inlet_relative_l2'],rtol=1e-12)
   replay_errors=[]
   for i in range(1,n+1):
    q=np.genfromtxt(folder/f'replay_{i}.csv',delimiter=',',names=True);obs=np.genfromtxt(OUT/'observations'/name/f'track_{i}.csv',delimiter=',',names=True)
    np.testing.assert_array_equal([q['x'][0],q['y'][0]],LAUNCHES[i-1][0])
    delta=np.column_stack([q['x'],q['y']])-np.column_stack([obs['x'],obs['y']])[:len(q)]
    rmse=np.sqrt(np.mean(delta**2));np.testing.assert_allclose(rmse,r['trajectories'][i-1]['position_rmse_observed'],rtol=1e-10,atol=1e-14)
    replay_errors.append(float(rmse))
   sel=r['candidates'][r['selected_start']]
   allrows.append(dict(profile=name,label=LABELS[name],roughness=metadata[name]['roughness'],N=n,inlet_error_percent=100*error,
    shape_error_percent=100*r['shape_relative_l2'],mean_speed_error=r['mean_speed_error'],selected_start=r['selected_start'],
    optimizer_converged=sel['optimizer_success'],iterations=sel['iterations'],optimizer_reason=sel['reason'],
    replay_complete=all(t['completed'] for t in r['trajectories']),maximum_track_rmse=max(replay_errors),
    candidate_min_inlet_percent=100*min(c['inlet_relative_l2_evaluation_only'] for c in r['candidates']),candidate_max_inlet_percent=100*max(c['inlet_relative_l2_evaluation_only'] for c in r['candidates']),seconds=r['seconds_including_compile']))
   errors.append(100*error)
  matrix.append(errors)
 assert sha(CP)==SHA
 matrix=np.array(matrix)
 summary=dict(profile_order=ordered,inlet_errors_percent=matrix.tolist(),N=[2,3,4,5,6],cases=len(allrows),
  N6_better_than_N2=int(np.sum(matrix[:,-1]<matrix[:,0])),N6_worse_than_N2=int(np.sum(matrix[:,-1]>matrix[:,0])),
  strictly_monotonic_profiles=[ordered[i] for i in range(10) if np.all(np.diff(matrix[i])<0)],
  improving_steps=int(np.sum(np.diff(matrix)<0)),total_steps=40,median_error_percent_by_N=np.median(matrix,axis=0).tolist(),
  mean_error_percent_by_N=matrix.mean(0).tolist(),median_relative_reduction_2_to_6_percent=float(np.median(100*(1-matrix[:,-1]/matrix[:,0]))),
  selected_optimizer_converged=sum(r['optimizer_converged'] for r in allrows),complete_replays=sum(r['replay_complete'] for r in allrows),
  checkpoint_sha256=SHA,roughness_definition='sqrt(integral_0^1 f_second_derivative(y)^2 dy); proxy for variation, not universal complexity',rows=allrows)
 dump(OUT/'summary.json',summary)
 with (OUT/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]),lineterminator='\n');w.writeheader();w.writerows(allrows)
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
 colors=plt.cm.viridis(np.linspace(.05,.9,5))
 fig,axes=plt.subplots(2,5,figsize=(15,6.3),sharex=True)
 for i,(name,ax) in enumerate(zip(ordered,axes.flat)):
  for j,n in enumerate(range(2,7)):
   a=np.genfromtxt(OUT/'fits'/name/f'N{n}/inlet.csv',delimiter=',',names=True)
   ax.plot(a['y'],a['estimated'],color=colors[j],lw=1.15,label=f'N={n}')
  ax.plot(a['y'],a['truth'],'k--',lw=1.8,label='FEM/analytic truth',zorder=10)
  ax.set_title(f'{i+1}. {name}',fontsize=9);ax.set(xlabel='y / H',ylabel='Inlet velocity / U_ref',xlim=(0,1));ax.grid(alpha=.18)
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=6,frameon=False)
 fig.tight_layout(rect=(0,.07,1,1));fig.savefig(figdir/'reconstructed_inlets.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(figdir/'reconstructed_inlets.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.45,1]})
 im=axes[0].imshow(matrix,aspect='auto',cmap='magma_r');axes[0].set(xticks=np.arange(5),xticklabels=range(2,7),yticks=np.arange(10),yticklabels=[f'{i+1}. {n}' for i,n in enumerate(ordered)],xlabel='Number of observed particles',title='Inlet relative L2 error (%)')
 for i in range(10):
  for j in range(5):axes[0].text(j,i,(f'{matrix[i,j]:.3f}' if matrix[i,j]<.1 else f'{matrix[i,j]:.1f}'),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.6 else 'black',fontsize=8)
 fig.colorbar(im,ax=axes[0],shrink=.8,label='%')
 for i,n in enumerate(ordered):axes[1].plot(range(2,7),matrix[i],'-o',lw=1,ms=3,label=str(i+1))
 axes[1].plot(range(2,7),np.median(matrix,axis=0),'k--',lw=2.5,label='Median');axes[1].set(xlabel='Number of observed particles',ylabel='Inlet error (%)',xticks=range(2,7));axes[1].grid(alpha=.2);axes[1].legend(ncol=3,fontsize=7)
 fig.tight_layout();fig.savefig(figdir/'inlet_errors.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(figdir/'inlet_errors.png',dpi=180);plt.close(fig)
 # Separate truth profiles and the nested launch design for auditability.
 fig,axes=plt.subplots(1,2,figsize=(11,3.3))
 for i,name in enumerate(ordered):
  y=np.linspace(0,1,2049);axes[0].plot(y,profile(name).spline(y),label=f'{i+1}: {name}',lw=1)
 axes[0].set(xlabel='y / H',ylabel='True inlet / U_ref',title='Reference inlets, Re=30');axes[0].legend(fontsize=6,ncol=2)
 for i in range(1,7):
  q=np.genfromtxt(OUT/'observations'/ordered[-1]/f'clean_{i}.csv',delimiter=',',names=True)
  line,=axes[1].plot(q['x'],q['y'],lw=1,label=f'Particle {i}');axes[1].plot(q['x'][0],q['y'][0],'s',ms=3,color=line.get_color())
 axes[1].set(xlim=(0,5),ylim=(0,1),aspect='equal',xlabel='x / H',ylabel='y / H',title='Nested tracks (most oscillatory inlet)');axes[1].legend(fontsize=6,ncol=3)
 fig.tight_layout();fig.savefig(figdir/'design.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(figdir/'design.png',dpi=180);plt.close(fig)
 # Also regenerate the simple-to-detailed reference gallery.
 fig,axes=plt.subplots(2,5,figsize=(12,4.8),sharex=True)
 for i,(name,ax) in enumerate(zip(ordered,axes.flat)):
  y=np.linspace(0,1,2049);ax.plot(y,profile(name).spline(y),color='#24588b',lw=1.3)
  ax.set(title=f'{i+1}. {name}',xlabel='y/H',ylabel='u/U_ref');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(figdir/'reference_profiles.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(figdir/'reference_profiles.png',dpi=160);plt.close(fig)
 # Scientific metrics are reported as screening, without truth-based gates.
 lines=['# اثر تعداد ذرات بر بازسازی پروفایل ورودی','', 'رینولدز همهٔ پروفایل‌ها ۳۰ و سرعت میانگین حقیقی یک است.',
 'هر ستون از همان ذرات ستون قبلی به‌اضافهٔ یک ذره استفاده می‌کند.',
 'ترتیب برحسب RMS مشتق دوم ورودی است، نه فقط تعداد قله‌ها.',
 'خطا، نرم نسبی پروفایل کامل در واحد سرعت مرجع است.','', '| پروفایل | ۲ ذره | ۳ ذره | ۴ ذره | ۵ ذره | ۶ ذره |','|---|---:|---:|---:|---:|---:|']
 for name,values in zip(ordered,matrix):lines.append('| '+LABELS[name]+' | '+' | '.join((f'{v:.4f}%' if v<.01 else f'{v:.2f}%') for v in values)+' |')
 lines+=['',f'در {summary["N6_better_than_N2"]} پروفایل، شش ذره بهتر از دو ذره بود.',f'از ۴۰ افزایش تک‌ذره‌ای، {summary["improving_steps"]} مورد خطای کمتری داشت.',f'تعداد پروفایل‌های با بهبود کاملاً یکنواخت: {len(summary["strictly_monotonic_profiles"])}.',
 f'میانهٔ خطا از {np.median(matrix[:,0]):.2f}% با دو ذره به {np.median(matrix[:,-1]):.2f}% با شش ذره رسید.',
 f'تعداد جواب‌های منتخب با همگرایی تأییدشدهٔ بهینه‌ساز: {summary["selected_optimizer_converged"]} از ۵۰.',
 'بودجهٔ همهٔ حالت‌ها دو شروع و حداکثر ۲۰۰ تکرار برای هر شروع بود.',
 'بنابراین نتایج، غربالگری با بودجهٔ محدود هستند؛ کمینهٔ بهینه‌سازی تضمین نشده است.',
 '', '![خطاهای پروفایل](figures/inlet_errors.png)','![ورودی‌های بازسازی‌شده](figures/reconstructed_inlets.png)','',
 'این یک آزمون عددی مشروط به مدل منجمد و چیدمان معلوم است.',
 'افزایش تعداد ذرات، تعداد نمونه‌ها و پوشش مکانی را هم‌زمان افزایش می‌دهد.',
 'هر ذره ۸۱ مشاهده دارد؛ دو شروع، عدم‌قطعیت آماری را تعیین نمی‌کنند.',
 'نویز ثابت، همان تحقق نویز را در همهٔ پیشوندها حفظ می‌کند.',
 'پروفایل حقیقی فقط برای ارزیابی نهایی استفاده شده است.',
 'پروفیل سهموی با حدس اولیه و پیشین هموارسازی منطبق است؛ خطای کم آن فقط اثر تعداد ذرات نیست.',
 'انتخاب شروع با تابع هدف انجام شد، نه با خطای پروفایل حقیقی.',
 'خطای عملگر مستقیم، نویز، منظم‌سازی و بودجهٔ محدود می‌توانند مانع بهبود شوند.',
 'شروع‌های ناموفق و توقف‌های سقف تکرار در گزارش هر حالت محفوظ‌اند.',
 'مقایسهٔ همهٔ حالت‌ها در `summary.csv` و گزارش‌های `fits/` موجود است.']
 (OUT/'RESULTS_FA.md').write_text('\n'.join(lines)+'\n')
 dump(OUT/'runtime.json',{n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','scipy','scikit-fem','matplotlib']})
 dump(OUT/'audit.json',dict(fits=50,inlet_metrics_recomputed=50,replays_recomputed=sum(range(2,7))*10,checkpoint_unchanged=True,all_count_conditions_present=True))
 (OUT/'SHA256SUMS').write_text('\n'.join(sha(f)+'  '+str(f.relative_to(OUT)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='SHA256SUMS')+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
