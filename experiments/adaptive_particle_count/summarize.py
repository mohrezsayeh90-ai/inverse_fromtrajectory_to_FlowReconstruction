"""Plot actual evaluated adaptive fits; do not interpolate missing experiments."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/setup/mpl-cache')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/setup/cache')
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'New_Adaptive_Particle_Results'/os.environ.get('ADAPTIVE_VARIANT','continuation');FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
NAMES=['beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed','waves3_r30','lobes4_r30','nine30']
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
rows=[];manifest=[]
for name in NAMES:
 path=OUT/'progress'/f'{name}.json'
 if not path.exists():continue
 status=json.loads(path.read_text())
 if status['phase']!='completed':continue
 available=sorted(v['N'] for v in status['evaluated'] if v['N']<=status['final_count'])
 if len(available)>=5:
  counts=sorted(set(min(available,key=lambda n:abs(n-target)) for target in np.linspace(2,status['final_count'],6)))
  while len(counts)<min(6,len(available)):
   counts.append(max((n for n in available if n not in counts),key=lambda n:min(abs(n-k) for k in counts)));counts.sort()
  status['display_counts']=counts;path.write_text(json.dumps(status,indent=2)+'\n')
 else:counts=status['display_counts']
 fig,axs=plt.subplots(2,3,figsize=(6.27,4.45),sharex=True,sharey=True)
 inputs={};errors=[]
 for ax,n in zip(axs.flat,counts):
  p=OUT/f'fits/{name}/N{n}/inlet.csv';d=np.genfromtxt(p,delimiter=',',names=True);r=json.loads(p.with_name('report.json').read_text())
  e=100*np.linalg.norm(d['estimated']-d['truth'])/np.linalg.norm(d['truth']);np.testing.assert_allclose(e,100*r['inlet_relative_l2'],rtol=1e-12)
  ax.plot(d['truth'],d['y'],'k-',lw=1.15,label='Reference')
  if OUT.name=='near_inlet':
   control=OUT.parent/f'continuation/fits/{name}/N{n}/inlet.csv'
   if control.exists():
    b=np.genfromtxt(control,delimiter=',',names=True);np.testing.assert_array_equal(b['truth'],d['truth']);inputs[str(control.relative_to(ROOT))]=hashlib.sha256(control.read_bytes()).hexdigest();ax.plot(b['estimated'],b['y'],color='#173f59',ls=':',lw=1.1,label='Mixed design (base)')
  ax.plot(d['estimated'],d['y'],color='#bb2888',ls='--',lw=1.2,label='Near inlet' if OUT.name=='near_inlet' else 'Reconstructed')
  ax.set(xlabel='$u/U_{ref}$',ylabel='$y/H$',ylim=(0,1),title=f'N={n}; error={e:.2f}%');ax.grid(alpha=.15)
  inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();inputs[str(p.with_name('report.json').relative_to(ROOT))]=hashlib.sha256(p.with_name('report.json').read_bytes()).hexdigest();errors.append(e)
 for ax in list(axs.flat)[len(counts):]:ax.axis('off')
 fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3 if OUT.name=='near_inlet' else 2,frameon=False)
 fig.suptitle(name.replace('_',' '),fontsize=10);fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.86,wspace=.28,hspace=.65)
 for ext in ['pdf','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=250,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
 plt.close(fig)
 manifest.append(dict(profile=name,display_counts=counts,errors_percent=errors,inputs_sha256=inputs))
 rows.append(dict(profile=name,final_N=status['final_count'],final_error_percent=100*next(v['error'] for v in status['evaluated'] if v['N']==status['final_count']),stopping_reason=status['stopping_reason'],smallest_evaluated_target_N=min([v['N'] for v in status['evaluated'] if v['complete'] and v['error']<=status['target_relative_l2']],default=None)))
with (OUT/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['profile','final_N','final_error_percent','stopping_reason','smallest_evaluated_target_N']);w.writeheader();w.writerows(rows)
(OUT/'figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Completed figures:',len(rows),'/',9 if OUT.name=='continuation' else 1)

if len(rows)==9:
 import fitz
 pack=fitz.open()
 for name in NAMES:
  with fitz.open(FIG/f'{name}.pdf') as source:pack.insert_pdf(source)
 target=FIG/'all_profiles.pdf'
 if target.exists():target.unlink()
 pack.save(target)
 pack.close()
