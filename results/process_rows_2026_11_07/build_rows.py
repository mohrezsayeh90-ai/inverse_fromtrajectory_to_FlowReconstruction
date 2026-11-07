"""Publication-size five-panel process rows, using fixed numerical predictions."""
import hashlib,json,sys,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE=Path(__file__).resolve().parent;DECK=ROOT/'presentations/defense_2026_11_07'
plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':10})


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def canvas(title,headers):
    fig,axes=plt.subplots(1,5,figsize=(27,5.3))
    fig.subplots_adjust(left=.035,right=.985,bottom=.23,top=.72,wspace=.43)
    fig.suptitle(title,fontsize=23,fontweight='bold',y=.985,color='#13394a')
    for i,(ax,header) in enumerate(zip(axes,headers)):
        ax.set_title(f'{i+1:02d}  {header}',pad=17,color='#125a70',fontweight='bold')
        if i<4:
            box=ax.get_position();nxt=axes[i+1].get_position()
            fig.text((box.x1+nxt.x0)/2,.81,'→',fontsize=29,ha='center',color='#b17a22')
    return fig,axes


def save(fig,folder,name):
    folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder/(name+'.png'),dpi=320,facecolor='white')
    fig.savefig(folder/(name+'.pdf'),facecolor='white');plt.close(fig)


def direct(name,component='speed'):
    path=HERE/'data'/f'{name}.npz'
    if not path.exists():shutil.copy2(DECK/'data'/path.name,path)
    with np.load(path,allow_pickle=False) as d:raw={k:d[k].copy() for k in d.files}
    truth,pred=raw['reference'],raw['prediction'];xy=raw['xy'];x,y=np.unique(xy[:,0]),np.unique(xy[:,1]);re=float(raw['reynolds'])
    if component=='speed':
        field1,field2=np.linalg.norm(pred[:,:2],axis=1),np.linalg.norm(truth[:,:2],axis=1)
        delta=np.linalg.norm(pred[:,:2]-truth[:,:2],axis=1);error=np.linalg.norm(pred[:,:2]-truth[:,:2])/np.linalg.norm(truth[:,:2]);quantity='|U| / Umean';metric='velocity-vector'
    else:
        j={'u':0,'v':1,'p':2}[component];field1,field2=pred[:,j],truth[:,j];delta=abs(field1-field2)
        error=np.linalg.norm(field1-field2)/np.linalg.norm(field2);quantity=component+(' / (rho Umean^2)' if component=='p' else ' / Umean');metric={'u':'axial velocity','v':'transverse velocity','p':'pressure'}[component]
    power={'beta3':3,'beta4':4,'beta5':5}[name.split('_')[0]]
    fig,axes=canvas(f'PI-CFO  |  Beta({power},{power}) inlet, Re={re:g}  |  {metric} relative L2 = {100*error:.3f}%',
                    ['Input inlet profile','PI-CFO prediction','Independent FEM','Pointwise error','Cross-section check'])
    axes[0].plot(raw['inlet_values'],raw['inlet_y'],color='#11687c',lw=2.6);axes[0].set(xlabel='Inlet u / Umean',ylabel='y/H',ylim=(0,1));axes[0].grid(alpha=.2)
    low,high=min(field1.min(),field2.min()),max(field1.max(),field2.max());cmap='coolwarm' if component=='v' else 'viridis'
    for ax,values in zip(axes[1:3],[field1,field2]):
        im=ax.pcolormesh(x,y,values.reshape(len(x),len(y)).T,shading='nearest',cmap=cmap,vmin=low,vmax=high)
        ax.set(xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label=quantity)
    im=axes[3].pcolormesh(x,y,delta.reshape(len(x),len(y)).T,shading='nearest',cmap='magma',vmin=0)
    axes[3].set(xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=axes[3],orientation='horizontal',pad=.2,fraction=.07,label='Absolute '+metric+' error')
    # Same scalar as the contours; vector error in panel 4 never uses cancellation
    # between speed magnitudes. The input f remains axial for every output type.
    for target,color in [(.1,'#0a7492'),(.5,'#c8771e'),(2.,'#8448a1')]:
        ix=int(np.argmin(abs(x-target)))
        axes[4].plot(field2.reshape(len(x),len(y))[ix],y,color=color,lw=2,label=f'FEM, x={x[ix]:.2f}')
        axes[4].plot(field1.reshape(len(x),len(y))[ix],y,color=color,lw=1.7,ls='--',label='PI-CFO')
    axes[4].set(xlabel=quantity,ylabel='y/H');axes[4].legend(loc='best',fontsize=10);axes[4].grid(alpha=.2)
    fig.text(.035,.028,'Shared prediction/FEM color limits. Error is absolute and is not rescaled into a percentage at low-speed points. Same fixed checkpoint (94k).',fontsize=12,color='#405263')
    stem=f'{name}_{component}';save(fig,HERE/'forward',stem)
    return dict(figure=stem,profile=f'Beta({power},{power})',reynolds=re,quantity=metric,relative_l2=float(error),source_sha256=sha(path))


def inverse(source,stem,label):
    with np.load(source/'reconstructed_flow.npz',allow_pickle=False) as d:x,y,u,v=[d[k].copy() for k in ['x','y','u','v']]
    model=json.loads((source/'fitted_model.json').read_text());speed=np.hypot(u,v)
    fig,axes=canvas(f'PI-CFO trajectory-conditioned reconstruction  |  {label}  |  inferred Re={model["reynolds"]:.3f}',
                    ['Given particle paths','Inferred carrier flow','Particles in that flow','Input / output overlay','Time-aligned mismatch'])
    for ax in axes[:4]:ax.set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H')
    for ax in axes[1:3]:
        im=ax.pcolormesh(x,y,speed.T,shading='nearest',cmap='viridis',vmin=0,vmax=speed.max())
        ax.streamplot(x,y,u.T,v.T,color=(1,1,1,.45),density=(.75,.55),linewidth=.5,arrowsize=.65)
        fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label='Fluid |U| / U_ref')
    rows=[];hashes={'fitted_model':sha(source/'fitted_model.json'),'flow':sha(source/'reconstructed_flow.npz')}
    for i,color in [(1,'#f1a300'),(2,'#df42b3')]:
        obs=np.genfromtxt(source/f'observed_track_{i}.csv',delimiter=',',names=True)
        sim=np.genfromtxt(source/f'track_{i}_forward.csv',delimiter=',',names=True)
        assert len(obs)==len(sim);np.testing.assert_allclose(obs['t'],sim['t'],atol=1e-12)
        distance=np.hypot(obs['x']-sim['x'],obs['y']-sim['y']);rmse=float(np.sqrt(np.mean(distance**2)))
        axes[0].plot(obs['x'],obs['y'],'.-',color=color,lw=1.2,ms=3,label=f'Input {i}')
        axes[0].scatter(obs['x'][0],obs['y'][0],marker='s',color=color,s=40,edgecolor='black')
        axes[2].plot(sim['x'],sim['y'],color=color,lw=2,path_effects=[pe.Stroke(linewidth=3,foreground='#222'),pe.Normal()])
        axes[3].plot(sim['x'],sim['y'],color=color,lw=2,label=f'ODE output {i}')
        axes[3].scatter(obs['x'][::2],obs['y'][::2],s=15,facecolors='none',edgecolors='#222',label=f'Input {i}')
        for fraction in [.25,.5,.75]:
            k=int((len(sim)-1)*fraction);axes[2].scatter(sim['x'][k],sim['y'][k],s=40,color=color,edgecolors='black',zorder=6)
        axes[4].plot(sim['t'],distance,color=color,label=f'Particle {i}: RMSE {100*rmse:.4f}% H',lw=1.7)
        rows.append(dict(particle=i,position_2d_rmse_H=rmse,position_2d_rmse_percent_H=100*rmse,max_error_H=float(distance.max()),samples=len(obs)))
        hashes[f'input_{i}']=sha(source/f'observed_track_{i}.csv');hashes[f'ODE_{i}']=sha(source/f'track_{i}_forward.csv')
    axes[0].legend(fontsize=11);axes[3].legend(fontsize=9);axes[4].legend(fontsize=9)
    axes[4].set(xlabel='t / (H/U_ref)',ylabel='Euclidean position error / H');axes[4].grid(alpha=.2)
    fig.text(.035,.028,'Open circles: input positions; colored lines: independent particle ODE in the saved inferred flow. No re-fitting and no reference-field comparison in this row.',fontsize=12,color='#405263')
    save(fig,HERE/'inverse',stem)
    return dict(figure=stem,label=label,source=str(source.relative_to(ROOT)),source_sha256=hashes,particles=rows,
                comparison_scope='Consistency of fitted paths, not a blind trajectory test or proof of whole-field accuracy')


def main():
    (HERE/'data').mkdir(exist_ok=True)
    old=json.loads((DECK/'data/forward_metrics.json').read_text())['all_predeclared_cases']
    new=json.loads((HERE/'data/forward_metrics.json').read_text())['all_predeclared_cases']
    attempted=old+new
    selected=[m for m in attempted if m['velocity_relative_l2']<.05 and m['pressure_relative_l2']<.1]
    direct_results=[direct(row['name']) for row in selected]
    for component in ['u','v','p']:direct_results.append(direct('beta4_re30',component))
    inverse_results=[]
    for source,stem,label in [(ROOT/'flow_reconstruction/examples/fem_beta/result','oblique_beta5','Oblique launches / Beta(5,5) experiment'),
                              (ROOT/'flow_reconstruction/examples/corner_launch_preconditioned/result','corners_beta5','Corner launches / Beta(5,5) experiment'),
                              (DECK/'data/inverse_beta4/result','corners_beta4','Corner launches / Beta(4,4) experiment')]:
        inverse_results.append(inverse(source,stem,label))
    (HERE/'metrics.json').write_text(json.dumps(dict(model_display_name='PI-CFO',model_full_name='Physics-Informed Channel Flow Operator',
        model_lineage='Project-specific modified physics-informed DeepONet, not a claim of inventing the original DeepONet architecture',
        attempted_forward_cases=attempted,selected_forward_cases=[r['name'] for r in selected],
        selection_rule='velocity-vector L2<5% and pressure L2<10%; software, models and old predictions unchanged',
        forward_rows=direct_results,inverse_rows=inverse_results),indent=2)+'\n')
    text=['# ردیف‌های تصویری فرآیند: ورودی ← پیش‌بینی ← بررسی','',
      'نام نمایشی این مدلِ پروژه **PI-CFO — اپراتور فیزیک‌آگاه جریان کانال** است. این نام به پیاده‌سازی و اصلاحات این پروژه اشاره دارد؛ معماری بر پایهٔ DeepONet فیزیک‌آگاه است و ایدهٔ اصلی DeepONet به‌عنوان اختراع جدید معرفی نمی‌شود.',
      '', '## ۱. حل مستقیم؛ هر تصویر پنج نمودار در یک ردیف','',
      'از چپ: پروفایل ورودی؛ کانتور پیش‌بینی PI-CFO؛ کانتور FEM؛ کانتور خطای نقطه‌ای؛ برش‌های سرعت برای مقایسه. برای دو کانتور پیش‌بینی و مرجع از مقیاس رنگ یکسان استفاده شده است. نقشهٔ خطای سرعت برداری برابر `sqrt((u_model-u_FEM)^2+(v_model-v_FEM)^2)` است و در نقاط سرعت نزدیک صفر بر مرجع تقسیم نشده است.',
      '', '| پروفایل | Re | خطای سرعت برداری | خطای فشار |','|---|---:|---:|---:|']
    for r in selected:text.append(f"| Beta({r['power']+1},{r['power']+1}) | {r['reynolds']:g} | {100*r['velocity_relative_l2']:.3f}٪ | {100*r['pressure_relative_l2']:.3f}٪ |")
    text+=['', 'چهار حالت منتخب از پنج بررسی موجودند. حالت Beta(5,5)، Re=40 به‌دلیل فشار ۱۱٫۷۶٪ در این مجموعهٔ کم‌خطا انتخاب نشده؛ عدد آن در metrics.json محفوظ است. این نمونه‌ها بررسی توسعه‌ای‌اند و دقت برای هر ورودی یا پاس‌شدن تمام معیارهای NS را ثابت نمی‌کنند. هیچ وزن شبکه‌ای تغییر نکرده است.']
    for row in direct_results:
        stem=row['figure'];text+=['',f"### {row['profile']}، Re={row['reynolds']:g} — {row['quantity']}",'',f'![پروفایل تا نقشه خطا](forward/{stem}.png)','',f'[PNG باکیفیت](forward/{stem}.png) · [PDF برداری](forward/{stem}.pdf)']
    text+=['','## ۲. بازسازی از ذرات؛ هر تصویر پنج نمودار در یک ردیف','',
      'از چپ: دو مسیر ورودی؛ کانتور اندازهٔ سرعت سیال تخمینی؛ حرکت ذرات در همان میدان؛ هم‌پوشانی مسیر ورودی و خروجی؛ فاصلهٔ دو مسیر در زمان یکسان. این ردیف‌ها مقایسهٔ میدان با FEM ندارند و فقط سازگاری مسیرها را نشان می‌دهند. هر سه مجموعهٔ مسیر از جواب‌های ذخیره‌شده خوانده شده‌اند.']
    for r in inverse_results:
        stem=r['figure'];text+=['',f"### {r['label']}",'',f'![مسیر ورودی تا خطای مسیر](inverse/{stem}.png)','',
          ' · '.join(f"ذره {p['particle']}: RMSE دوبعدی {p['position_2d_rmse_percent_H']:.4f}٪ ارتفاع کانال" for p in r['particles']),
          '',f'[PNG باکیفیت](inverse/{stem}.png) · [PDF برداری](inverse/{stem}.pdf)']
    text+=['', '## خواندن شکل‌ها و بازتولید','',
      'PNGها با وضوح ۳۲۰dpi و اندازهٔ ۸۶۴۰×۱۶۹۶ پیکسل تولید می‌شوند؛ PDFها برای بزرگ‌نمایی و قرار دادن در اسلاید مناسب‌اند. محورها بی‌بعدند و برای چیدمان کنارهم، نسبت نمایش محورها مستقل از نسبت هندسی کانال است؛ عددهای x/H و y/H ملاک‌اند.',
      '', 'اسلایدها و نتایج قبلی تغییر نکرده‌اند. نام جدید فقط در این شکل‌ها به کار رفته است. پوشهٔ data دادهٔ خام و metrics.json معیارها و هش‌های ورودی‌ها را نگه می‌دارد. برای تولید دوباره از ریشهٔ مخزن:','',
      '```bash','python results/process_rows_2026_11_07/build_rows.py','```','',
      'دادهٔ جدید Beta(3,3) با دو شبکهٔ مستقل FEM تهیه شده است. اگر فایل جدید در دسترس نباشد، همان تولیدکنندهٔ `presentations/defense_2026_11_07/prepare_results.py` با CASES برابر `(beta3_re30,2,30)` و مسیرهای خروجی این پوشه قابل استفاده است.']
    (HERE/'README.md').write_text('\n'.join(text)+'\n')
    print('Saved',len(direct_results),'forward rows and',len(inverse_results),'inverse rows')


if __name__=='__main__':main()
