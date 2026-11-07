"""Requested chains only: one inlet and x(t)/y(t), no re-fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
LABELS={'corners_beta4':'Corner launches / experiment B','corners_beta5':'Corner launches / experiment A','oblique_beta5':'Oblique launches'}
COLORS=['#f1a300','#df42b3']
plt.rcParams.update({'font.size':13,'axes.titlesize':15,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42})
SOURCES={}

def read(p):
    SOURCES[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return np.genfromtxt(p,delimiter=',',names=True)

def plot_case(name):
    result=BASE/'experiments'/name/'result';inputs=BASE/'experiments'/name/'input';comparison=BASE/'comparisons'/name
    fig=plt.figure(figsize=(35,5.7));grid=fig.add_gridspec(1,6,width_ratios=[1,1,1,1,1.13,1.3],left=.033,right=.986,bottom=.23,top=.72,wspace=.43)
    axes=[fig.add_subplot(grid[i]) for i in range(5)]
    clock_grid=grid[5].subgridspec(2,1,hspace=.2);ax_xt=fig.add_subplot(clock_grid[0]);ax_yt=fig.add_subplot(clock_grid[1],sharex=ax_xt)
    fig.suptitle(f'PI-CFO | {LABELS[name]} | reconstruction without measured time',fontsize=23,fontweight='bold',y=.98,color='#13394a')
    headers=['Given ordered paths','Inferred inlet profile','Flow without time','Particles in that flow','Both path predictions','x(t) and y(t)']
    title_axes=axes+[ax_xt]
    for i,(ax,title) in enumerate(zip(title_axes,headers)):
        box=ax.get_position();fig.text((box.x0+box.x1)/2,.77,f'{i+1:02d}  {title}',ha='center',fontsize=15,fontweight='bold',color='#125a70')
        if i<5:
            nxt=title_axes[i+1].get_position();fig.text((box.x1+nxt.x0)/2,.84,'→',ha='center',fontsize=29,color='#b17a22')
    inlet=read(result/'inlet_reference_units.csv');iy=np.linspace(0,1,2049);iv=CubicSpline(inlet['y'],inlet['u'])(iy)
    axes[1].plot(iv,iy,color='#11687c',lw=2.5);axes[1].fill_betweenx(iy,0,iv,color='#11687c',alpha=.12)
    axes[1].set(xlabel='Inferred inlet u / U_ref',ylabel='y/H',ylim=(0,1));axes[1].grid(alpha=.2)
    p=result/'reconstructed_flow.npz';SOURCES[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as z:x,y,u,v=[z[k].copy() for k in ['x','y','u','v']]
    for ax in axes[2:4]:
        im=ax.pcolormesh(x,y,np.hypot(u,v).T,shading='nearest',cmap='viridis',vmin=0,vmax=np.hypot(u,v).max())
        ax.streamplot(x,y,u.T,v.T,color=(1,1,1,.45),density=(.75,.55),linewidth=.5,arrowsize=.65)
        fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label='Fluid |U| / U_ref')
    for i in [0,2,3,4]:axes[i].set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H')
    for i,color in enumerate(COLORS,1):
        q=read(inputs/f'curve_{i}.csv');timed=read(comparison/f'timed_track_{i}_dense.csv');nt=read(result/f'track_{i}_forward.csv')
        axes[0].plot(q['x'],q['y'],'.-',ms=2,lw=1.2,color=color,label=f'Input {i}')
        axes[0].scatter(q['x'][0],q['y'][0],s=35,marker='s',color=color,edgecolor='black')
        axes[3].plot(nt['x'],nt['y'],color=color,lw=2,path_effects=[pe.Stroke(linewidth=3,foreground='#222'),pe.Normal()])
        for fraction in [.25,.5,.75]:
            k=int((len(nt)-1)*fraction);axes[3].scatter(nt['x'][k],nt['y'][k],s=35,color=color,edgecolor='black',zorder=5)
        axes[4].plot(timed['x'],timed['y'],'--',color=color,lw=2.2,label=f'With time {i}')
        axes[4].plot(nt['x'],nt['y'],color=color,lw=1.5,label=f'Without time {i}')
        axes[4].scatter(q['x'][::5],q['y'][::5],s=14,facecolors='none',edgecolors='#222',label=f'Input {i}',zorder=5)
        # Use the original clocks of both saved replays, with no time warping.
        for ax,coordinate in [(ax_xt,'x'),(ax_yt,'y')]:
            ax.plot(timed['t'],timed[coordinate],'--',color=color,lw=1.9,label=f'With time {i}')
            ax.plot(nt['inferred_t'],nt[coordinate],color=color,lw=1.5,label=f'Without time {i}')
    axes[0].legend();axes[4].legend(fontsize=8)
    ax_xt.set(ylabel='x/H');ax_xt.tick_params(labelbottom=False);ax_xt.grid(alpha=.2)
    ax_yt.set(xlabel='t / (H/U_ref)',ylabel='y/H');ax_yt.grid(alpha=.2)
    ax_yt.legend(loc='upper center',bbox_to_anchor=(.5,-.43),ncol=2,fontsize=8,frameon=False)
    fig.text(.033,.025,'Solid: without measured time; dashed: with time. Panel 02 shows only the no-time inlet. Panel 06 uses inferred time for the no-time fit; neither clock is rescaled.',fontsize=11,color='#405263')
    fig.savefig(HERE/f'{name}_chain.png',dpi=320,facecolor='white');fig.savefig(HERE/f'{name}_chain.pdf',facecolor='white')
    return fig

def main():
    figures=[plot_case(name) for name in LABELS]
    with PdfPages(HERE/'chains_only.pdf') as pdf:
        for fig in figures:pdf.savefig(fig,facecolor='white');plt.close(fig)
    (HERE/'sources.json').write_text(json.dumps(dict(inputs_sha256=SOURCES,refitting=False,previous_results_modified=False,
        panel_2='No-time inferred inlet only',panel_6='Independent ODE x(t), y(t), both methods and particles, without time-axis rescaling'),indent=2)+'\n')
    (HERE/'README.md').write_text('# زنجیره‌های اصلاح‌شده\n\nفقط سه زنجیرهٔ تصویری در `chains_only.pdf` آمده است. پنل دوم فقط ورودی تخمینی روش بدون زمان را دارد. پنل ششم دو نمودار x(t) و y(t) را برای دو ذره و هر دو روش نشان می‌دهد؛ خط پیوسته بدون زمان و خط‌چین با زمان است. در روش بدون زمان، محور t ساعت تخمینی است؛ هیچ کشش یا نرمال‌سازی زمانی برای منطبق‌کردن منحنی‌ها انجام نشده است. تمام زمان‌ها بی‌بعدند.\n\nاین تغییر فقط نمایش نتایج است؛ هیچ برازش، پارامتر، داده یا خروجی قبلی تغییر نکرده است. بازتولید از ریشهٔ مخزن: `python flow_reconstruction_no_time/chain_revision_xt_yt/build.py`\n')
    print('Saved 3 chains; no re-fitting.')

if __name__=='__main__':main()
