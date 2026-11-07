"""Post-fit comparison only: timestamped information never enters the new fit."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from flow_reconstruction.io import load_fit
from flow_reconstruction.particles import Track
from flow_reconstruction_no_time.geometry import resample_curve,curve_metrics,point_polyline_distance

HERE=Path(__file__).resolve().parent
COLORS=['#f1a300','#df42b3'];FIGS=[]
LABELS={'corners_beta4':'Corner launches / experiment B','corners_beta5':'Corner launches / experiment A','oblique_beta5':'Oblique launches'}
plt.rcParams.update({'font.size':13,'axes.titlesize':15,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True);stem=folder/name
    fig.savefig(stem.with_suffix('.png'),dpi=320,facecolor='white');fig.savefig(stem.with_suffix('.pdf'),facecolor='white');FIGS.append(fig)

def canvas(title,headers):
    n=len(headers);fig,axes=plt.subplots(1,n,figsize=(5.4*n,5.3));fig.subplots_adjust(left=.035,right=.985,bottom=.23,top=.72,wspace=.43)
    fig.suptitle(title,fontsize=22,fontweight='bold',y=.985,color='#13394a')
    for i,(ax,title) in enumerate(zip(axes,headers)):
        ax.set_title(f'{i+1:02d}  {title}',color='#125a70',fontweight='bold',pad=17)
        if i<n-1:
            a,b=ax.get_position(),axes[i+1].get_position();fig.text((a.x1+b.x0)/2,.81,'→',fontsize=29,ha='center',color='#b17a22')
    return fig,axes

def read_csv(p):return np.genfromtxt(p,delimiter=',',names=True)
def xy(d):return np.column_stack([d['x'],d['y']])

def process(name,source):
    source=ROOT/source;new=HERE/'experiments'/name/'result';dest=HERE/'comparisons'/name;dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((new/'report.json').read_text());spec=json.loads((source/'problem.json').read_text())
    old_model=json.loads((source/'fitted_model.json').read_text())
    f,theta=load_fit(source/'fitted_model.json',ROOT/'models/candidates/pressure_long/checkpoint.npz',allow_unvalidated=True)
    observations=[];geometry_inputs=[];timed=[];untimed=[];paths=[];errors=[]
    for i,item in enumerate(spec['tracks']):
        d=read_csv(source/item['csv']);q=xy(d);observations.append(q)
        dense=np.linspace(d['t'][0],d['t'][-1],1001)
        interpolated=np.column_stack([np.interp(dense,d['t'],q[:,j]) for j in range(2)])
        track=Track(item['name'],dense,interpolated,item['initial_velocity'],item['stokes_time'])
        replay=f.simulate(theta,track);timed.append(replay['position'])
        np.savetxt(dest/f'timed_track_{i+1}_dense.csv',np.column_stack([replay['time'],replay['position'],replay['velocity']]),delimiter=',',header='t,x,y,vx,vy',comments='')
        nt=read_csv(new/f'track_{i+1}_forward.csv');untimed.append(xy(nt))
        metrics=dict(particle=i+1,timed=curve_metrics(q,timed[-1]),without_time=curve_metrics(q,untimed[-1]),
                     timed_completed=replay['completed'],without_time_completed=report['paths'][i]['completed'],
                     true_duration_for_posthoc_comparison_only=float(d['t'][-1]-d['t'][0]),inferred_duration=report['inferred_durations'][i])
        exported=xy(read_csv(HERE/'experiments'/name/'input'/f'curve_{i+1}.csv'))
        geometry_inputs.append(exported)
        with np.load(new/'fit_arrays.npz',allow_pickle=False) as arrays:fit_curve=arrays['resampled_curves'][i]
        metrics['uniform_export_vs_original']=curve_metrics(q,exported)
        metrics['fit_resampling_vs_original']=curve_metrics(q,fit_curve)
        paths.append(metrics)
        obs,s=resample_curve(q,1001)
        error1=point_polyline_distance(obs,resample_curve(timed[-1],1001)[0]);error2=point_polyline_distance(obs,resample_curve(untimed[-1],1001)[0]);errors.append((s/s[-1],error1,error2))
        np.savetxt(dest/f'geometric_error_{i+1}.csv',np.column_stack([s/s[-1],error1,error2]),delimiter=',',header='normalized_arclength,timed_distance_H,no_time_distance_H',comments='')
    iy=np.linspace(0,1,4097);a=read_csv(source/'inlet_reference_units.csv');b=read_csv(new/'inlet_reference_units.csv')
    u1=CubicSpline(a['y'],a['u'])(iy);u2=CubicSpline(b['y'],b['u'])(iy)
    normalized1=u1/old_model['mean_speed'];normalized2=u2/report['mean_speed']
    profile_difference=float(np.linalg.norm(u2-u1)/np.linalg.norm(u1));shape_difference=float(np.linalg.norm(normalized2-normalized1)/np.linalg.norm(normalized1))
    np.savetxt(dest/'inlet_comparison.csv',np.column_stack([iy,u1,u2,normalized1,normalized2]),delimiter=',',header='y,timed_u,no_time_u,timed_normalized,no_time_normalized',comments='')
    fig,axes=canvas(f"PI-CFO | {LABELS[name]} | with time vs without measured time",
        ['Given ordered paths','Two inferred inlets','Flow without time','Particles in that flow','Both path predictions','Geometric mismatch'])
    axes[1].plot(u1,iy,'--',color='#ca7026',lw=2,label='With time');axes[1].plot(u2,iy,color='#11687c',lw=2,label='Without time')
    axes[1].set(xlabel='Inferred inlet u / U_ref',ylabel='y/H',ylim=(0,1));axes[1].grid(alpha=.2);axes[1].legend()
    axes[1].text(.03,.97,f'Inlet difference: {100*profile_difference:.2f}%',va='top',fontsize=10,transform=axes[1].transAxes)
    with np.load(new/'reconstructed_flow.npz',allow_pickle=False) as d:x,y,u,v=[d[k].copy() for k in ['x','y','u','v']]
    for ax in axes[2:4]:
        im=ax.pcolormesh(x,y,np.hypot(u,v).T,shading='nearest',cmap='viridis',vmin=0,vmax=np.hypot(u,v).max())
        ax.streamplot(x,y,u.T,v.T,color=(1,1,1,.45),density=(.75,.55),linewidth=.5,arrowsize=.65)
        fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label='Fluid |U| / U_ref')
    for j in [0,2,3,4]:axes[j].set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H')
    for i,color in enumerate(COLORS):
        q=observations[i];q_display=geometry_inputs[i]
        axes[0].plot(q_display[:,0],q_display[:,1],'.-',ms=2,lw=1.2,color=color,label=f'Input {i+1}')
        axes[0].scatter(q[0,0],q[0,1],marker='s',s=35,color=color,edgecolor='black')
        axes[3].plot(*untimed[i].T,color=color,lw=2,path_effects=[pe.Stroke(linewidth=3,foreground='#222'),pe.Normal()])
        for fraction in [.25,.5,.75]:
            k=int((len(untimed[i])-1)*fraction);axes[3].scatter(*untimed[i][k],s=35,color=color,edgecolor='black',zorder=6)
        axes[4].plot(*timed[i].T,'--',color=color,lw=2.2,label=f'With time {i+1}')
        axes[4].plot(*untimed[i].T,color=color,lw=1.5,label=f'Without time {i+1}')
        axes[4].scatter(*q[::2].T,s=14,facecolors='none',edgecolors='#222',label=f'Input {i+1}',zorder=5)
        s,e1,e2=errors[i];axes[5].plot(s,100*e1,'--',color=color,lw=1.7,label=f'With time {i+1}');axes[5].plot(s,100*e2,color=color,lw=1.7,label=f'Without time {i+1}')
    axes[0].legend();axes[4].legend(fontsize=8);axes[5].legend(fontsize=8)
    axes[5].set(xlabel='Normalized input arclength',ylabel='Input-to-output curve distance [% H]');axes[5].grid(alpha=.2)
    fig.text(.035,.028,'Solid: without measured time; dashed: with time; open circles: given geometry. Both outputs are independent ODE replays. Curve agreement does not guarantee matching inlet profiles.',fontsize=11,color='#405263')
    save(fig,HERE/'figures',name+'_comparison_row')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,first,second,label in zip(axes,[u1,normalized1],[u2,normalized2],['Inlet u / U_ref','Inlet u / own mean']):
        ax.plot(iy,first,'--',color='#ca7026',lw=2.2,label='With time');ax.plot(iy,second,color='#11687c',lw=2.2,label='Without measured time')
        ax.set(xlabel='y/H',ylabel=label,xlim=(0,1));ax.grid(alpha=.2);ax.legend()
    axes[0].set_title(f'Absolute inlet difference: {100*profile_difference:.2f}% L2')
    axes[1].set_title(f'Normalized shape difference: {100*shape_difference:.2f}% L2')
    fig.suptitle(f'PI-CFO | {LABELS[name]} | estimated inlets',fontsize=18,fontweight='bold');save(fig,HERE/'figures',name+'_inlet_comparison')
    result=dict(case=name,label=LABELS[name],paths=paths,profile_relative_l2_difference=profile_difference,normalized_profile_relative_l2_difference=shape_difference,
                timed_mean_speed=old_model['mean_speed'],without_time_mean_speed=report['mean_speed'],timed_reynolds=old_model['reynolds'],without_time_reynolds=report['reynolds'],
                source=str(source.relative_to(ROOT)),fit_report_sha256=sha(new/'report.json'),timed_fit_sha256=sha(source/'fitted_model.json'),
                comparison_scope='Same observed curves; geometry-only scores for both methods; profile difference is not error against truth')
    (dest/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    sources=json.loads((HERE/'comparison_sources.json').read_text());results=[]
    for name,source in sources.items():
        if (HERE/'experiments'/name/'result/report.json').exists():results.append(process(name,source))
    (HERE/'comparison_metrics.json').write_text(json.dumps(dict(cases=results),indent=2)+'\n')
    with PdfPages(HERE/'all_comparisons.pdf') as pdf:
        for fig in FIGS:pdf.savefig(fig,facecolor='white');plt.close(fig)
    print('Compared',len(results),'cases',flush=True)

if __name__=='__main__':main()
