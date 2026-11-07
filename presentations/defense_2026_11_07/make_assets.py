"""High-resolution scientific figures for the defense, from saved numeric arrays."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

HERE=Path(__file__).resolve().parent;OUT=HERE/'assets'
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})


def equation(name,lines,width=12,height=2.5):
    fig=plt.figure(figsize=(width,height),facecolor='white')
    for i,line in enumerate(lines):fig.text(.5,1-(i+1)/(len(lines)+1),line,ha='center',va='center',fontsize=23)
    fig.savefig(OUT/name,dpi=180,bbox_inches='tight',pad_inches=.28);plt.close(fig)


def flow_paths(source,name,title):
    with np.load(source/'reconstructed_flow.npz',allow_pickle=False) as d:x,y,u,v=[d[k].copy() for k in ['x','y','u','v']]
    fig,ax=plt.subplots(figsize=(14,3.55),layout='constrained')
    im=ax.pcolormesh(x,y,np.hypot(u,v).T,shading='nearest',cmap='viridis',vmin=0)
    ax.streamplot(x,y,u.T,v.T,density=(1.5,.65),color=(1,1,1,.55),linewidth=.6,arrowsize=.75)
    records=[]
    for i,color in [(1,'#ffb000'),(2,'#ff4fb8')]:
        q=np.genfromtxt(source/f'track_{i}_forward.csv',delimiter=',',names=True)
        obs=np.genfromtxt(source/f'observed_track_{i}.csv',delimiter=',',names=True)
        ax.plot(q['x'],q['y'],color=color,lw=2.5,path_effects=[pe.Stroke(linewidth=3.5,foreground='#222'),pe.Normal()])
        ax.scatter(obs['x'][::3],obs['y'][::3],s=13,facecolors='none',edgecolors='white',lw=.8)
        records.append(dict(t=q['t'],distance=np.hypot(q['x']-obs['x'],q['y']-obs['y'])))
    ax.set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H',title=title,aspect='equal')
    fig.colorbar(im,ax=ax,label='Inferred fluid speed / U_ref',fraction=.025,pad=.02)
    fig.savefig(OUT/name,dpi=200);plt.close(fig)
    return records


def main():
    OUT.mkdir(exist_ok=True)
    equation('ns_equations.png',[r'$\nabla\cdot\mathbf{u}=0$',r'$(\mathbf{u}\cdot\nabla)\mathbf{u}=-\nabla p+Re^{-1}\nabla^2\mathbf{u}$',r'$Re=\overline{U}_{in}H/\nu,\quad \Omega=[0,5]\times[0,1]$'],height=3)
    equation('particle_equations.png',[r'$\dot{\mathbf{q}}=\mathbf{w},\qquad \dot{\mathbf{w}}=(\mathbf{U}(\mathbf{q})-\mathbf{w})/St$',r'$St=\tau_p U_{ref}/H,\quad Re=A\,Re_{ref},\quad\mathbf{U}=A\,\mathbf{u}_\theta$'],height=2)
    equation('weak_equation.png',[r'$\mathbf{q}(t)=\mathbf{q}_0+St\,\mathbf{w}_0\left(1-e^{-(t-t_0)/St}\right)$',r'$\qquad+\int_{t_0}^{t}\left(1-e^{-(t-s)/St}\right)\mathbf{U}(\mathbf{q}(s))\,ds$'],height=2.4)
    equation('loss.png',[r'$\mathcal{L}_{forward}=\langle |\mathbf{r}_{NS}|^2/s\rangle+10\langle |\mathbf{t}_{out}|^2\rangle$',r'$u=\partial_y\psi,\quad v=-\partial_x\psi,\quad F(y)=\int_0^y f(\eta)\,d\eta$'],height=2.1)
    equation('inverse_loss.png',[r'$\mathcal{J}=\mathcal{J}_{positions}+\lambda_f\int_0^1|f^{\prime\prime}(y)|^2dy$',r'$\qquad+\lambda_{NS}\langle |\mathbf{r}_{NS}|^2\rangle+\mathcal{R}(f,A)$'],height=2.2)
    # Geometry and boundary conditions, using precisely the implemented traction.
    fig,ax=plt.subplots(figsize=(14,3.6),layout='constrained');ax.add_patch(plt.Rectangle((0,0),5,1,facecolor='#e9f4f7',edgecolor='#17364c',lw=3))
    yy=np.linspace(0,1,13)
    ax.quiver(np.full_like(yy,-.4),yy,.45*6*yy*(1-yy),np.zeros_like(yy),angles='xy',scale_units='xy',scale=1,color='#16819b')
    ax.text(2.5,1.1,'No slip: u=v=0',ha='center',fontsize=15);ax.text(2.5,-.16,'No slip: u=v=0',ha='center',fontsize=15)
    ax.text(-.55,.5,'u=f(y)\nv=0',ha='right',va='center',fontsize=15)
    ax.text(5.13,.5,'u_x/Re - p = 0\nv_x/Re = 0\np_external = 0',va='center',fontsize=13)
    ax.text(2.5,.5,'Steady 2-D incompressible channel | L/H=5',ha='center',fontsize=15,color='#17364c')
    ax.set(xlim=(-1.2,6.4),ylim=(-.35,1.35),aspect='equal');ax.axis('off');fig.savefig(OUT/'geometry.png',dpi=190);plt.close(fig)
    metrics=json.loads((HERE/'data/forward_metrics.json').read_text())['all_predeclared_cases']
    rr=[m for m in metrics if m['power']==4];fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,key,label,limit in zip(axes,['velocity_relative_l2','pressure_relative_l2'],['Velocity vector relative L2 (%)','Pressure relative L2 (%)'],[5,10]):
        ax.plot([m['reynolds'] for m in rr],[100*m[key] for m in rr],'o-',lw=2.5,color='#167d91',markersize=8)
        ax.axhline(limit,color='#bd5a32',ls='--',label='Registered field threshold');ax.legend(fontsize=9)
        ax.set(xlabel='Reynolds number',ylabel=label,xticks=[20,30,40]);ax.grid(alpha=.2)
    fig.savefig(OUT/'reynolds_scan.png',dpi=190);plt.close(fig)
    known=json.loads((ROOT/'reports/experiments/pressure_long/metrics.json').read_text())['cases']
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax,key,limit in zip(axes,['velocity_relative_l2','pressure_relative_l2'],[5,10]):
        ax.bar([m['name'] for m in known],[100*m[key] for m in known],color=['#c18145' if m[key]>=limit/100 else '#19849a' for m in known])
        ax.axhline(limit,color='#c95148',ls='--');ax.set(ylabel=key.replace('_',' ')+' (%)');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'all_original_cases.png',dpi=190);plt.close(fig)
    base=ROOT/'flow_reconstruction/examples/corner_launch_preconditioned/result'
    r1=flow_paths(base,'inverse_beta5.png','Beta(5,5) experiment | reconstructed carrier flow and inertial particles')
    second=HERE/'data/inverse_beta4/result'
    if second.exists() and (second/'report.json').exists():
        r2=flow_paths(second,'inverse_beta4.png','Beta(4,4) experiment | reconstructed carrier flow and inertial particles')
        fig,axes=plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
        for ax,records,title in zip(axes,[r1,r2],['Beta(5,5)','Beta(4,4)']):
            for r,color,label in zip(records,['#d18a00','#d53ba5'],['Upper-left particle','Lower-right particle']):ax.plot(r['t'],r['distance'],color=color,label=label,lw=1.7)
            ax.set(title=title,xlabel='t / (H/U_ref)',ylabel='Particle position mismatch / H');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(OUT/'inverse_errors.png',dpi=190);plt.close(fig)
    # Input positions alone, supplied to the inverse model.
    fig,ax=plt.subplots(figsize=(12,3.5),layout='constrained')
    for i,color in [(1,'#16819b'),(2,'#bd5b48')]:
        q=np.genfromtxt(base/f'observed_track_{i}.csv',delimiter=',',names=True)
        ax.plot(q['x'],q['y'],'.-',color=color,ms=3,label=f'Input particle {i}');ax.scatter(q['x'][0],q['y'][0],marker='s',s=60,color=color)
    ax.set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H',aspect='equal',title='Given timestamped positions; launch velocities and St are known');ax.legend()
    fig.savefig(OUT/'observed_tracks.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    comp=np.load(base/'truth_comparison.npz');axes[0].plot(comp['inlet_y'],comp['inlet_truth'],label='Generating inlet')
    axes[0].plot(comp['inlet_y'],comp['inlet_prediction'],label='Inferred inlet');axes[0].legend();axes[0].set(xlabel='y/H',ylabel='u_in / U_ref',title='Good paths do not imply unique inlet recovery')
    report=json.loads((base/'report.json').read_text());sv=np.asarray(report['data_jacobian_singular_values'])
    axes[1].semilogy(np.arange(1,len(sv)+1),sv,'o-',color='#16819b',ms=3);axes[1].axhline(sv[0]*1e-5,color='#bd5b48',ls='--')
    axes[1].set(xlabel='Singular-value index',ylabel='Data Jacobian singular value',title='Local observability (coordinate-dependent)')
    fig.savefig(OUT/'identifiability.png',dpi=190);plt.close(fig)


if __name__=='__main__':main()
