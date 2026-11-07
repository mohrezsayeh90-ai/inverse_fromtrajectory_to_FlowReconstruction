"""Five-panel forward and six-panel inverse rows, using saved numerical arrays."""
import hashlib,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':13,'axes.titlesize':15,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42})
FIGURES=[]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def canvas(title,headers):
    n=len(headers);fig,axes=plt.subplots(1,n,figsize=(5.4*n,5.3))
    fig.subplots_adjust(left=.035,right=.985,bottom=.23,top=.72,wspace=.43)
    fig.suptitle(title,fontsize=22,fontweight='bold',y=.985,color='#13394a')
    for i,(ax,header) in enumerate(zip(axes,headers)):
        ax.set_title(f'{i+1:02d}  {header}',pad=17,color='#125a70',fontweight='bold')
        if i<n-1:
            a,b=ax.get_position(),axes[i+1].get_position()
            fig.text((a.x1+b.x0)/2,.81,'→',fontsize=29,ha='center',color='#b17a22')
    return fig,axes

def save(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True);stem=folder/name
    fig.savefig(stem.with_suffix('.png'),dpi=320,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white')
    FIGURES.append((fig,stem))

def inlet_curve(d):
    y=np.linspace(0,1,2049)
    return y,CubicSpline(d['inlet_y'],d['inlet_values'])(y)

def forward(row):
    path=HERE/'data'/f"{row['name']}.npz"
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    pred,ref,xy=d['prediction'],d['reference'],d['xy'];x,y=np.unique(xy[:,0]),np.unique(xy[:,1])
    ev=np.linalg.norm((pred-ref)[:,:2])/np.linalg.norm(ref[:,:2]);ep=np.linalg.norm((pred-ref)[:,2])/np.linalg.norm(ref[:,2])
    np.testing.assert_allclose([ev,ep],[row['velocity_relative_l2'],row['pressure_relative_l2']],rtol=1e-12)
    assert ev<.05 and ep<.1 and row['mesh_verified']
    a,b=np.linalg.norm(pred[:,:2],axis=1),np.linalg.norm(ref[:,:2],axis=1)
    error=np.linalg.norm((pred-ref)[:,:2],axis=1)
    fig,axes=canvas(f"PI-CFO | {row['label']} | Re={row['reynolds']:g} | velocity L2: {100*ev:.3f}% | pressure L2: {100*ep:.3f}%",
                    ['Input inlet profile','PI-CFO prediction','Independent FEM','Absolute vector error','Cross-section check'])
    iy,iv=inlet_curve(d)
    axes[0].fill_betweenx(iy,0,iv,color='#11687c',alpha=.12);axes[0].plot(iv,iy,color='#11687c',lw=2.5)
    axes[0].set(xlabel='Inlet u / Umean',ylabel='y/H',ylim=(0,1));axes[0].grid(alpha=.2)
    for ax,values in zip(axes[1:3],[a,b]):
        im=ax.pcolormesh(x,y,values.reshape(len(x),len(y)).T,shading='nearest',cmap='viridis',vmin=0,vmax=max(a.max(),b.max()))
        ax.set(xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label='Fluid |U| / Umean')
    im=axes[3].pcolormesh(x,y,error.reshape(len(x),len(y)).T,shading='nearest',cmap='magma',vmin=0)
    axes[3].set(xlabel='x/H',ylabel='y/H');fig.colorbar(im,ax=axes[3],orientation='horizontal',pad=.2,fraction=.07,label='sqrt((du)^2 + (dv)^2) / Umean')
    for target,color in [(.05,'#0a7492'),(.2,'#c8771e'),(1.,'#8448a1')]:
        ix=int(np.argmin(abs(x-target)))
        axes[4].plot(b.reshape(len(x),len(y))[ix],y,color=color,lw=2,label=f'FEM x={x[ix]:.2f}')
        axes[4].plot(a.reshape(len(x),len(y))[ix],y,color=color,lw=1.7,ls='--',label='PI-CFO')
    axes[4].set(xlabel='Fluid |U| / Umean',ylabel='y/H');axes[4].legend(fontsize=9);axes[4].grid(alpha=.2)
    fig.text(.035,.028,'Same frozen 94k checkpoint. Independent mesh-refined FEM. Shared flow color limits. Selected exploratory examples; metrics use the saved entrance-dense grid.',fontsize=11,color='#405263')
    save(fig,HERE/'forward',row['name'])
    return dict(**row,data_sha256=sha(path))

def inverse(source,stem,label):
    origin=source
    local=HERE/'data'/'inverse'/stem
    local.mkdir(parents=True,exist_ok=True)
    needed=['reconstructed_flow.npz','fitted_model.json','inlet_normalized.csv','inlet_reference_units.csv']
    needed += [f'observed_track_{i}.csv' for i in [1,2]]
    needed += [f'track_{i}_forward.csv' for i in [1,2]]
    for name in needed:
        if origin.exists():shutil.copy2(origin/name,local/name)
        assert (local/name).exists()
    source=local
    with np.load(source/'reconstructed_flow.npz',allow_pickle=False) as z:x,y,u,v=[z[k].copy() for k in ['x','y','u','v']]
    model=json.loads((source/'fitted_model.json').read_text())
    inlet=np.genfromtxt(source/'inlet_normalized.csv',delimiter=',',names=True)
    yi=np.linspace(0,1,2049);ui=CubicSpline(inlet['y'],inlet['u'])(yi)*model['mean_speed']
    # These are the selected inferred nodal values, not the generating truth.
    actual=np.genfromtxt(source/'inlet_reference_units.csv',delimiter=',',names=True)
    np.testing.assert_allclose(inlet['u']*model['mean_speed'],actual['u'],rtol=1e-13,atol=1e-14)
    fig,axes=canvas(f"PI-CFO | {label} | inferred Re={model['reynolds']:.3f}",
                    ['Given particle paths','Inferred inlet profile','Inferred carrier flow','Particles in that flow','Input / output overlay','Time-aligned mismatch'])
    axes[1].fill_betweenx(yi,0,ui,color='#11687c',alpha=.12);axes[1].plot(ui,yi,color='#11687c',lw=2.5)
    axes[1].set(xlabel='Inferred inlet u / U_ref',ylabel='y/H',ylim=(0,1));axes[1].grid(alpha=.2)
    axes[1].text(.03,.97,f"Mean = {model['mean_speed']:.4f} U_ref",va='top',transform=axes[1].transAxes,fontsize=10)
    for i in [0,2,3,4]:axes[i].set(xlim=(0,5),ylim=(0,1),xlabel='x/H',ylabel='y/H')
    for ax in axes[2:4]:
        im=ax.pcolormesh(x,y,np.hypot(u,v).T,shading='nearest',cmap='viridis',vmin=0,vmax=np.hypot(u,v).max())
        ax.streamplot(x,y,u.T,v.T,color=(1,1,1,.45),density=(.75,.55),linewidth=.5,arrowsize=.65)
        fig.colorbar(im,ax=ax,orientation='horizontal',pad=.2,fraction=.07,label='Fluid |U| / U_ref')
    metrics=[];hashes={p:sha(source/p) for p in ['fitted_model.json','reconstructed_flow.npz','inlet_normalized.csv','inlet_reference_units.csv']}
    for i,color in [(1,'#f1a300'),(2,'#df42b3')]:
        obs=np.genfromtxt(source/f'observed_track_{i}.csv',delimiter=',',names=True)
        sim=np.genfromtxt(source/f'track_{i}_forward.csv',delimiter=',',names=True)
        np.testing.assert_allclose(obs['t'],sim['t'],atol=1e-12)
        distance=np.hypot(obs['x']-sim['x'],obs['y']-sim['y']);rmse=float(np.sqrt(np.mean(distance**2)))
        axes[0].plot(obs['x'],obs['y'],'.-',color=color,lw=1.3,ms=3,label=f'Input {i}')
        axes[0].scatter(obs['x'][0],obs['y'][0],marker='s',s=40,color=color,edgecolor='black')
        axes[3].plot(sim['x'],sim['y'],color=color,lw=2,path_effects=[pe.Stroke(linewidth=3,foreground='#222'),pe.Normal()])
        for f in [.25,.5,.75]:
            k=int((len(sim)-1)*f);axes[3].scatter(sim['x'][k],sim['y'][k],s=40,color=color,edgecolor='black',zorder=6)
        axes[4].plot(sim['x'],sim['y'],color=color,lw=2,label=f'ODE output {i}')
        axes[4].scatter(obs['x'][::2],obs['y'][::2],s=15,facecolors='none',edgecolors='#222',label=f'Input {i}')
        axes[5].plot(sim['t'],distance,color=color,lw=1.7,label=f'Particle {i}: RMSE {100*rmse:.4f}% H')
        metrics.append(dict(particle=i,position_2d_rmse_percent_H=100*rmse,max_position_error_H=float(distance.max())))
        for p in [f'observed_track_{i}.csv',f'track_{i}_forward.csv']:hashes[p]=sha(source/p)
    axes[0].legend();axes[4].legend(fontsize=9);axes[5].legend(fontsize=9)
    axes[5].set(xlabel='t / (H/U_ref)',ylabel='Euclidean position error / H');axes[5].grid(alpha=.2)
    fig.text(.035,.028,'Panel 02 is the inferred inlet, not the hidden generating profile. Open circles: input paths; lines: independent particle ODE. Path agreement does not establish unique or globally accurate flow.',fontsize=11,color='#405263')
    save(fig,HERE/'inverse',stem)
    # An additional separate image is useful when the six-panel row is too wide.
    single,ax=plt.subplots(figsize=(7,5.5));single.subplots_adjust(left=.15,right=.96,bottom=.15,top=.83)
    ax.plot(ui,yi,lw=2.5,color='#11687c');ax.fill_betweenx(yi,0,ui,alpha=.12,color='#11687c')
    ax.set(xlabel='Inferred inlet u / U_ref',ylabel='y/H',ylim=(0,1),title=f'{label}\nInlet inferred from the two paths');ax.grid(alpha=.2)
    save(single,HERE/'inverse',stem+'_inlet')
    return dict(name=stem,label=label,source=str(origin.relative_to(ROOT)),plot_data=str(local.relative_to(HERE)),mean_speed=model['mean_speed'],reynolds=model['reynolds'],particles=metrics,source_hashes=hashes)

def gallery(selected):
    n=len(selected);cols=3;rows=int(np.ceil(n/cols));fig,axes=plt.subplots(rows,cols,figsize=(15,4.3*rows),squeeze=False)
    for ax,row in zip(axes.flat,selected):
        with np.load(HERE/'data'/f"{row['name']}.npz",allow_pickle=False) as d:y,u=inlet_curve(d)
        ax.plot(y,u,lw=2.5,color='#11687c');ax.fill_between(y,0,u,color='#11687c',alpha=.12)
        ax.set(xlabel='y/H',ylabel='Inlet u / Umean',xlim=(0,1),title=f"{row['label']} | Re={row['reynolds']:g}\nvelocity {100*row['velocity_relative_l2']:.3f}% | pressure {100*row['pressure_relative_l2']:.3f}%")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[n:]:ax.axis('off')
    fig.suptitle('PI-CFO | Diverse inlet profiles with measured low field errors',fontsize=21,fontweight='bold',color='#13394a')
    fig.tight_layout(rect=(0,0,1,.93));save(fig,HERE,'profile_gallery')

def main():
    sweep=json.loads((HERE/'sweep.json').read_text());assert len(sweep['cases'])==len(sweep['declared_candidates']),'Wait for the full FEM sweep'
    # At most one case per profile-shape family, to avoid cosmetic diversity by Re.
    selected=[]
    for group in ['skew','twin','waves','lobes','mixed','shoulder','nine']:
        eligible=[r for r in sweep['cases'] if r['group']==group and r['selected']]
        if eligible:selected.append(min(eligible,key=lambda r:max(r['velocity_relative_l2']/.05,r['pressure_relative_l2']/.1)))
    direct=[forward(r) for r in selected];gallery(selected)
    inverse_rows=[]
    for source,stem,label in [(ROOT/'flow_reconstruction/examples/fem_beta/result','oblique_beta5','Oblique launches'),
        (ROOT/'flow_reconstruction/examples/corner_launch_preconditioned/result','corners_beta5','Corner launches / experiment A'),
        (ROOT/'presentations/defense_2026_11_07/data/inverse_beta4/result','corners_beta4','Corner launches / experiment B')]:
        inverse_rows.append(inverse(source,stem,label))
    (HERE/'metrics.json').write_text(json.dumps(dict(selected_forward_cases=direct,inverse_rows=inverse_rows,model_display_name='PI-CFO',model_lineage='Project-specific modified physics-informed DeepONet',selection='One per shape family; minimize max(velocityL2/5%,pressureL2/10%) among mesh-verified passing cases',new_training=False),indent=2)+'\n')
    with PdfPages(HERE/'all_new_figures.pdf') as pdf:
        for fig,stem in FIGURES:pdf.savefig(fig,facecolor='white')
    thumbs=[]
    for fig,stem in FIGURES:
        if stem.parent.name in ['forward','inverse'] and not stem.name.endswith('_inlet'):
            im=Image.open(stem.with_suffix('.png')).convert('RGB');im.thumbnail((1800,360))
            band=Image.new('RGB',(1840,im.height+50),'#eef3f5');band.paste(im,(20,35));ImageDraw.Draw(band).text((20,10),str(stem.relative_to(HERE)),fill='#13394a');thumbs.append(band)
        plt.close(fig)
    overview=Image.new('RGB',(1840,sum(i.height for i in thumbs)), 'white');top=0
    for im in thumbs:overview.paste(im,(0,top));top+=im.height
    overview.save(HERE/'overview.jpg',quality=90)
    print('Created',len(direct),'distinct forward rows,',len(inverse_rows),'six-panel inverse rows and standalone inlet images',flush=True)

if __name__=='__main__':main()
