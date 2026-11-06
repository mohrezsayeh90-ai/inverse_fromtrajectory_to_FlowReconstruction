"""Plot saved reconstructed fluid and particle motion together, without fitting."""
import hashlib,json,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'flow_reconstruction/trajectory_comparisons'
SOURCES=['development/trajectory_only_initial','fem_beta','corner_launch','corner_launch_preconditioned']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    produced={str((OUT/f'case_{i}'/f'flow_and_particles.{ext}').relative_to(ROOT)) for i in range(1,5) for ext in ['png','pdf','json']}
    produced|={str((OUT/name).relative_to(ROOT)) for name in ['FLOW_AND_PARTICLES.md','flow_views_manifest.json']}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    protected={name:sha(ROOT/name) for name in tracked if name and name not in produced}
    for index,folder in enumerate(SOURCES,1):
        source=ROOT/'flow_reconstruction/examples'/folder/'result';target=OUT/f'case_{index}'
        target.mkdir(parents=True,exist_ok=True)
        with np.load(source/'reconstructed_flow.npz',allow_pickle=False) as data:
            x,y,u,v=[data[name].copy() for name in ['x','y','u','v']]
        fit=json.loads((source/'fitted_model.json').read_text())
        metrics=json.loads((target/'metrics.json').read_text())
        assert sha(source/'fitted_model.json')==metrics['saved_fit_sha256']
        assert fit['checkpoint_sha256']==metrics['checkpoint_sha256']
        assert u.shape==v.shape==(len(x),len(y)) and np.isfinite(u).all() and np.isfinite(v).all()
        speed=np.hypot(u,v)
        fig=plt.figure(figsize=(15,9),layout='constrained')
        grid=fig.add_gridspec(2,2,height_ratios=[1.1,1])
        flow=fig.add_subplot(grid[0,:]);paths=fig.add_subplot(grid[1,0]);errors=fig.add_subplot(grid[1,1])
        im=flow.pcolormesh(x,y,speed.T,shading='nearest',cmap='viridis',vmin=0)
        fig.colorbar(im,ax=flow,label='Reconstructed fluid speed |U| / U_ref',fraction=.025,pad=.015)
        flow.streamplot(x,y,u.T,v.T,color=(1,1,1,.55),density=(1.8,.7),linewidth=.65,arrowsize=.8)
        hashes={'flow_grid':sha(source/'reconstructed_flow.npz'),'fitted_model':sha(source/'fitted_model.json')}
        for particle,(color,label) in enumerate([('#ffb000','Upper-left particle'),('#ff51ce','Lower-right particle')],1):
            observed_path=source/f'observed_track_{particle}.csv';predicted_path=source/f'track_{particle}_forward.csv'
            observed=np.genfromtxt(observed_path,delimiter=',',names=True)
            predicted=np.genfromtxt(predicted_path,delimiter=',',names=True)
            assert len(observed)==len(predicted)
            np.testing.assert_allclose(observed['t'],predicted['t'],atol=1e-12,rtol=0)
            distance=np.hypot(predicted['x']-observed['x'],predicted['y']-observed['y'])
            rmse=float(np.sqrt(np.mean(distance**2)))
            np.testing.assert_allclose(rmse,metrics['particles'][particle-1]['position_2d_rmse_H'],atol=1e-12)
            for axis in [flow,paths]:
                axis.plot(predicted['x'],predicted['y'],color=color,lw=2.3,label=label+' in reconstructed flow',
                          path_effects=[pe.Stroke(linewidth=3.3,foreground='#222222'),pe.Normal()],zorder=5)
                axis.scatter(observed['x'][::2],observed['y'][::2],s=13,facecolors='none',edgecolors='white' if axis is flow else '#222222',
                             linewidths=.75,label=label+' input positions',zorder=6)
                axis.scatter(predicted['x'][0],predicted['y'][0],marker='s',s=55,color=color,edgecolors='black',zorder=7)
            # Positions and directions of the actual inertial particle at three
            # times; arrow lengths show direction only, not velocity magnitude.
            for fraction in [.25,.5,.75]:
                k=int(round(fraction*(len(predicted)-1)));velocity=np.array([predicted['vx'][k],predicted['vy'][k]])
                direction=.16*velocity/max(np.linalg.norm(velocity),1e-12)
                point=np.array([predicted['x'][k],predicted['y'][k]])
                flow.scatter(*point,s=65,color=color,edgecolors='black',zorder=8)
                flow.annotate('',xy=point+direction,xytext=point,arrowprops=dict(arrowstyle='->',color=color,lw=2),zorder=9)
            errors.plot(predicted['t'],distance,color=color,lw=1.5,label=f'{label}: 2D RMSE={100*rmse:.4f}% H')
            hashes[f'particle_{particle}_input']=sha(observed_path);hashes[f'particle_{particle}_ODE']=sha(predicted_path)
        for axis in [flow,paths]:
            axis.set(xlim=(x[0],x[-1]),ylim=(0,1),xlabel='x/H',ylabel='y/H',aspect='equal')
        flow.set_title('Reconstructed fluid + particles moving in that SAME reconstructed fluid',fontsize=13)
        paths.set_title('Input positions overlaid on particle paths',fontsize=11)
        paths.legend(fontsize=7,loc='lower left');paths.grid(alpha=.2)
        errors.set(title='Time-aligned particle-position mismatch',xlabel='t / (H/U_ref)',ylabel='Distance from input path / H')
        errors.legend(fontsize=8);errors.grid(alpha=.2)
        fig.suptitle(f"Saved inverse result {index} | mean speed={fit['mean_speed']:.4f} U_ref | Re={fit['reynolds']:.3f}\n"
                     'Background: fluid speed; white arrows: fluid streamlines; colored paths/circles: inertial particles; open dots: input data',fontsize=12)
        fig.savefig(target/'flow_and_particles.png',dpi=180)
        fig.savefig(target/'flow_and_particles.pdf');plt.close(fig)
        report=dict(saved_result=str(source.relative_to(ROOT)),source_sha256=hashes,model_refitted=False,
                    new_flow_prediction=False,new_particle_simulation=False,
                    description='Direct visualization of the previously saved inferred fluid grid and its own saved forward-ODE particle paths; no reference field comparison.')
        (target/'flow_and_particles.json').write_text(json.dumps(report,indent=2)+'\n')
    changed=[name for name,digest in protected.items() if sha(ROOT/name)!=digest];assert not changed,changed
    text=['# سیال بازسازی‌شده و ذرات در همان سیال','','پس‌زمینه، اندازهٔ سرعت **همان سیال تخمینی ذخیره‌شده** است. پیکان‌های سفید جهت جریان سیال را نشان می‌دهند. خط‌ها، دایره‌ها و پیکان‌های رنگی، مسیر و جهت حرکت ذرات اینرسی‌دار در آن سیال‌اند؛ نقطه‌های توخالی، مسیر ورودی هستند.',
          '', 'ذره به دلیل اینرسی الزاماً روی خط جریان حرکت نمی‌کند؛ برگشت جهت ذرهٔ پرتاب‌شده از راست در تصویر دیده می‌شود. دایره‌های رنگی، مکان ذرات در یک‌چهارم، نصف و سه‌چهارم بازهٔ مشاهده هستند. طول پیکان ذرات فقط برای نمایش جهت است.',
          '', 'در هر شکل، پایینِ تصویر مسیرهای ورودی/خروجی روی هم قرار گرفته‌اند و فاصلهٔ آن‌ها در زمان یکسان نمایش داده شده است. همهٔ داده‌ها از جواب‌های قبلی خوانده شده‌اند؛ مدل، پروفایل، میدان و مسیرهای قبلی تغییر نکرده‌اند.']
    for i in range(1,5):text+=['',f'## جواب ذخیره‌شدهٔ {i}','',f'![سیال بازسازی‌شده، ذرات و مقایسهٔ مسیر](case_{i}/flow_and_particles.png)','',f'[PDF](case_{i}/flow_and_particles.pdf) · [منشأ و هش داده‌های تصویر](case_{i}/flow_and_particles.json)']
    text+=['', '[جدول و نمودارهای تفصیلی مقایسهٔ مسیر](README.md)','','بازتولید: `python -m scripts.plot_reconstructed_fluid_and_particles`']
    (OUT/'FLOW_AND_PARTICLES.md').write_text('\n'.join(text)+'\n')
    manifest={name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(produced) if (ROOT/name).exists() and not name.endswith('flow_views_manifest.json')}
    (OUT/'flow_views_manifest.json').write_text(json.dumps(dict(artifacts=manifest,
        protected_existing_files=len(protected),all_previous_files_unchanged=True),indent=2)+'\n')
    print('Saved four fluid-plus-particle figures; previous files unchanged:',len(protected))


if __name__=='__main__':main()
