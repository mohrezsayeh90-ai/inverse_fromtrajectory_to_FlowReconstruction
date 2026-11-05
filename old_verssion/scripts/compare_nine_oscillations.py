"""Nine full inlet oscillations: graded-mesh FEM and frozen V2 OOD diagnostic.

The ordinary predictor rejects this inlet. Forced raw-network evaluation is
reported explicitly and must not be interpreted as a supported nine-mode solve.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from channel_operator.profiles import Y
from channel_operator.reference import solve_channel
from channel_operator.v2_model import load
from channel_operator.v2_profiles import A, PROJECT
from scripts.render_v2_comparisons import style, NAVY, TEAL, RED

AMPLITUDE, RE = .6, 55.
NORMALIZATION = 1 + 12*AMPLITUDE/(18*np.pi)**2


def inlet(y):
    y = np.asarray(y)
    return 6*y*(1-y)*(1-AMPLITUDE*np.cos(18*np.pi*y))/NORMALIZATION


def norm(field, x, y):
    squares = np.asarray(field)**2
    if squares.ndim == 3:
        squares = squares.sum(-1)
    return float(np.sqrt(np.trapezoid(np.trapezoid(squares, y, axis=1), x, axis=0)))


def errors(pred, ref, x, y):
    e = pred-ref
    rel = lambda z: norm(e[..., z], x, y)/max(norm(ref[..., z], x, y), 1e-15)
    return dict(velocity_relative_l2=rel(slice(0, 2)), u_relative_l2=rel(0),
                v_relative_l2=rel(1), pressure_relative_l2=rel(2),
                v_rmse=norm(e[..., 1], x, y)/np.sqrt((x[-1]-x[0])*(y[-1]-y[0])))


def model_values(model, profile, xy):
    result = []
    with torch.no_grad():
        for points in np.array_split(xy, max(1, int(np.ceil(len(xy)/1024)))):
            result.append(model(torch.tensor(profile[None], dtype=torch.float64),
                                torch.tensor([RE], dtype=torch.float64),
                                torch.tensor(points[None], dtype=torch.float64)).numpy()[0])
    return np.concatenate(result)


def fields_plot(x, y, ref, pred, title, xmax):
    keep = x <= xmax
    x, ref, pred = x[keep], ref[keep], pred[keep]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), layout='constrained')
    for i, field in enumerate(['u / U', 'v / U', 'p / (rho U^2)']):
        low, high = min(ref[..., i].min(), pred[..., i].min()), max(ref[..., i].max(), pred[..., i].max())
        if i == 1:
            high = max(abs(low), abs(high)); low = -high
        for j, values in enumerate([ref[..., i], pred[..., i], pred[..., i]-ref[..., i]]):
            lim = max(np.max(abs(pred[..., i]-ref[..., i])), 1e-12)
            limits = dict(vmin=-lim, vmax=lim) if j == 2 else dict(vmin=low, vmax=high)
            im = axes[i, j].pcolormesh(x, y, values.T, shading='nearest', rasterized=True,
                                       cmap='RdBu_r' if j == 2 or i == 1 else 'viridis', **limits)
            axes[i, j].set(title=field+'\n'+['FEM: requested 9-peak inlet', 'V2 forced OOD evaluation', 'V2 - FEM'][j],
                           xlabel='x / H', ylabel='y / H', xlim=(0, xmax), ylim=(0, 1))
            if j == 1:
                fig.colorbar(im, ax=axes[i, :2], shrink=.8, pad=.01, format='%.3g')
            elif j == 2:
                fig.colorbar(im, ax=axes[i, j], shrink=.8, pad=.01, format='%.3g')
    fig.suptitle(title+'\nRe=55 | frozen 70k checkpoint | requested inlet is outside the 4-mode family',
                 fontsize=15, fontweight='bold')
    fig.supxlabel('The raw V2 model does not impose the requested 9-peak boundary. '
                  'This is an OOD failure diagnostic, not a validated solution.', fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/v2/nine_oscillations')
    args = parser.parse_args()
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    style(); torch.set_num_threads(4)
    checkpoint = ROOT/'models/v2/stream_operator.pt'
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, state = load(checkpoint); model.double()
    dense_y = np.linspace(0, 1, 5001); requested = inlet(dense_y)
    assert len(find_peaks(requested)[0]) == 9
    assert requested.min() >= 0 and abs(np.trapezoid(requested, dense_y)-1) < 1e-6
    with np.load(ROOT/'data/v2/train.npz') as data:
        training_max = float(data['profiles'].max())
    assert requested.max() < training_max and 10 <= RE <= 100
    # Exact function to FEM; unmodified 33 samples to the original network branch.
    profile = inlet(Y)
    spline_min = float(CubicSpline(Y, profile)(dense_y).min())
    projection_error = float(np.linalg.norm(A@(PROJECT@profile)-profile)/np.linalg.norm(profile))
    np.savetxt(output/'inlet.csv', np.column_stack([dense_y, requested]), delimiter=',',
               header='y,u', comments='', fmt='%.12g')
    rejection = subprocess.run([sys.executable, '-m', 'channel_operator.v2_predict',
        '--checkpoint', str(checkpoint), '--re', str(RE), '--inlet-csv', str(output/'inlet.csv'),
        '--output', str(output/'supported_prediction.npz')], cwd=ROOT, capture_output=True, text=True)
    assert rejection.returncode != 0 and any(reason in rejection.stderr for reason in
        ['outside the four-mode inlet family', 'Supported inlets must be nonnegative'])
    (output/'predictor_rejection.txt').write_text(rejection.stderr)
    x = np.unique(np.r_[np.linspace(0, .03, 13), np.linspace(.03, .3, 28), np.linspace(.3, 5, 48)])
    y = np.linspace(0, 1, 181)
    xx, yy = np.meshgrid(x, y, indexing='ij'); xy = np.stack([xx, yy], -1).reshape(-1, 2)
    prediction = model_values(model, profile, xy).reshape(len(x), len(y), 3)
    model_inlet = model_values(model, profile, np.column_stack([np.zeros_like(dense_y), dense_y]))
    inlet_error = float(np.sqrt(np.trapezoid((model_inlet[:, 0]-requested)**2, dense_y)
                                 /np.trapezoid(requested**2, dense_y)))
    references, mesh_records = [], []
    converged = False
    for nx, ny in [(64, 72), (80, 108), (104, 144), (128, 192)]:
        cache = output/f'reference_{nx}x{ny}.npz'
        if cache.exists():
            with np.load(cache) as d:
                assert np.array_equal(d['x'], x) and np.array_equal(d['y'], y)
                assert float(d['reynolds']) == RE and float(d['amplitude']) == AMPLITUDE
                reference = d['fields'].copy(); info = json.loads(str(d['metadata']))
        else:
            start = time.perf_counter()
            print(f'FEM solve: {nx} x {ny}, inlet-graded x mesh, Re={RE:g}', flush=True)
            evaluate, info = solve_channel(profile, RE, inlet_function=inlet,
                                           x_nodes=5*np.linspace(0, 1, nx+1)**2,
                                           y_nodes=np.linspace(0, 1, ny+1))
            reference = np.concatenate([evaluate(points) for points in np.array_split(xy, int(np.ceil(len(xy)/512)))])
            reference = reference.reshape(len(x), len(y), 3)
            info['seconds'] = time.perf_counter()-start
            np.savez_compressed(cache, x=x, y=y, fields=reference, metadata=json.dumps(info),
                                reynolds=RE, amplitude=AMPLITUDE)
        if references:
            info['change_from_previous_mesh'] = errors(reference, references[-1], x, y)
            entrance = x <= .3
            info['entrance_change'] = errors(reference[entrance], references[-1][entrance], x[entrance], y)
            diff, ent = info['change_from_previous_mesh'], info['entrance_change']
            converged = (len(references) >= 2 and diff['velocity_relative_l2'] < .01
                         and diff['pressure_relative_l2'] < .02
                         and ent['velocity_relative_l2'] < .02 and ent['pressure_relative_l2'] < .03)
        mesh_records.append(info); references.append(reference)
        print(json.dumps(info), flush=True)
        if converged:
            break
    reference = references[-1]
    metrics = errors(prediction, reference, x, y)
    for label, region in [('entrance_x_le_0.3', x <= .3), ('downstream_x_ge_1', x >= 1)]:
        metrics[label] = errors(prediction[region], reference[region], x[region], y)
    fem_inlet_error = float(np.sqrt(np.trapezoid((reference[0, :, 0]-inlet(y))**2, y)/np.trapezoid(inlet(y)**2, y)))
    report = dict(case='Nine full oscillations / nine interior velocity peaks', reynolds=RE, amplitude=AMPLITUDE,
        inlet_formula='6*y*(1-y)*(1-0.6*cos(18*pi*y))/(1+7.2/(18*pi)^2)',
        inlet_peak_count=9, input_sensor_count=33, inlet_mean=float(np.trapezoid(requested, dense_y)),
        inlet_min=float(requested.min()), inlet_max=float(requested.max()), training_sample_max=training_max,
        scalar_ranges_respected=True, within_trained_function_family=False,
        standard_predictor_rejects=True, evaluation='forced_raw_network_OOD_diagnostic',
        sensor_spline_min=spline_min, four_mode_sensor_fit_relative_l2=projection_error,
        same_boundary_value_problem=False, checkpoint_sha256=sha, step=state['step'], training_performed=False,
        inlet_velocity_relative_l2=inlet_error, fem_inlet_interpolation_relative_l2=fem_inlet_error,
        model_inlet_peak_count=int(len(find_peaks(model_inlet[:, 0])[0])),
        reference_mesh_converged=converged, reference_mesh_criteria=dict(velocity=.01, pressure=.02,
            entrance_velocity=.02, entrance_pressure=.03, minimum_meshes=3),
        mesh_refinement=mesh_records, metrics=metrics,
        norm='Trapezoid quadrature over actual nonuniform x and uniform y; no pressure alignment')
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/'comparison_fields.npz', x=x, y=y, reference=reference, prediction=prediction,
                        inlet_y=dense_y, inlet_requested=requested, inlet_model=model_inlet[:, 0],
                        reynolds=RE, amplitude=AMPLITUDE)
    with (output/'pointwise_fields.csv').open('w', newline='') as f:
        writer = csv.writer(f);writer.writerow(['x','y','u_FEM','v_FEM','p_FEM','u_model','v_model','p_model'])
        writer.writerows(np.column_stack([xy, reference.reshape(-1,3), prediction.reshape(-1,3)]))
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
    axes[0].plot(requested, dense_y, color=NAVY, label='Requested 9-peak inlet', lw=2)
    axes[0].plot(model_inlet[:, 0], dense_y, color=RED, label='Actual V2 boundary', lw=2)
    axes[0].plot(reference[0, :, 0], y, '.', color=TEAL, ms=3, label='FEM boundary samples')
    axes[0].set(title=f'Inlet relative L2 mismatch: {100*inlet_error:.2f}%', xlabel='u_in / U', ylabel='y / H')
    axes[0].legend(fontsize=8)
    axes[1].plot(dense_y, requested, color=NAVY, lw=1.7)
    axes[1].plot(Y, profile, 'o', color=TEAL, ms=4, label='33 branch input sensors')
    axes[1].set(title='Nine full cycles: cos(18 pi y)', xlabel='y / H', ylabel='u_in / U')
    axes[1].legend(fontsize=8)
    labels = ['Full channel', 'Entrance x <= 0.3', 'Downstream x >= 1']
    scope = [metrics, metrics['entrance_x_le_0.3'], metrics['downstream_x_ge_1']]
    pos = np.arange(3)
    axes[2].bar(pos-.18, [100*m['velocity_relative_l2'] for m in scope], .36, color=TEAL, label='Velocity vector L2')
    axes[2].bar(pos+.18, [100*m['pressure_relative_l2'] for m in scope], .36, color=RED, label='Pressure L2')
    axes[2].set(xticks=pos, xticklabels=labels, ylabel='Error (%)', title='Forced V2 vs requested-inlet FEM')
    axes[2].tick_params(axis='x', labelsize=8); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.6); ax.set_axisbelow(True)
    fig.suptitle('NINE-OSCILLATION INLET  /  SAME SCALAR RANGE, NEW FUNCTION FAMILY\n'
                 'Re=55 | mean inlet=1 | no per-case training | ordinary predictor rejects this inlet',
                 fontsize=15, fontweight='bold')
    figures.append(('inlet_and_errors', fig))
    figures.append(('full_fields', fields_plot(x, y, reference, prediction, 'Full-channel comparison', 5.)))
    figures.append(('entrance_fields', fields_plot(x, y, reference, prediction, 'Entrance zoom', .3)))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
    for j, target in enumerate([.005, .03, .1, 1.]):
        k = int(np.argmin(abs(x-target)))
        for i, label in enumerate(['u / U', 'v / U', 'p / (rho U^2)']):
            ax = axes[i,j]
            ax.plot(reference[k,:,i], y, color=NAVY, lw=1.7, label='FEM: requested inlet')
            ax.plot(prediction[k,:,i], y, '--', color=RED, lw=1.5, label='V2 forced OOD')
            ax.set(title=f'x / H = {x[k]:.4f}', xlabel=label, ylabel='y / H', ylim=(0,1))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,3), useOffset=True)
            ax.tick_params(axis='x', labelsize=8); ax.grid(alpha=.6)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Cross-sections: requested nine-oscillation FEM vs forced V2 evaluation\n'
                 'The inlet boundary conditions differ; frozen V2 is not a valid solution of this new problem.',
                 fontsize=14, fontweight='bold')
    figures.append(('cross_sections', fig))
    fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    for key, label in [('velocity_relative_l2','Velocity'), ('pressure_relative_l2','Pressure')]:
        pairs = mesh_records[1:]
        axes[0].plot([f"{r['nx']}x{r['ny']}" for r in pairs],
                     [100*r['change_from_previous_mesh'][key] for r in pairs], 'o-', label=label)
        axes[1].plot([f"{r['nx']}x{r['ny']}" for r in pairs],
                     [100*r['entrance_change'][key] for r in pairs], 'o-', label=label)
    for ax, title in zip(axes, ['Full domain', 'Entrance x <= 0.3']):
        ax.set(title=title, ylabel='Change from previous mesh (%)', xlabel='Refined mesh intervals (graded x, uniform y)')
        ax.legend();ax.grid(alpha=.6)
    fig.suptitle(f'Independent FEM refinement | convergence criteria met: {converged}', fontsize=15, fontweight='bold')
    figures.append(('mesh_convergence', fig))
    with PdfPages(output/'nine_oscillations.pdf') as pdf:
        for name, fig in figures:
            fig.savefig(output/f'{name}.png', dpi=150);pdf.savefig(fig);plt.close(fig)
    mesh_table = '\n'.join(f"| {r['nx']}×{r['ny']} | {r['iterations']} | {r['nonlinear_residual']:.2e} | "
                           + (f"{100*r['change_from_previous_mesh']['velocity_relative_l2']:.3f} | {100*r['change_from_previous_mesh']['pressure_relative_l2']:.3f} |" if 'change_from_previous_mesh' in r else '— | — |') for r in mesh_records)
    metric_table = '\n'.join(f"| {label} | {100*m['velocity_relative_l2']:.3f} | {100*m['pressure_relative_l2']:.3f} | {100*m['v_relative_l2']:.3f} | {m['v_rmse']:.5f} |" for label, m in zip(['کل کانال', 'ورودی x≤0.3', 'پایین‌دست x≥1'], scope))
    text = f'''# آزمایش پروفایل ورودی با ۹ نوسان کامل

این آزمایش **۹ قلهٔ سرعت در عرض کانال** دارد؛ منظور ۹ سیکل کامل است، نه مود cos(9πy) که فقط ۴٫۵ سیکل دارد.

`u_in(y) = 6 y (1-y) [1 - 0.6 cos(18πy)] / [1 + 7.2/(18π)²]`

Re=55، میانگین ورودی ۱، کمینه صفر و بیشینه **{requested.max():.4f}** است. بیشینهٔ نمونه‌های ورودی آموزشی {training_max:.4f} است.
دامنهٔ سرعت و Re در بازهٔ قبلی هستند، ولی **فرکانس و خانوادهٔ تابع ورودی خارج از دامنهٔ آموزش چهارمودی‌اند**.

## نتیجهٔ اصلی

این CSV از پیش‌بینی‌کنندهٔ معمول عبور نمی‌کند: [متن خطا](predictor_rejection.txt).
با اینکه تابع واقعی نامنفی است، بازسازی cubic spline از ۳۳ حسگر نزدیک دیواره به کمینهٔ {spline_min:.5f} می‌رسد و ابتدا کنترل نامنفی‌بودن شکست می‌خورد.
جدا از این مشکل نمونه‌برداری، خطای برازش چهارمودی روی حسگرها {100*projection_error:.2f}٪ است؛ بنابراین رفع overshoot هم پشتیبانی از این خانواده را فراهم نمی‌کند.
برای تشخیص محدودیت، همین ۳۳ نمونهٔ پروفایل بدون تغییر به شبکهٔ خام داده شدند؛ وزن‌ها ثابت ماندند.
معماری V2 در x=0 فقط تصویر چهارمودی پروفایل را اعمال می‌کند؛ بنابراین خطای شرط ورودی **{100*inlet_error:.2f}٪** است.
پس خروجی خام شبکه **حل معتبر مسئلهٔ ۹نوسانی نیست** و مقایسهٔ زیر یک آزمون شکست خارج از خانواده است، نه موفقیت تعمیم.
همچنین میدان‌های مقایسه‌شده شرط مرزی یکسان ندارند؛ این اختلاف عمداً اندازه‌گیری و آشکار شده است.

![ورودی و خطاها](inlet_and_errors.png)

| ناحیه | خطای بردار سرعت ٪ | خطای فشار ٪ | خطای نسبی v ٪ | RMSE مؤلفه v نسبت به U |
|---|---:|---:|---:|---:|
{metric_table}

خطا با انتگرال‌گیری ذوزنقه‌ای روی مختصات واقعی شبکهٔ غیر یکنواخت محاسبه شده است؛ تراکم زیاد نمونه‌ها نزدیک ورودی، وزن کل آن ناحیه را مصنوعی زیاد نمی‌کند. فشار پس از حل جابه‌جا یا برازش نشده است.
خطای نسبی v در پایین‌دست بسیار بزرگ می‌شود چون خودِ v مرجع بسیار کوچک است؛ RMSE مطلق برای تفسیر این عدد کنار آن آمده است.

## مرجع عددی مستقل و بررسی مش

حل Taylor–Hood P2/P1 با شرط ورودی تحلیلیِ دقیق، دیوارهٔ بدون لغزش و خروجی طبیعی انجام شده است.
مش در جهت x نزدیک ورودی ریزتر است: `x_i=5(i/nx)^2`؛ جهت y یکنواخت است.
این اندازه‌ها تعداد بازه‌های مش‌اند؛ تعداد درجات آزادی P2/P1 بیشتر است.
معیار ریزسازی: اختلاف دو مش آخر در کل کانال کمتر از ۱٪ سرعت و ۲٪ فشار، و نزدیک ورودی کمتر از ۲٪ سرعت و ۳٪ فشار، با حداقل سه مش.
**وضعیت معیار ریزسازی: {converged}.** این برآورد ریزسازی است، نه کران ریاضی خطا.
خطای درون‌یابی شرط ورودی FEM روی نقاط مقایسه: {100*fem_inlet_error:.4f}٪.

| مش | تکرار Picard | باقی‌ماندهٔ جبری غیرخطی | تغییر سرعت نسبت به مش قبل ٪ | تغییر فشار نسبت به مش قبل ٪ |
|---|---:|---:|---:|---:|
{mesh_table}

![همگرایی مش](mesh_convergence.png)

## مقایسهٔ میدان‌ها

![کل کانال](full_fields.png)

![بزرگ‌نمایی ورودی](entrance_fields.png)

![مقاطع کانال](cross_sections.png)

## فایل‌ها و بازتولید

- [PDF پنج‌صفحه‌ای](nine_oscillations.pdf)
- [پروفایل ورودی CSV](inlet.csv)
- [اعداد و متادیتا](metrics.json)
- [میدان مدل و مرجع NPZ](comparison_fields.npz)
- [مقادیر نقطه‌ای CSV](pointwise_fields.csv)

```bash
python scripts/compare_nine_oscillations.py
```

وزن مدل تغییر نکرده است: `{sha}`. هیچ داده یا پاسخی از این آزمایش برای آموزش استفاده نشده است.
برای پشتیبانی واقعی از چنین پروفایلی باید نمایش شرط مرزی و پوشش خانوادهٔ ورودی گسترش یابد و مدل دوباره به‌صورت عمومی آموزش و ارزیابی شود.
'''
    (output/'README.md').write_text(text)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
