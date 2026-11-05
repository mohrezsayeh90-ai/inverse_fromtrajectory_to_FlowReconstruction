"""Reproduce a 24-case visual atlas using frozen weights and stored FEM references.

No training, reference interpolation, pressure alignment, or case selection is used.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from channel_operator.v2_model import load
from channel_operator.v2_profiles import inlet_values

NAVY, TEAL, RED = '#142b49', '#008b8b', '#e3584f'


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelcolor': NAVY, 'text.color': NAVY,
        'axes.edgecolor': '#ced8e2', 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'grid.color': '#e2e8ee',
        'pdf.fonttype': 42,
    })


def caption(row):
    a = ', '.join(f'{v:+.3f}' for v in row['coefficients'])
    return (f"{row['id']}  |  Re = {row['reynolds']:g}  |  a = [{a}]\n"
            f"Velocity L2: {row['velocity_pct']:.3f}%    Pressure L2: {row['pressure_pct']:.3f}%"
            f"    v RMSE: {row['v_rmse']:.5f} U")


def field_figure(row, x, y, ref, pred):
    # Plot the native stored FEM sample grid, never an invented high-resolution reference.
    fig, axes = plt.subplots(4, 3, figsize=(15, 10.5), layout='constrained')
    fields = [('Streamwise velocity u / U', ref[..., 0], pred[..., 0]),
              ('Transverse velocity v / U', ref[..., 1], pred[..., 1]),
              ('Pressure p / (rho U^2)', ref[..., 2], pred[..., 2]),
              ('Speed |V| / U', np.linalg.norm(ref[..., :2], axis=-1),
               np.linalg.norm(pred[..., :2], axis=-1))]
    for k, (label, truth, estimate) in enumerate(fields):
        low, high = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
        if k == 1:
            high = max(abs(low), abs(high)); low = -high
        for j, data in enumerate([truth, estimate, estimate-truth]):
            ax = axes[k, j]
            limit = max(np.max(np.abs(estimate-truth)), 1e-12)
            limits = {'vmin': -limit, 'vmax': limit} if j == 2 else {'vmin': low, 'vmax': high}
            im = ax.pcolormesh(x, y, data.T, shading='nearest', rasterized=True,
                               cmap='RdBu_r' if j == 2 or k == 1 else 'viridis', **limits)
            ax.set(xlim=(0, 5), ylim=(0, 1), xlabel='x / H', ylabel='y / H')
            ax.set_title(f'{label}\n' + ['FEM reference', 'PINN / DeepONet', 'Signed error: model - FEM'][j])
            if j == 1:
                fig.colorbar(im, ax=axes[k, :2], shrink=.78, pad=.015, format='%.3g')
            elif j == 2:
                fig.colorbar(im, ax=ax, shrink=.78, pad=.015, format='%.2g')
    fig.suptitle(caption(row), fontsize=14, fontweight='bold')
    fig.supxlabel('Shared color scale for FEM and model in each row; error scale is separate. '
                  f'Native comparison samples: {len(x)} x {len(y)}. All fields dimensionless.', fontsize=9)
    return fig


def profile_figure(row, x, y, ref, pred):
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5), layout='constrained')
    indices = [int(np.argmin(abs(x-v))) for v in [.1, .5, 2., 4.9]]
    for i, label in enumerate(['u / U', 'v / U', 'p / (rho U^2)']):
        for j, xi in enumerate(indices):
            ax = axes[i, j]
            ax.plot(ref[xi, :, i], y, 'o', ms=3, mfc='white', mec=NAVY, label='FEM samples')
            ax.plot(pred[xi, :, i], y, color=TEAL, lw=1.7, label='PINN / DeepONet')
            ax.set(xlabel=label, ylabel='y / H', ylim=(0, 1), title=f'x / H = {x[xi]:.3f}')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='x', labelsize=8)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3), useOffset=True)
            ax.grid(alpha=.7)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle(caption(row), fontsize=14, fontweight='bold')
    fig.supxlabel('Cross-sections use the nearest available x sample; its actual coordinate is printed. '
                  'Identical pressure datum; no pressure offset fitted.', fontsize=9)
    return fig


def overview(rows):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    y = np.linspace(0, 1, 257)
    for group, ax in zip(['blind', 'monitor'], axes[0]):
        cases = [r for r in rows if r['set'] == group]
        colors = plt.cm.tab20(np.linspace(0, 1, len(cases)))
        for r, color in zip(cases, colors):
            ax.plot(inlet_values(r['coefficients'], y), y, color=color,
                    label=f"{r['id']} Re={r['reynolds']:g}", lw=1.4)
        ax.set(title=('Final blind tests' if group == 'blind' else 'Repeated validation') + ': inlet profiles',
               xlabel='u_in / U', ylabel='y / H', ylim=(0, 1))
        ax.grid(alpha=.7)
        ax.legend(fontsize=6.5, ncol=2, loc='best')
    labels = [r['id'] for r in rows]
    for ax, key, title in [(axes[1, 0], 'velocity_pct', 'Velocity vector relative L2 error'),
                           (axes[1, 1], 'pressure_pct', 'Pressure relative L2 error')]:
        ax.bar(np.arange(len(rows)), [r[key] for r in rows],
               color=[TEAL if r['set'] == 'blind' else '#879bb2' for r in rows])
        ax.set_xticks(np.arange(len(rows)), labels, rotation=75, fontsize=7)
        ax.set(title=title, ylabel='Error (%)')
        ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    fig.suptitle('CHANNEL FLOW  /  FIXED OPERATOR vs FEM\n'
                 '24 held-out conditions | 70,000 training updates | no per-case training',
                 fontsize=17, fontweight='bold')
    fig.supxlabel('B01-B08: final blind tests. M01-M16: repeatedly monitored validation. '
                  'Field accuracy does not imply all PDE residual gates pass.', fontsize=10)
    return fig


def make_dashboard(output, rows, sha):
    # Embed plots so a downloaded HTML file works without a server or adjacent assets.
    cases = []
    for r in rows:
        entry = dict(r)
        for kind in ['fields', 'profiles']:
            entry[kind] = 'data:image/png;base64,' + base64.b64encode(
                (output/'cases'/f"{r['id']}_{kind}.png").read_bytes()).decode()
        cases.append(entry)
    overview_uri = 'data:image/png;base64,' + base64.b64encode((output/'overview.png').read_bytes()).decode()
    html = '''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>مقایسه مدل جریان کانال با FEM</title>
<style>
body{margin:0;background:#f1f5f9;color:#142b49;font:16px/1.8 Tahoma,Arial,sans-serif}
main{max-width:1440px;margin:auto;padding:28px}header{background:#142b49;color:white;padding:28px;border-radius:18px}
h1{font-size:28px;margin:0}h2{font-size:21px}.muted{opacity:.8}.tag{display:inline-block;background:#22445c;padding:4px 13px;border-radius:20px;margin:5px}
.card{background:white;padding:22px;margin-top:20px;border-radius:16px;box-shadow:0 4px 20px #142b4909}
.warning{border-right:5px solid #e6a141}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
select,button{font:inherit;padding:8px 14px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:#142b49}
button.active{background:#008b8b;color:white}.plot{width:100%;height:auto;display:block}.ltr{direction:ltr;unicode-bidi:embed}
table{width:100%;border-collapse:collapse;text-align:center;font-size:14px}td,th{padding:9px;border-bottom:1px solid #e2e8f0}th{background:#edf4f8}
tbody tr{cursor:pointer}tbody tr:hover{background:#e5f5f3}.scroll{overflow:auto}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#edf7f7;padding:10px 18px;border-radius:10px}
small{font-size:12px}code{overflow-wrap:anywhere}summary{cursor:pointer}
</style><main><header><div class="muted">CHANNEL FLOW · PHYSICS-INFORMED DEEPONET</div>
<h1>مقایسهٔ مدل با حل عددی مرجع</h1><p>تمام ۲۴ حالت خارج از داده‌های آموزش؛ یک مدل ثابت و بدون آموزش مجدد برای هر ورودی</p>
<span class="tag">۸ آزمون نهایی</span><span class="tag">۱۶ حالت پایش</span><span class="tag">Re = 10–100</span><span class="tag">کانال ۵ × ۱</span></header>
<div class="card warning">مدل هیبریدی FEM + معادلات فیزیک است. معیار سخت‌گیرانهٔ باقی‌ماندهٔ مومنتوم در بعضی حالت‌ها پاس نشده است؛ دقت خوب میدان به معنی عبور از همهٔ معیارهای فیزیکی نیست.
حالت‌های M در طول توسعه پایش شده‌اند؛ حالت‌های B پس از ثابت‌شدن وزن‌ها ارزیابی شدند. رسم دوباره هیچ وزنی را تغییر نمی‌دهد.</div>
<div class="card"><h2>نمای کلی پروفایل‌ها و خطاها</h2><img class="plot" src="OVERVIEW_URI" alt="All inlet profiles and errors"></div>
<section class="card" id="viewer"><h2>بررسی هر حالت</h2><div class="controls"><label>مجموعه <select id="group"><option value="all">همهٔ حالت‌ها</option><option value="blind">آزمون نهایی B</option><option value="monitor">پایش M</option></select></label>
<label>حالت <select id="case"></select></label><button id="fields" class="active">میدان‌ها و اختلاف</button><button id="profiles">مقاطع کانال</button></div>
<p id="condition" class="ltr"></p><div id="stats" class="stats"></div><p id="physics"></p><img id="plot" class="plot" alt="Selected comparison">
<small>مقیاس رنگ مرجع و مدل در هر ردیف یکسان است؛ خطا مقیاس جداگانه دارد. تمام مقادیر بی‌بعدند. مقاطع بر نقاط اصلی حل مرجع رسم شده‌اند.</small></section>
<div class="card"><h2>جدول تمام حالت‌ها</h2><p>روی هر ردیف کلیک کنید تا نمودار آن نمایش داده شود.</p><div class="scroll"><table class="ltr"><thead><tr><th>Case</th><th>Re</th><th>Velocity L2 %</th><th>u L2 %</th><th>v L2 %</th><th>v RMSE / U</th><th>Pressure L2 %</th><th>Momentum RMS</th></tr></thead><tbody id="table"></tbody></table></div></div>
<div class="card"><details><summary>روش محاسبه و بازتولید</summary><p>مرجع، حل اجزای محدود Taylor–Hood است. مقایسه روی شبکهٔ نمونه‌برداری اصلی ۴۹ × ۲۵ انجام می‌شود؛ این اندازهٔ مش اجزای محدود نیست. هیچ انتقال دلخواه فشار یا هموارسازی مرجع انجام نشده است.</p>
<p>خطای نسبی L2 برابر نسبت نرم اختلاف به نرم پاسخ مرجع روی همین نقاط است. خطای سرعت برای بردار (u,v) محاسبه می‌شود. به‌دلیل کوچک‌بودن v، خطای نسبی آن همراه RMSE مطلق ارائه شده است.</p>
<p>پروفایل‌های ورودی با خانوادهٔ چهارمودی تعریف می‌شوند؛ این نتایج تضمین هر پروفایل دلخواه یا جریان آشفته نیستند. معیار مومنتوم از ارزیابی ثبت‌شدهٔ همین وزن‌ها نقل می‌شود.</p>
<p class="ltr"><code>python scripts/render_v2_comparisons.py</code></p><p class="ltr"><code>Checkpoint SHA256: CHECKPOINT_SHA</code></p></details></div>
</main><script>
const data=CASE_DATA;let mode='fields';const byId=id=>document.getElementById(id);
function choices(selected){const group=byId('group').value;byId('case').replaceChildren();data.filter(r=>group==='all'||r.set===group).forEach(r=>{const o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · Re '+r.reynolds;byId('case').append(o)});if(selected)byId('case').value=selected;render()}
function render(){const r=data.find(r=>r.id===byId('case').value);byId('plot').src=r[mode];byId('condition').textContent=r.id+' | Re = '+r.reynolds+' | a = ['+r.coefficients.map(v=>v.toFixed(4)).join(', ')+']';byId('stats').innerHTML=[['خطای بردار سرعت',r.velocity_pct.toFixed(3)+'%'],['خطای فشار',r.pressure_pct.toFixed(3)+'%'],['RMSE مؤلفه v',r.v_rmse.toFixed(5)+' U']].map(([label,val])=>'<div class="stat">'+label+'<br><strong class="ltr">'+val+'</strong></div>').join('');byId('physics').textContent='باقی‌ماندهٔ مومنتوم: '+r.momentum_rms.toFixed(5)+'؛ آستانه: 0.05؛ '+(r.momentum_rms<.05?'پاس شده':'پاس نشده');['fields','profiles'].forEach(k=>byId(k).classList.toggle('active',mode===k))}
byId('group').onchange=()=>choices();byId('case').onchange=render;['fields','profiles'].forEach(k=>byId(k).onclick=()=>{mode=k;render()});
data.forEach(r=>{const tr=document.createElement('tr');[r.id,r.reynolds,r.velocity_pct.toFixed(3),r.u_pct.toFixed(3),r.v_pct.toFixed(3),r.v_rmse.toFixed(5),r.pressure_pct.toFixed(3),r.momentum_rms.toFixed(5)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});tr.onclick=()=>{byId('group').value=r.set;choices(r.id);byId('viewer').scrollIntoView({behavior:'smooth'})};byId('table').append(tr)});choices('B04');
</script></html>'''
    html = html.replace('OVERVIEW_URI', overview_uri).replace('CHECKPOINT_SHA', sha).replace('CASE_DATA', json.dumps(cases))
    (output/'dashboard.html').write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/v2/comparisons')
    args = parser.parse_args()
    output = args.output
    (output/'cases').mkdir(parents=True, exist_ok=True)
    style(); torch.set_num_threads(4)
    checkpoint = ROOT/'models/v2/stream_operator.pt'
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = json.loads((ROOT/'models/v2/metadata.json').read_text())
    assert sha == metadata['checkpoint_sha256'], 'Unexpected trained model'
    model, state = load(checkpoint); model.double()
    with np.load(ROOT/'data/v2/train.npz') as d:
        training_profiles = d['profiles'].copy().astype(np.float32)
    rows, arrays = [], {}
    pdf_path = output/'comparison_atlas.pdf'
    with PdfPages(pdf_path) as pdf, (output/'pointwise_fields.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'set', 'Re', 'x', 'y', 'u_FEM', 'v_FEM', 'p_FEM',
                         'u_model', 'v_model', 'p_model', 'u_error', 'v_error', 'p_error'])
        pages = []
        for group, prefix in [('blind', 'B'), ('monitor', 'M')]:
            with np.load(ROOT/f'data/v2/{group}.npz') as d:
                data = {k: d[k].copy() for k in d.files}
            existing = json.loads((ROOT/f'reports/v2/{group}/metrics.json').read_text())
            assert existing['step'] == state['step'] == 70000
            assert not np.any(np.all(training_profiles[:, None] == data['profiles'][None].astype(np.float32), axis=-1))
            xy = data['xy']; x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
            gx, gy = np.meshgrid(x, y, indexing='ij')
            assert np.allclose(xy, np.stack([gx, gy], -1).reshape(-1, 2))
            predictions = []
            for i in range(len(data['profiles'])):
                t = lambda v: torch.as_tensor(v, dtype=torch.float64)
                with torch.no_grad():
                    pred = model(t(data['profiles'][i:i+1]), t(data['reynolds'][i:i+1]), t(xy[None])).numpy()[0]
                ref = data['fields'][i]; err = pred-ref
                assert np.isfinite(pred).all()
                rel = lambda sl: float(np.linalg.norm(err[:, sl])/max(np.linalg.norm(ref[:, sl]), 1e-15))
                velocity, pressure = rel(slice(0, 2)), rel(2)
                old = existing['cases'][i]
                assert abs(velocity-old['velocity_relative_l2']) < 1e-10
                assert abs(pressure-old['pressure_relative_l2']) < 1e-10
                row = dict(id=f'{prefix}{i+1:02d}', set=group, source_case=i,
                           reynolds=float(data['reynolds'][i]), coefficients=data['coefficients'][i].tolist(),
                           velocity_pct=100*velocity, u_pct=100*rel(0), v_pct=100*rel(1),
                           v_rmse=float(np.sqrt(np.mean(err[:, 1]**2))), pressure_pct=100*pressure,
                           momentum_rms=old['momentum_rms'], flux_max_error=old['flux_max_error'])
                rows.append(row); predictions.append(pred)
                for point, truth, estimate, delta in zip(xy, ref, pred, err):
                    writer.writerow([row['id'], group, row['reynolds'], *point, *truth, *estimate, *delta])
                rgrid, pgrid = [v.reshape(len(x), len(y), 3) for v in [ref, pred]]
                pages.append((row, x, y, rgrid, pgrid))
                print(f"{row['id']} Re={row['reynolds']:g} velocity={row['velocity_pct']:.3f}% pressure={row['pressure_pct']:.3f}%", flush=True)
            for key in ['xy', 'fields', 'coefficients', 'reynolds', 'profiles']:
                arrays[f'{group}_{key}'] = data[key]
            arrays[f'{group}_prediction'] = np.array(predictions)
        fig = overview(rows)
        fig.savefig(output/'overview.png', dpi=160)
        fig.savefig(output/'overview.svg')
        pdf.savefig(fig); plt.close(fig)
        for row, x, y, ref, pred in pages:
            for kind, plotter in [('fields', field_figure), ('profiles', profile_figure)]:
                fig = plotter(row, x, y, ref, pred)
                fig.savefig(output/'cases'/f"{row['id']}_{kind}.png", dpi=140)
                pdf.savefig(fig); plt.close(fig)
            print(f"Rendered {row['id']} fields and cross-sections", flush=True)
        pdf.infodict().update(Title='Channel operator vs FEM: all 24 held-out cases',
                              Subject='Fixed 70k checkpoint; native sample grid; no retraining')
    np.savez_compressed(output/'all_fields.npz', **arrays)
    keys = ['id', 'set', 'source_case', 'reynolds', 'velocity_pct', 'u_pct', 'v_pct',
            'v_rmse', 'pressure_pct', 'momentum_rms', 'flux_max_error']
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys+['a1', 'a2', 'a3', 'a4'])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in keys}, **dict(zip(['a1','a2','a3','a4'], r['coefficients']))})
    audit = dict(checkpoint_sha256=sha, step=state['step'], cases=24, field_figures=24,
                 profile_figures=24, pdf_pages=49, pointwise_rows=sum(len(arrays[f'{g}_fields'])*len(arrays[f'{g}_xy']) for g in ['blind','monitor']),
                 training_performed=False, heldout_profiles_disjoint=True,
                 field_metrics_match_existing_reports=True, pressure_offset_fitted=False,
                 reference_interpolated=False, sampling_grid=[49,25], cases_summary=rows)
    (output/'manifest.json').write_text(json.dumps(audit, indent=2))
    make_dashboard(output, rows, sha)
    md = ['# مقایسهٔ تصویری مدل ثابت با حل عددی مرجع', '',
          'تمام **۲۴ حالت خارج از آموزش** نمایش داده شده‌اند: ۸ آزمون نهایی B و ۱۶ حالت پایش M. '
          'حالت‌های M در توسعه و انتخاب مدل بررسی شده‌اند؛ با آزمون نهایی B یکسان نیستند. '
          'مدل هیبریدی FEM + physics-informed DeepONet است و وزن‌های ۷۰هزارگامی در این گزارش تغییر نکرده‌اند.', '',
          '[گزارش PDF، ۴۹ صفحه](comparison_atlas.pdf) · [داشبورد تعاملی آفلاین](dashboard.html) · '
          '[جدول خطاها](summary.csv) · [مقادیر نقطه‌ای مرجع، مدل و اختلاف](pointwise_fields.csv) · '
          '[تمام میدان‌ها در NPZ](all_fields.npz)', '',
          'برای داشبورد، فایل HTML را دانلود و در مرورگر باز کنید؛ تصاویر در خود فایل قرار دارند. '
          'GitHub آن را مستقیماً اجرا نمی‌کند.', '',
          '![نمای کلی پروفایل‌ها و خطاها](overview.png)', '',
          '## روش مقایسه', '',
          '- مرجع: حل عددی اجزای محدود Taylor–Hood؛ مقایسه روی نقاط اصلی ذخیره‌شدهٔ ۴۹×۲۵، نه اندازهٔ مش FEM.',
          '- میدان‌های u، v، p و اندازهٔ سرعت برای هر حالت: مرجع، مدل و اختلاف علامت‌دار، با مقیاس یکسان مرجع و مدل.',
          '- مقاطع u، v و p در چهار موقعیت کانال؛ موقعیت واقعی نزدیک‌ترین نمونه روی نمودار نوشته شده است.',
          '- خطای نسبی L2 از نسبت نرم اختلاف به نرم مرجع روی همین نقاط محاسبه شده است؛ خطای سرعت برای بردار (u,v) است.',
          '- هیچ جابه‌جایی دلخواه فشار، هموارسازی مرجع یا آموزش مجدد برای ورودی آزمون انجام نشده است.',
          '- به علت کوچک‌بودن مؤلفهٔ v، خطای نسبی آن را همراه RMSE مطلق ببینید.', '',
          'مقاطع فشار می‌توانند اختلاف‌ها و نوسان‌های موضعی را آشکار کنند که در خطای کلی L2 کوچک به نظر می‌رسند؛ برای مثال B04 را ببینید.', '',
          '**محدودیت:** معیار سخت‌گیرانهٔ مومنتوم هنوز در بعضی حالت‌ها پاس نشده است. '
          'مقادیر مومنتوم از ارزیابی ثبت‌شدهٔ همین checkpoint نقل شده‌اند. '
          'این نتایج محدود به خانوادهٔ چهارمودی، Re بین ۱۰ و ۱۰۰ و هندسهٔ کانال ۵×۱ هستند.', '',
          '## تمام حالت‌ها', '',
          '| حالت | Re | خطای بردار سرعت ٪ | خطای u ٪ | خطای v ٪ | RMSE مؤلفه v | خطای فشار ٪ | مومنتوم RMS | نمودار |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        md.append(f"| {r['id']} | {r['reynolds']:g} | {r['velocity_pct']:.3f} | {r['u_pct']:.3f} | {r['v_pct']:.3f} | {r['v_rmse']:.5f} | {r['pressure_pct']:.3f} | {r['momentum_rms']:.5f} | [میدان](cases/{r['id']}_fields.png) · [مقاطع](cases/{r['id']}_profiles.png) |")
    md += ['', '## تصاویر هر هشت آزمون نهایی', '']
    for r in rows[:8]:
        md += [f"### {r['id']} — Re={r['reynolds']:g}", '',
               f"ضرایب ورودی: `{', '.join(f'{a:+.4f}' for a in r['coefficients'])}`", '',
               f"![{r['id']} مرجع، مدل و اختلاف](cases/{r['id']}_fields.png)", '',
               f"[مقایسهٔ مقاطع این حالت](cases/{r['id']}_profiles.png)", '']
    md += ['## بازتولید', '', '```bash', 'pip install -r requirements.txt',
           'python scripts/render_v2_comparisons.py', '```', '',
           f'هش وزن ثابت: `{sha}`. تطابق خطاهای میدان با گزارش قبلی برای هر ۲۴ حالت کنترل شده است.']
    (output/'README.md').write_text('\n'.join(md)+'\n')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
    print(f'Wrote 24 comparisons, 48 case images, 49-page PDF and offline dashboard to {output}', flush=True)


if __name__ == '__main__':
    main()
