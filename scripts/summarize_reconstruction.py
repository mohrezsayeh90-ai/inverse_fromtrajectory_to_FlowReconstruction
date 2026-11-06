"""Publish all completed inverse examples, including unfavorable results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'flow_reconstruction'
CASES=[('development/trajectory_only_initial','پرتاب مایل، تطبیق مسیر'),
       ('fem_beta','پرتاب مایل، فیزیک و همواری'),
       ('corner_launch','پرتاب گوشه‌ای، بدون پیش‌شرط‌سازی'),
       ('corner_launch_preconditioned','پرتاب گوشه‌ای، پیش‌شرط‌سازی با تمام گره‌ها')]


def main():
    rows=[]
    for folder,label in CASES:
        root=ROOT/'examples'/folder/'result'
        if not (root/'truth_comparison.json').exists():continue
        report=json.loads((root/'report.json').read_text());truth=json.loads((root/'truth_comparison.json').read_text())
        rows.append(dict(folder=folder,label=label,trajectory_rmse_max=max(t['position_rmse'] for t in report['trajectories']),
                         velocity_error_pct=100*truth['velocity_relative_l2'],pressure_error_pct=100*truth['pressure_relative_l2'],
                         inlet_error_pct=100*truth['inlet_relative_l2'],momentum_rms=report['physical_momentum_rms'],
                         selected_mean_speed=report['mean_speed'],rank=report['data_jacobian_effective_rank'],parameters=report['free_parameters'],
                         seconds=report['seconds_including_compilation'],
                         all_optimizer_starts_converged=all(c['optimizer_success'] for c in report['candidates'])))
    text=['# نتیجهٔ واقعی بازسازی از دو مسیر','','همهٔ مثال‌ها از میدان مستقل FEMِ Beta ساخته شده‌اند. میدان و پروفایل واقعی به objective مسئلهٔ معکوس داده نشده‌اند. داده‌ها مصنوعی و برای توسعه‌اند؛ این‌ها تضمین دقت روی آزمایش واقعی یا همهٔ ورودی‌ها نیستند.','',
          '| مثال | بیشینهٔ RMSE موقعیت / H | خطای کل میدان سرعت ٪ | خطای ورودی ٪ | خطای فشار ٪ | RMS مومنتوم |',
          '|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| [{r['label']}](examples/{r['folder']}/result/report.json) | {r['trajectory_rmse_max']:.6f} | {r['velocity_error_pct']:.2f} | {r['inlet_error_pct']:.2f} | {r['pressure_error_pct']:.2f} | {r['momentum_rms']:.4f} |")
    text+=['','جدول، نتیجهٔ بهترین شروع را مطابق objective همان اجرا نشان می‌دهد. ورودی‌های پرتابِ مثال مایل و گوشه‌ای متفاوت‌اند؛ اختلاف ردیف‌ها آزمایش کنترل‌شدهٔ تک‌عامل نیست. کاهش خطای مسیر به معنی کاهش خطای کل میدان نیست.',
           '', 'در نمونهٔ نخست، تطبیق مسیر بسیار خوب بود ولی خطای کل سرعت بیش از ۲۰٪ ماند. به همین دلیل فیزیک و prior در حل‌کنندهٔ فعلی لحاظ شده‌اند. این اصلاح، تضمین صحت جریان یا رفع خطای اپراتور مبنا نیست.',
           '', 'پیش‌شرط‌سازی یک تبدیل مثبت‌معین و وارون‌پذیر روی تمام پارامترهای گره‌ای است؛ هیچ مود یا گره‌ای حذف نشده است. سپس tanh حدود log-amplitude را اعمال می‌کند. همواری روی خود spline و NS در objective سنجیده می‌شوند. رتبهٔ Jacobian فقط داده، به آستانه و مختصات پارامترها وابسته است.',
           '', 'نتایج ناموفق و شروع‌های ناتمام حذف نشده‌اند. وضعیت `optimizer_success` هر شروع و پیام رسیدن به سقف iteration در JSON محفوظ است. `trajectory_consistency_passed` فقط کنترل مسیر است و پذیرش فیزیک یا بازسازی کل میدان محسوب نمی‌شود.',
           '', 'مدل جریان مبنا همچنان وضعیت پذیرش منفی دارد. دو مرجع رزروشدهٔ اصلی پروژه در این کار استفاده نشده‌اند. پراکندگی چند شروع، عدم‌قطعیت کالیبره‌شده نیست.']
    for r in rows:
        name=r['folder'];text+=['',f"## {r['label']}",'',f"میانگین سرعت تخمینی {r['selected_mean_speed']:.6f}، مقدار واقعی ۱؛ رتبهٔ محلی داده {r['rank']} از {r['parameters']} پارامتر آزاد. زمان اجرا با compile حدود {r['seconds']:.1f} ثانیه؛ همگرایی تمام شروع‌ها: {r['all_optimizer_starts_converged']}.",'',
            f"![مقایسهٔ میدان FEM و تخمین](examples/{name}/result/fem_comparison.png)",'',
            f"![مقایسهٔ ورودی](examples/{name}/result/inlet_comparison.png)",'',
            f"[دو مسیر و تخمین ورودی](examples/{name}/result/trajectories_and_inlet.png) · [مقایسهٔ عددی با حقیقت FEM](examples/{name}/result/truth_comparison.json) · [ورودی قابل اجرا](examples/{name}/problem.json)"]
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n');(ROOT/'examples/summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
