"""Produce the delivery report only after all nine profiles and thesis build."""
from pathlib import Path
import json,csv,re
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'New_Adaptive_Particle_Results';OUT=BASE/'continuation'
rows=list(csv.DictReader((OUT/'summary.csv').open()));assert len(rows)==9
aux=(ROOT/'build/final/chapters/adaptive-particle-results.aux')
# input (rather than include) writes labels into results.aux.
text='\n'.join(p.read_text() for p in (ROOT/'build/final').rglob('*.aux'))
labels={k:(num,page) for k,num,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',text)}
report='''# نتایج جدید افزایش تطبیقی ذرات

این مطالعه به نتایج قبلی افزوده شد؛ شکل‌ها و داده‌های پژوهش قبلی حذف نشدند.
نه شکل جدید برای نه ورودی، با تقریباً پنج یا شش مرحلهٔ واقعی تهیه شدند.
مسیرهای هر مرحله زیرمجموعهٔ تودرتوی مرحلهٔ بعد هستند.
مکان، سرعت و جهت رهاسازی‌های افزوده متنوع و بخشی تصادفی هستند.
شرایط اولیه معلوم‌اند و رهاسازی‌ها مستقل، در جریان حامل ثابت بررسی می‌شوند.

## قرارداد ارزیابی

آستانهٔ ارزیابی ۵٪ خطای نسبی L2 پروفایل در مقیاس U_ref بود.
این حد یک قرارداد آزمایش است؛ برای دادهٔ واقعی بدون حقیقت قابل محاسبه نیست.
شمار ذرات با روند خطا افزایش یافت؛ سقف این پیمایش ۲۴ بود.
پاسخ مرحلهٔ قبلی می‌تواند شروع برازش مرحلهٔ بزرگ‌تر باشد؛ هیچ مرحله‌ای از ذرات آینده مقداردهی نمی‌شود.
هر شروع حداکثر ۲۰۰ تکرار دارد. انتخاب جواب با هزینهٔ داده و منظم‌سازی است.
محل توقف جست‌وجو لزوماً کمترین تعداد موفقِ مشاهده‌شده نیست.
تعداد بیشتر لزوماً خطای کمتر نمی‌دهد؛ همگرایی بهینه‌ساز نیز جداگانه ثبت شده است.
این نتایج به تحقق نویز، چیدمان، اپراتور ثابت، منظم‌سازی و بودجهٔ حاضر محدودند.
ناکامی در سقف تعداد، اثبات ناممکن‌بودن بازسازی نیست.

## شکل‌های افزوده‌شده

| پروفایل | شمارهٔ شکل | تعدادهای نمایش | انتهای جست‌وجو | خطای پایانی (%) | کمترین تعداد موفقِ آزموده‌شده | توقف |
| --- | --- | --- | --- | --- | --- | --- |
'''
for row in rows:
 name=row['profile'];d=json.loads((OUT/'progress'/f'{name}.json').read_text());number,page=labels['fig:adaptive-count-'+name]
 report+=f"| {name} | {number}، صفحهٔ {page} | {', '.join(map(str,d['display_counts']))} | {row['final_N']} | {float(row['final_error_percent']):.3f} | {row['smallest_evaluated_target_N'] or 'نرسید'} | {row['stopping_reason']} |\n"
report+='''
## داده‌ها و کنترل‌ها

- نتایج پایلوت: `New_Adaptive_Particle_Results/fits/`.
- اجرای ادامه‌دار: `New_Adaptive_Particle_Results/continuation/fits/`.
- شرایط رهاسازی، تلاش‌های کنارگذاشته‌شده و مسیرهای مرجع/نویزی: `New_Adaptive_Particle_Results/observations/`.
- نمودارهای PDF و PNG: `New_Adaptive_Particle_Results/continuation/figures/`.
- قرارداد مطالعه و نسخه‌های عددی: `design.json` و `runtime.json` در ریشهٔ نتایج.
- ممیزی نرم خطا، انتخاب شروع، بازپخش‌ها و عدم استفاده از مراحل آینده: `continuation/audit.json`.
- هش ورودی نمودارها: `continuation/figures_manifest.json`.
- تشخیص‌های پایلوت و مقداردهی از تعداد بالاتر، اگر موجود باشند، جدا نگهداری و از مقایسهٔ تعداد کنار گذاشته شدند.
- مخزن منبع FLOW و وزن آموزش‌دیده فقط خوانده شدند.
- نسخهٔ JAX و تحقق نویز با مطالعهٔ قدیمی یکسان فرض نمی‌شوند؛ مقایسهٔ علّی با نتایج قبلی ادعا نمی‌شود.

فایل پایان‌نامه با XeLaTeX و BibTeX بازسازی شد.
گزارش قدیمی `RESULTS_FIGURE_REVISION_FA.md` مربوط به تحویل پیشین است؛ شماره‌صفحه‌های آن تاریخی‌اند.

## بازتولید

از ریشهٔ Main_Proposal:

```sh
bash experiments/adaptive_particle_count/run_all.sh /workspace/revision-sources/flow
/workspace/setup/adaptive-env/bin/python experiments/adaptive_particle_count/integrate.py
. /workspace/setup/activate.sh
latexmk -e '$max_repeat=12;' -xelatex -outdir=build/final -interaction=nonstopmode -halt-on-error -file-line-error thesis.tex
```

توضیح الگوریتم و محدودیت‌ها در `experiments/adaptive_particle_count/README_FA.md` آمده است.
'''

audit=json.loads((OUT/'audit.json').read_text())
report+=f'\nممیزی پایه: {audit["counts"]["fits"]} برازش و {audit["counts"]["replays"]} بازپخش ذخیره‌شده بررسی شدند.\n'
report+='نسخهٔ پیش از افزونه، شامل PDF ۱۰۲صفحه‌ای، در `New_Adaptive_Particle_Results/baseline_before_extension/` حفظ شده است.\n'
report+='تطبیق اجرای واقعی شبکه با آرایهٔ مرجع ذخیره‌شده نیز موفق بود؛ خطای نسبی حدود 8.76e-8 بود.\n'
for path in sorted((BASE/'near_inlet/progress').glob('*.json')):
 d=json.loads(path.read_text())
 if d['phase']!='completed':continue
 name=d['profile'];n=d['final_count'];r=next(v for v in d['evaluated'] if v['N']==n)
 report+='\n## آزمون تکمیلی هدفمند\n\n'
 report+=f'نمونهٔ {name} با رهاسازی‌های نزدیک ورودی، زمان‌های متراکم در آغاز و تنظیمات متفاوت بررسی شد. در {n} ذره، خطای ورودی {100*r["error"]:.3f}٪ بود.\n'
 report+='ضریب زبری 1e-8، وزن جریمهٔ باقیماندهٔ شبکه صفر و سقف هر شروع ۴۰۰ تکرار بود. وزن‌های آموزش‌دیده تغییر نکردند. این مقایسه چندعامل را تغییر می‌دهد و اثر خالص مکان رهاسازی یا تعداد ذرات را جدا نمی‌کند.\n'
 report+='شکل پایه فقط در تعدادهای مشترک به‌صورت مقایسه‌ای نمایش داده شده است؛ برای تعداد اجرا‌نشده منحنی ساخته نشده است.\n'
 label='fig:adaptive-near-'+name
 if label in labels:report+=f'شکل تکمیلی: {labels[label][0]}، صفحهٔ {labels[label][1]}.\n'
 report+='داده‌ها، شکل و ممیزی تکمیلی در `New_Adaptive_Particle_Results/near_inlet/` هستند.\n'
(ROOT/'docs/ADAPTIVE_PARTICLE_COUNT_FA.md').write_text(report)

print('Wrote nine-profile delivery report.')
