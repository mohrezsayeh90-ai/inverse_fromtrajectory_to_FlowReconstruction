"""Write the Persian delivery report using the final saved results and TeX labels."""
import csv
import json
import re

import fitz

from run import BASE, ROOT, REVISION, SHA

primary = BASE/'soap_314k'
focused = BASE/'soap_314k_near'


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


summary = rows(primary/'summary.csv')
paired = rows(primary/'same_count_comparison.csv')
forward = rows(primary/'forward_comparison.csv')
old = {r['profile']:r for r in rows(BASE/'continuation/summary.csv')}
audit = json.loads((primary/'audit.json').read_text())
assert len(summary)==9 and audit['passed']
labels = dict(re.findall(r'\\newlabel\{(fig:soap-particle-[^}]+)\}\{\{([^}]+)\}',
                       (ROOT/'build/final/chapters/results.aux').read_text()))
with fitz.open(ROOT/'thesis.pdf') as tex:
    pages = len(tex)

match = [r for r in paired if r['old_complete']=='True' and r['soap_complete']=='True']
improved = sum(float(r['soap_percent'])<float(r['old_percent']) for r in match)
forward_improved = sum(float(r['soap_velocity_l2_percent'])<float(r['old_velocity_l2_percent']) for r in forward)
successful = sum(bool(r['first_tested_below_5_N']) for r in summary)
lines = [
    '# بازسازی جریان با وزن جدید SOAP',
    '',
    f'این آزمایش از وزن ثابت گام ۳۱۴٬۰۰۰، شاخهٔ `results/soap-cosine-200k`، نسخهٔ `{REVISION}` استفاده کرد.',
    f'هش SHA-256 وزن `{SHA}` است.',
    'گزارش و وزن مدل در [شاخهٔ نتایج GitHub](https://github.com/mohrezsayeh90-ai/inverse_fromtrajectory_to_FlowReconstruction/tree/f6b8690059716e809c2dac27c8715e65fcb1d267/reports/experiments/soap_cosine_200k) قرار دارند.',
    '',
    'پیمایش از دو ذره آغاز شد. متناسب با خطا، تعداد با ضریب تقریبی ۱٫۵ یا ۱٫۸ افزایش یافت.',
    'در پیمایش پایه، پایان بازهٔ ۲۴ ذره، سکون خطا یا رسیدن به ۵٪ همراه با بازپخش کامل، توقف را تعیین کرد.',
    'برای نمایش پیوسته، پنج یا شش تعداد مناسب انتخاب و هر کدام واقعاً برازش شدند.',
    'تعداد توقف جست‌وجو و نخستین تعداد موفقِ مشاهده‌شده، جدا ثبت شده‌اند.',
    'گزینش تعداد از حقیقت ورودی در این آزمایش مصنوعی استفاده می‌کند. خودِ برازش فقط از مسیرها و شرایط اولیه استفاده می‌کند.',
    '',
    'همان مجموعهٔ تودرتوی رهاسازی و همان نویز موقعیت، بین دو بررسی وزن مشترک بود.',
    'مجموعهٔ هر پروفایل شامل دو رهاسازی پایه، ۱۶ رهاسازی تصادفی لایه‌بندی‌شده و شش رهاسازی با زاویهٔ تصادفی است.',
    'بازهٔ تندی اولیه تقریباً ۰٫۷۷ تا ۶٫۱۲ در مقیاس سرعت مرجع بود.',
    'مسیرها شامل شروع به سمت بالادست، پایین‌دست و هر دو جهت عرضی بودند.',
    'آزمون پایه همان منظم‌سازی، دو شروع و سقف ۲۰۰ تکرار برای هر شروع را دارد.',
    'شروع ادامه‌دار هر مرحله از برازش همان وزن با ذرات کمتر انتخاب شد.',
    '',
    f'در {successful} پروفایل، دست‌کم یک تعداد آزموده‌شده به ۵٪ همراه با بازپخش کامل رسید.',
    f'در {improved} مورد از {len(match)} مقایسهٔ هم‌تعداد با بازپخش کامل، خطای معکوس با وزن جدید کمتر بود.',
    f'در مسئلهٔ مستقیم، خطای نسبی بردار سرعت روی ۷٬۰۸۱ نقطهٔ FEM در {forward_improved} نمونه از ۹ نمونه کمتر شد.',
    'خطای فشار و خطای بازیابی ورودی، معیارهای مستقل‌اند و در CSVهای مربوط گزارش شده‌اند.',
    'گام ۳۱۴٬۰۰۰ در کار مرجع هنوز دو معیار فشار و باقیماندهٔ مومنتوم را برآورده نکرده بود.',
    '',
    '| ورودی | شمارهٔ شکل | ذرات نمایش | توقف جدید | خطای پایانی جدید (%) | نخستین تعداد موفق آزموده‌شده | توقف / خطای پایان قبلی (%) |',
    '| --- | --- | --- | ---: | ---: | ---: | --- |',
]
for s in summary:
    name=s['profile']
    previous=old[name]
    lines.append(f"| {name} | {labels['fig:soap-particle-'+name]} | {s['display_N']} | {s['search_stop_N']} | "
                 f"{float(s['final_error_percent']):.2f} در {s['final_N']} | {s['first_tested_below_5_N'] or 'نرسید'} | "
                 f"{previous['final_N']} / {float(previous['final_error_percent']):.2f} |")
lines.extend([
    '',
    'پایان دو پیمایش ممکن است در تعداد متفاوتی باشد. برای سنجش اثر وزن، جدول `same_count_comparison.csv` تعدادهای مشترک را گزارش می‌کند.',
    'برای تخصیص علّی اثر وزن، باید حساسیت به مسیر، شروع و بودجهٔ بهینه‌سازی را نیز سنجید.',
    f"ممیزی پایه {audit['fits']} برازش و {audit['replays']} بازپخش ذخیره‌شده را بررسی کرد.",
    'تابع هزینه و گرادیان محاسبه‌شده با تنهٔ ثابتِ ازپیش‌محاسبه‌شده، در دو نقطه با پیاده‌سازی اصلی تطبیق داده شدند.',
])
if (focused/'summary.csv').exists():
    near = rows(focused/'summary.csv')[0]
    prior = rows(BASE/'near_inlet/summary.csv')[0]
    near_audit = json.loads((focused/'audit.json').read_text())
    long_run = BASE/'soap_314k_near_extended'
    lines.extend([
        '',
        f"آزمون نزدیک ورودی در شکل {labels['fig:soap-particle-near']} آمده است.",
        f"خطای انتهای پیمایش وزن جدید در {near['final_N']} ذره {float(near['final_error_percent']):.2f}٪ بود.",
        f"خطای انتهای پیمایش وزن قبلی در {prior['final_N']} ذره {float(prior['final_error_percent']):.2f}٪ بود.",
        f"ممیزی این آزمون {near_audit['fits']} برازش و {near_audit['replays']} بازپخش ذخیره‌شده را بررسی کرد.",
        'طراحی نزدیک ورودی تنظیمات و مسیرهای همان آزمون قبلی را دارد. مقایسه با طراحی پایه، چند عامل را هم‌زمان تغییر می‌دهد.',
    ])
    if (long_run/'summary.csv').exists():
        extension = rows(long_run/'summary.csv')[0]
        ext_audit = json.loads((long_run/'audit.json').read_text())
        lines.extend([
            f"در گسترش همین طراحی، ۲۴ مسیر نخست دقیقاً حفظ شدند. خطا در {extension['final_N']} ذره {float(extension['final_error_percent']):.2f}٪ بود.",
            f"ممیزی پیمایش گسترده {ext_audit['fits']} برازش و {ext_audit['replays']} بازپخش را بررسی کرد.",
            'مدل قبلی در تعدادهای بزرگ‌تر از ۲۴ با این طراحی اجرا نشده است. مقایسهٔ وزن‌ها در همان ۲۴ مسیر مشترک انجام می‌شود.',
        ])
lines.extend([
    '',
    f'پایان‌نامه پس از افزودن این نتایج با XeLaTeX و BibTeX ساخته شد و {pages} صفحه دارد.',
    'نسخهٔ ۱۱۲صفحه‌ای پیش از افزودن این نتایج در `soap_314k/prior_delivery/thesis.pdf` محفوظ است.',
    '',
    '## فایل‌های بازتولید',
    '',
    '- `experiments/soap_particle_count/run.py` برای برازش و بازپخش.',
    '- `experiments/soap_particle_count/cached_operator.py` برای محاسبهٔ دقیق و سریع تابع هدف.',
    '- `experiments/soap_particle_count/check_forward.py` برای مقایسهٔ مستقیم روی مرجع مشترک.',
    '- `experiments/soap_particle_count/extend_near.py` برای ساخت ۲۴ مسیر اضافهٔ هدفمند.',
    '- `experiments/soap_particle_count/deliver.py` برای ممیزی، ساخت شکل‌ها و افزودن به پایان‌نامه.',
    '- داده‌ها، گزارش هر مرحله و هش فایل‌ها در `New_Adaptive_Particle_Results/soap_314k/` ثبت شده‌اند.',
])
(ROOT/'docs/SOAP_ADAPTIVE_PARTICLE_COUNT_FA.md').write_text('\n'.join(lines)+'\n')
