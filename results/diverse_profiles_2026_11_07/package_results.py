"""Document, check and package the new figures; never retrain or overwrite old work."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import fitz

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LABELS={'skew':'نامتقارن','twin':'دو‌قله‌ای','waves':'موج‌دار','lobes':'سه‌قله‌ای','mixed':'نامتقارن چندمقیاسی','shoulder':'برآمدگی خارج از مرکز','nine':'مدولاسیون ۹ دوره‌ای'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m=json.loads((HERE/'metrics.json').read_text());s=json.loads((HERE/'sweep.json').read_text())
    chosen={r['name'] for r in m['selected_forward_cases']}
    text=['# پروفایل‌های متنوع‌تر و افزودن ورودی تخمینی به بازسازی از ذرات','',
      '## Beta(5,5) یعنی چه؟','',
      'این نام دو پارامتر شکل **۵ و ۵** را نشان می‌دهد، نه عدد اعشاری ۵٫۵. تابع بتای نرمال‌شده روی ارتفاع بی‌بعد کانال `0 ≤ y ≤ 1` به صورت زیر است:','',
      '`f(y) = y^(a-1) (1-y)^(b-1) / B(a,b)`','',
      'بنابراین `Beta(5,5) = 630 y^4 (1-y)^4` است: پروفایلی صاف، متقارن و زنگوله‌ای که روی دو دیواره صفر و میانگین آن یک است. این پروفایل سهمی نیست؛ سهمی نرمال‌شده `6y(1-y)` یا Beta(2,2) است. بزرگ‌شدن دو پارامتر مساوی، پروفایل را متمرکزتر می‌کند. در شکل‌های جدید نام توصیفی شکل‌ها به‌کار رفته تا خواندن آن‌ها ساده‌تر باشد.','',
      '## بخش اول: ورودی‌های واقعاً متفاوت با خطای اندازه‌گیری‌شده','',
      'همان وزن‌های ثابت مدل **PI-CFO** در گام ۹۴۰۰۰ استفاده شده‌اند؛ آموزش مجدد یا اصلاح وزن‌ها انجام نشده است. نام PI-CFO به اپراتور جریان کانالِ اصلاح‌شده در این پروژه اشاره دارد. پایهٔ معماری، DeepONet فیزیک‌آگاه است و نسبت‌دادن ایدهٔ اصلی DeepONet به این پروژه منظور نیست.','',
      'هر تصویر پنج پنل در یک ردیف دارد: **پروفایل ورودی → پیش‌بینی شبکه → FEM مستقل → نقشهٔ خطای برداری سرعت → مقایسهٔ برش‌های جریان**. رنگ‌بندی شبکه و FEM یکسان است. برای خطای نقطه‌ای از اندازهٔ بردار اختلاف سرعت استفاده شده، نه تفاضل اندازه‌های سرعت که ممکن است اختلاف جهت را پنهان کند.','',
      '[گالری پروفایل‌های منتخب](profile_gallery.png) · [PDF همهٔ شکل‌های جدید](all_new_figures.pdf) · [بستهٔ کامل](diverse_results_package.zip)','',
      '![گالری شکل‌های ورودی](profile_gallery.png)','',
      '| نوع ورودی | Re | خطای سرعت برداری | خطای فشار | قلهٔ برجسته |','|---|---:|---:|---:|---:|']
    for r in m['selected_forward_cases']:
        text.append(f"| {LABELS[r['group']]} | {r['reynolds']:g} | {100*r['velocity_relative_l2']:.3f}٪ | {100*r['pressure_relative_l2']:.3f}٪ | {r['prominent_peaks']} |")
    text+=['',
      'انتخاب نمونه‌ها توسعه‌ای و پس از محاسبهٔ خطاست. شرط ورود به مجموعه، خطای نسبی L2 سرعت برداری کمتر از ۵٪ و فشار کمتر از ۱۰٪، همراه با بررسی تغییر مش مرجع است. از هر خانواده یک نمونه انتخاب شده است؛ معیار ترجیح، کمینه‌کردن بزرگ‌ترین خطای نرمال‌شده به این دو آستانه است. تمام تلاش‌ها، از جمله حالت‌های انتخاب‌نشده، در جدول پایین و `sweep.json` نگه داشته شده‌اند. این مثال‌های منتخب، تضمین دقت برای هر ورودی یا قبولی تمام معیارهای فیزیکی مدل نیستند.','',
      '«مدولاسیون ۹ دوره‌ای» یعنی ضریب کسینوسی `cos(18*pi*y)` در کل ارتفاع ۹ دوره دارد؛ پس از ضرب در پوشِ صفرشونده روی دیواره، همهٔ آن‌ها الزاماً قلهٔ برجستهٔ جداگانه نیستند. تعداد قله‌های برجستهٔ پروفایل نهایی نیز جداگانه در جدول گزارش شده است.','',
      'مرجع، FEM مستقل Taylor–Hood با سرعت P2 و فشار P1 برای همان مسئلهٔ مرزی ناویر–استوکس پایا است. هیچ خروجی شبکه‌ای در تولید مرجع استفاده نشده است. مش در جهت طولی نزدیک ورودی ریز شده؛ موارد قابل انتخاب روی دو مش محاسبه شده‌اند و تغییر مرجع باید برای سرعت کمتر از ۱٪ و فشار کمتر از ۲٪ باشد. خطای گزارش‌شده روی ۷۰۸۱ نقطه با تراکم بیشتر نزدیک ورودی محاسبه می‌شود؛ این نرم نمونه‌ای است و انتگرال وزن‌دار مساحت نیست. فشار بدون جابه‌جایی برای کاهش خطا مقایسه شده است. اعداد روی تصویر مربوط به کل این شبکهٔ ارزیابی هستند، نه فقط برش‌های پنل آخر.']
    for r in m['selected_forward_cases']:
        name=r['name'];text+=['',f"### {LABELS[r['group']]} — Re={r['reynolds']:g}",'',
            f"فرمول: `normalize(y^{r['power']}*(1-y)^{r['power']}*({r['modulation']}))`؛ normalize یعنی میانگین ارتفاعی یک. ورودی، اسپلاین C2 از ۲۵۷ نمونهٔ این تابع است.",'',
            f'![پروفایل تا خطای میدان](forward/{name}.png)','',f'[PNG اصلی](forward/{name}.png) · [PDF برداری](forward/{name}.pdf)']
    text+=['','## بخش دوم: پروفایل ورودیِ تخمین‌زده‌شده هم دیده می‌شود','',
      'هر ردیف جدید شش پنل دارد: **دو مسیر ورودی → پروفایل ورودی تخمینی → میدان سرعت تخمینی → حرکت ذرات در همان میدان → هم‌پوشانی مسیرها → خطای مکانی در زمان یکسان**. علاوه بر آن، برای هر مثال یک تصویر مستقل از ورودی تخمینی آماده شده است.','',
      'پروفایل پنل دوم مستقیماً از مقادیر ذخیره‌شدهٔ `inlet_normalized.csv` و سرعت میانگین تخمینی خوانده می‌شود؛ این شکل، پروفایل حقیقی پنهانیِ سازندهٔ داده‌ها نیست. هیچ برازش مجددی انجام نشده و مسیر خروجی همان حل مستقل ODE ذرات در جریان تخمینی ذخیره‌شده است. نزدیک‌بودن مسیرها، دقت یا یکتایی بازسازی کل میدان را ثابت نمی‌کند؛ دو مسیر، اطلاعات محدودی از کل میدان می‌دهند. معیار این بخش، بازتولید مسیرهای استفاده‌شده در برازش است و آزمون مسیر کاملاً ندیده محسوب نمی‌شود.']
    for r in m['inverse_rows']:
        name=r['name'];text+=['',f"### {r['label']}",'',f'![ردیف شامل ورودی تخمینی](inverse/{name}.png)','',
          '؛ '.join(f"ذرهٔ {p['particle']}: RMSE دوبعدی {p['position_2d_rmse_percent_H']:.4f}٪ ارتفاع کانال" for p in r['particles']),
          '',f'[ردیف شش‌پنلی PNG](inverse/{name}.png) · [PDF برداری](inverse/{name}.pdf) · [تصویر مستقل پروفایل ورودی](inverse/{name}_inlet.png)']
    text+=['','## تمام حالت‌های بررسی‌شده','',
      '| نام پرونده | Re | سرعت L2 | فشار L2 | تغییر مش سرعت | تغییر مش فشار | در شکل‌های منتخب |','|---|---:|---:|---:|---:|---:|---|']
    for r in s['cases']:
        ch=r['mesh_change'];cv=f"{100*ch['velocity_relative_l2']:.4f}٪" if ch else 'بررسی نشده';cp=f"{100*ch['pressure_relative_l2']:.4f}٪" if ch else 'بررسی نشده'
        text.append(f"| {r['name']} | {r['reynolds']:g} | {100*r['velocity_relative_l2']:.3f}٪ | {100*r['pressure_relative_l2']:.3f}٪ | {cv} | {cp} | {'بله' if r['name'] in chosen else 'خیر'} |")
    text+=['','## کیفیت خروجی و بازتولید','',
      'PNGها با وضوح ۳۲۰dpi ذخیره شده‌اند. ردیف‌های مستقیم ۸۶۴۰×۱۶۹۶ و ردیف‌های معکوس ۱۰۳۶۸×۱۶۹۶ پیکسل‌اند. برای اسلاید و بزرگ‌نمایی، PDFها برداری هستند. نسبت نمایش محورها برای قرارگرفتن کنار هم تنظیم شده است؛ مقادیر x/H و y/H ملاک هندسه‌اند. کانال واقعی طول ۵ و ارتفاع ۱ دارد.','',
      'فایل‌های مدل، حل‌گر، اسلایدها و نتایج قبلی تغییر نکرده‌اند. فایل‌های خام جدید، اطلاعات دو مش، معیارها، هش مدل و ورودی‌ها، و داده‌های لازم برای بازتولید شکل‌های ذرات در این پوشه موجودند. این بسته یک نسخهٔ جدید از مدل آموزش‌دیده نیست؛ شکل‌ها و آزمایش‌های جدید با همان مدل‌اند.','',
      'از ریشهٔ مخزن، در محیط پایتون با وابستگی‌های پروژه:','',
      '```bash','python results/diverse_profiles_2026_11_07/evaluate_profiles.py',
      'python results/diverse_profiles_2026_11_07/build_figures.py','python results/diverse_profiles_2026_11_07/package_results.py','```','',
      'تولیدکنندهٔ FEM نتایج موجود را از پرونده‌ها می‌خواند. برای محاسبهٔ کامل مجدد، پوشهٔ data را ابتدا به محل دیگری منتقل کنید و سپس evaluate_profiles.py را اجرا کنید. بستهٔ ZIP شکل‌ها، جدول‌ها، کد و آرایه‌های عددی را دربرمی‌گیرد؛ وزن‌های مدل در مسیر اصلی مخزن نگهداری می‌شوند.']
    (HERE/'README.md').write_text('\n'.join(text)+'\n')
    cp=ROOT/'models/candidates/pressure_long/checkpoint.npz';assert sha(cp)=='ccf6fa2b3e179a22c443384083e2dec338fc497f9562a567ed7e0955d3c67bf8'
    for r in m['selected_forward_cases']:
        assert r['velocity_relative_l2']<.05 and r['pressure_relative_l2']<.1 and r['mesh_verified']
    figures=[]
    for p in sorted(HERE.rglob('*.png')):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            dimensions=im.size
        if p.parent.name=='forward':assert dimensions==(8640,1696)
        if p.parent.name=='inverse' and not p.stem.endswith('_inlet'):assert dimensions==(10368,1696)
        figures.append(dict(path=str(p.relative_to(HERE)),dimensions=dimensions,sha256=sha(p)))
    pdf=fitz.open(HERE/'all_new_figures.pdf');assert len(pdf)==len(figures)
    checks=dict(selected_count=len(chosen),all_attempts=len(s['cases']),checkpoint_sha256=sha(cp),figures=figures,pdf_pages=len(pdf),selected_velocity_max_percent=100*max(r['velocity_relative_l2'] for r in m['selected_forward_cases']),selected_pressure_max_percent=100*max(r['pressure_relative_l2'] for r in m['selected_forward_cases']))
    (HERE/'integrity_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix not in ['.zip','.pyc'] and p.name!='SHA256SUMS.json'}
    (HERE/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'diverse_results_package.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and p.suffix not in ['.zip','.pyc']:
                z.write(p,arcname=str(Path(HERE.name)/p.relative_to(HERE)))
    print(json.dumps({k:v for k,v in checks.items() if k!='figures'},indent=2))

if __name__=='__main__':main()
