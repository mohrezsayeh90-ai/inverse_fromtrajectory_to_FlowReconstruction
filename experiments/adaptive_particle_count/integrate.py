"""Add completed new results to the thesis; leave all prior results in place."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'New_Adaptive_Particle_Results/continuation'
names=['beta5','skew_b3','flat30','twin_b3_r30','shoulder','mixed','waves3_r30','lobes4_r30','nine30']
labels=['بتای باریک','نامتقارن','تقریباً تخت','دوقله‌ای','شانه‌دار','چندمقیاسی','سه‌نوسانی','چهارنوسانی','نه‌نوسانی']
records=[]
for name in names:
 d=json.loads((OUT/'progress'/f'{name}.json').read_text());assert d['phase']=='completed';assert (OUT/'figures'/f'{name}.pdf').exists();records.append(d)
text=r'''\par\medskip
\noindent\textbf{افزایش تطبیقی ذرات با رهاسازی‌های متنوع.}\par\smallskip
مطالعه‌ای مستقل برای نه پروفایل ورودی انجام شد.
نتایج این مطالعه به مقایسهٔ پیشین افزوده شده‌اند.
مکان، تندی و جهت رهاسازی ذرات افزوده متنوع بودند.
بخشی از شرایط رهاسازی با بذر تصادفی ثابت انتخاب شدند.
سرعت اولیه و موقعیت اولیهٔ هر ذره معلوم بودند.
مسیرها رهاسازی‌های مستقل در جریان یکسان را نمایندگی می‌کنند.
برهم‌کنش ذرات در این مدل لحاظ نشده است.
مسیرهای مرجع از میدان اجزای محدود ذخیره‌شده تولید شدند.
در هر مرحله، مسیرهای مرحلهٔ قبل حفظ شدند.
نویز هر مسیر نیز میان مراحل یکسان ماند.

تعداد ذرات با توجه به خطای پروفایل افزایش یافت.
سپس حدود شش مرحلهٔ میانی واقعاً برازش شدند.
منحنی‌های میانی، درون‌یابی پاسخ‌های ابتدا و انتها نیستند.
حد ارزیابی خطای پروفایل $5\%$ و سقف تعداد ذرات $24$ بود.
توقف ناشی از سکون خطا نیز جداگانه ثبت شد.
حقیقت ورودی فقط در تولید داده و ارزیابی استفاده شد.
انتخاب تعداد بر پایهٔ حقیقت، مخصوص همین آزمایش مصنوعی است.
این قاعده، بدون حقیقت معلوم، معیار توقف دادهٔ واقعی نیست.
وزن‌های شبکهٔ مستقیم در تمام برازش‌ها ثابت ماندند.
هر شروع بهینه‌سازی حداکثر $200$ تکرار داشت.
دو شروع در هر برازش بررسی شدند.
در برازش‌های تازه، پاسخ مرحلهٔ قبلی نیز نقطهٔ شروع بود.
همگرایی بهینه‌ساز با رسیدن به خطای کوچک یکسان نیست.

'''
for name,label,d in zip(names,labels,records):
 n=d['final_count'];r=next(v for v in d['evaluated'] if v['N']==n);err=100*r['error']
 text+=f'شکل~\\ref{{fig:adaptive-count-{name}}} روند بازسازی ورودی {label} را نشان می‌دهد.\n'
 text+=f'خطای پروفایل در مرحلهٔ ${n}$ذره‌ای برابر ${err:.2f}\\%$ بود.\n'
 if err>5:text+='این نمونه به حد ارزیابی پنج درصد نرسید.\n'
 text+='\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{New_Adaptive_Particle_Results/continuation/figures/'+name+'.pdf}\n\\caption{روند بازسازی ورودی '+label+' با افزایش تطبیقی تعداد ذرات.}\n\\label{fig:adaptive-count-'+name+'}\n\\end{figure}\n\\FloatBarrier\n'
text+='خطای نهایی به پوشش مسیر، منظم‌سازی و بودجهٔ برازش وابسته است.\nاز این پیمایش، کمینهٔ قطعی تعداد ذرات نتیجه نمی‌شود.\nتعداد پایانی جست‌وجو، الزاماً کمترین تعداد موفقِ ارزیابی‌شده نیست.\n'

near=OUT.parent/'near_inlet'
for path in sorted((near/'progress').glob('*.json')):
 d=json.loads(path.read_text())
 if d['phase']!='completed':continue
 name=d['profile'];n=d['final_count'];r=next(v for v in d['evaluated'] if v['N']==n)
 text+=r'''\FloatBarrier
\par\medskip
\noindent\textbf{طراحی هدفمند برای نمونهٔ دشوار.}\par\smallskip
ذرات افزوده نزدیک ورودی و در ارتفاع‌های مختلف رها شدند.
سرعت‌های اولیه کوچک‌تر و زمان‌های آغاز حرکت متراکم‌تر بودند.
ضریب جریمهٔ زبری به $10^{-8}$ کاهش یافت.
جریمهٔ باقیماندهٔ شبکه در این آزمون اعمال نشد.
وزن‌های شبکه ثابت ماندند و بازپخش مستقل همچنان انجام شد.
سقف تکرار هر شروع در این آزمون $400$ بود.
این آزمون چند عامل را هم‌زمان تغییر می‌دهد.
اثر جداگانهٔ این عوامل از این مقایسه نتیجه نمی‌شود.
تنظیمات این آزمون، انتخاب اکتشافیِ همین مطالعهٔ مصنوعی هستند.
پاسخ پایه فقط در تعدادهای مشترک برای مقایسه رسم شد.
'''
 text+=f'شکل~\\ref{{fig:adaptive-near-{name}}} مراحل این طراحی تکمیلی را نشان می‌دهد.\n'
 text+=f'خطای پروفایل در مرحلهٔ ${n}$ذره‌ای برابر ${100*r["error"]:.2f}\\%$ بود.\n'
 text+='\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{New_Adaptive_Particle_Results/near_inlet/figures/'+name+'.pdf}\n\\caption{روند بازسازی نمونهٔ دشوار با رهاسازی هدفمند نزدیک ورودی.}\n\\label{fig:adaptive-near-'+name+'}\n\\end{figure}\n\\FloatBarrier\n'

(ROOT/'chapters/adaptive-particle-results.tex').write_text(text)
p=ROOT/'chapters/results.tex';s=p.read_text();marker='\\input{chapters/adaptive-particle-results}'
if marker not in s:
 anchor='\\زیرقسمت{بازسازی جریان از هندسهٔ مسیرها بدون زمان اندازه‌گیری‌شده}'
 s=s.replace(anchor,marker+'\n\\FloatBarrier\n'+anchor);p.write_text(s)
print('Added nine completed figures; previous thesis results retained.')
