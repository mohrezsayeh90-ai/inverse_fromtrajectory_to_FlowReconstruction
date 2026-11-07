"""Make a numerical report, summary charts, vector PDF and reproducible ZIP."""
import csv,hashlib,importlib.metadata,json,platform,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image,ImageDraw
import fitz

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
NAMES={'corners_beta4':'گوشه‌ای B','corners_beta5':'گوشه‌ای A','oblique_beta5':'پرتاب مایل'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    metrics=json.loads((HERE/'comparison_metrics.json').read_text())['cases'];assert len(metrics)==3
    verification=json.loads((HERE/'verification.json').read_text());assert not verification['existing_files_changed']
    text=['# نتایج روش بدون زمان و مقایسه با جواب‌های زمان‌دار قبلی','',
      'سه مثال قبلی با همان وزن‌های ثابت جریان و ورودی‌های فقط `x,y` دوباره برازش شدند. زمان‌های اندازه‌گیری‌شده، مدت واقعی و پارامترهای برازش زمان‌دار وارد بهینه‌سازی جدید نشدند. نمودارها از حل مستقل حرکت ذرات در میدان تخمینی تهیه شده‌اند.','',
      '[توضیح روش و استفاده](README.md) · [PDF ردیف‌های مقایسه](all_comparisons.pdf) · [PDF کامل همراه خلاصه](presentation_figures.pdf) · [بستهٔ کامل](geometry_only_results.zip)','',
      '## مقایسهٔ مسیرها با معیار مشترک هندسی','',
      'همهٔ اعداد این جدول درصد ارتفاع کانال‌اند. «متقارن» برابر RMS فاصلهٔ دوطرفهٔ نقطه تا منحنی است. «مرتب» برابر RMS اختلاف نقاط متناظر در طول قوس نرمال‌شده است و ترتیب حرکت را هم لحاظ می‌کند. این اعداد، خطای موقعیت در زمان واقعی یکسان نیستند.','',
      '| مثال | ذره | متقارن، با زمان | متقارن، بدون زمان | مرتب، با زمان | مرتب، بدون زمان | هاوسدورف بدون زمان |','|---|---:|---:|---:|---:|---:|---:|']
    flat=[]
    for r in metrics:
        for p in r['paths']:
            a,b=p['timed'],p['without_time']
            text.append(f"| {NAMES[r['case']]} | {p['particle']} | {100*a['symmetric_distance_rmse_H']:.4f} | {100*b['symmetric_distance_rmse_H']:.4f} | {100*a['ordered_arclength_rmse_H']:.4f} | {100*b['ordered_arclength_rmse_H']:.4f} | {100*b['symmetric_hausdorff_H']:.4f} |")
            flat.append(dict(case=r['case'],particle=p['particle'],timed_symmetric_rmse_pct_H=100*a['symmetric_distance_rmse_H'],no_time_symmetric_rmse_pct_H=100*b['symmetric_distance_rmse_H'],
                             timed_ordered_rmse_pct_H=100*a['ordered_arclength_rmse_H'],no_time_ordered_rmse_pct_H=100*b['ordered_arclength_rmse_H'],no_time_hausdorff_pct_H=100*b['symmetric_hausdorff_H'],no_time_endpoint_pct_H=100*b['endpoint_error_H']))
    with (HERE/'path_comparison.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    text+=['','## آیا پروفایل‌های ورودی دو روش یکسان شده‌اند؟','',
      'این جدول اختلاف دو تخمین را نشان می‌دهد؛ مرجع آن پروفایل حقیقی نیست. اختلاف شکل پس از تقسیم هر پروفایل بر میانگین خودش محاسبه شده تا تفاوت شکل از تفاوت مقیاس سرعت جدا دیده شود.','',
      '| مثال | اختلاف پروفایل نسبت به روش با زمان | اختلاف شکل نرمال‌شده | سرعت میانگین با زمان | سرعت میانگین بدون زمان |','|---|---:|---:|---:|---:|']
    for r in metrics:text.append(f"| {NAMES[r['case']]} | {100*r['profile_relative_l2_difference']:.3f}٪ | {100*r['normalized_profile_relative_l2_difference']:.3f}٪ | {r['timed_mean_speed']:.5f} | {r['without_time_mean_speed']:.5f} |")
    text+=['',
      'نزدیکی مسیرها و نزدیکی پروفایل‌ها دو نتیجهٔ جدا هستند. حذف اطلاعات زمان و محدودبودن دو مسیر می‌تواند جواب‌های متفاوتِ سازگار با منحنی‌ها ایجاد کند. افزون بر این، بودجه و بعضی تنظیمات بهینه‌سازی دو نسخه یکسان نیست؛ این مقایسه، اندازه‌گیری خالص اثر حذف زمان نیست. هیچ یک از این اختلاف‌ها به‌تنهایی، خطای کل میدان نسبت به واقعیت را مشخص نمی‌کند.','',
      '## شکل‌های آمادهٔ اسلاید','',
      'هر ردیف شش پنل دارد: مسیرهای ورودی؛ پروفایل‌های ورودی دو روش؛ میدان بدون زمان؛ ذرات در همان میدان؛ هم‌پوشانی هر دو پاسخ با داده؛ فاصلهٔ هندسی در طول منحنی. خط پیوسته مربوط به روش بدون زمان و خط‌چین مربوط به روش با زمان است. حلقه‌های توخالی نقاط ورودی‌اند.']
    for r in metrics:
        name=r['case'];text+=['',f"### {NAMES[name]}",'',f'![مقایسه کامل](figures/{name}_comparison_row.png)','',
          f'[PNG ردیف](figures/{name}_comparison_row.png) · [PDF برداری](figures/{name}_comparison_row.pdf) · [مقایسهٔ مستقل پروفایل‌ها](figures/{name}_inlet_comparison.png)']
    text+=['','## زمان‌های تخمینی؛ فقط مقایسهٔ بعد از برازش','',
      'ستون مدت واقعی فقط در برنامهٔ مقایسه خوانده شده است و بهینه‌ساز جدید آن را دریافت نکرده است.','',
      '| مثال | ذره | مدت واقعی، صرفاً برای بررسی بعدی | مدت تخمینی بدون زمان |','|---|---:|---:|---:|']
    for r in metrics:
        for p in r['paths']:text.append(f"| {NAMES[r['case']]} | {p['particle']} | {p['true_duration_for_posthoc_comparison_only']:.6f} | {p['inferred_duration']:.6f} |")
    text+=['','## وضعیت اجرا و حفظ نتایج قبلی','',f"هش محتوای **{verification['existing_files_verified']} فایل قبلی** بررسی شد؛ هیچ‌کدام تغییر نکرده‌اند. وزن شبکه نیز ثابت مانده است. هر شش بازپخش مستقل ذرات به پایان رسیده است. وضعیت موفقیت عددی بهینه‌ساز از کیفیت مسیر جدا گزارش می‌شود:",'',
      '| مثال | شروع منتخب | تعداد تکرارهای هر شروع | اعلام همگرایی L-BFGS-B |','|---|---:|---|---|']
    for r in metrics:
        report=json.loads((HERE/'experiments'/r['case']/'result/report.json').read_text())
        text.append(f"| {NAMES[r['case']]} | {report['selected_start']} | {', '.join(str(c['iterations']) for c in report['candidates'])} | {', '.join(str(c['success']) for c in report['candidates'])} |")
    text+=['',
      'رسیدن به سقف تکرار، اعلام همگرایی نیست؛ نمودارهای خوب به معنی یافتن کمینهٔ سراسری یا بازسازی یکتای جریان نیستند. تمام شروع‌ها و تاریخچهٔ تابع هدف ذخیره شده‌اند. آزمون‌های نرم‌افزاری، بررسی گرادیان و خطاهای واقعی بالا جداگانه مستندند.','',
      'PNG ردیف‌ها ۱۰۳۶۸×۱۶۹۶ پیکسل با وضوح ۳۲۰dpi است؛ PDFها برداری‌اند. نسبت نمایش محورها برای چیدمان کنارهم تنظیم شده و مقادیر عددی x/H و y/H ملاک‌اند. فایل‌های CSV و NPZ دادهٔ خام تمام نمودارها را نگه می‌دارند. برای استفاده از مدل برازش‌شده به README مراجعه کنید.']
    (HERE/'RESULTS.md').write_text('\n'.join(text)+'\n')
    # Parallel bars show the same geometry metric on both old and new outputs.
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    figs=[];labels=[f"{r['case'].replace('_',' ')} / P{r['particle']}" for r in flat];x=np.arange(len(flat));width=.36
    fig,axes=plt.subplots(1,2,figsize=(15,5.6),layout='constrained')
    for ax,metric,title in zip(axes,['symmetric','ordered'],['Symmetric point-to-curve RMS','Ordered normalized-arclength RMS']):
        old=[r[f'timed_{metric}_rmse_pct_H'] for r in flat];new=[r[f'no_time_{metric}_rmse_pct_H'] for r in flat]
        b1=ax.bar(x-width/2,old,width,label='With time',color='#ca7026');b2=ax.bar(x+width/2,new,width,label='Without measured time',color='#11687c')
        ax.bar_label(b1,fmt='%.3f',fontsize=8,padding=3);ax.bar_label(b2,fmt='%.3f',fontsize=8,padding=3)
        ax.set(xticks=x,xticklabels=labels,ylabel='Position mismatch [% H]',title=title);ax.tick_params(axis='x',rotation=35);ax.legend();ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
    fig.suptitle('PI-CFO | Same geometric scores for both reconstruction methods',fontsize=18,fontweight='bold')
    fig.savefig(HERE/'figures/path_score_summary.png',dpi=320);fig.savefig(HERE/'figures/path_score_summary.pdf');figs.append(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,key,title in zip(axes,['profile_relative_l2_difference','normalized_profile_relative_l2_difference'],['Estimated inlet difference','Normalized inlet shape difference']):
        values=[100*r[key] for r in metrics];bars=ax.bar([r['case'].replace('_',' ') for r in metrics],values,color='#11687c',width=.6)
        ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set(ylabel='Difference relative to timed estimate [% L2]',title=title);ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Similar paths need not imply identical inlet profiles',fontsize=18,fontweight='bold')
    fig.savefig(HERE/'figures/inlet_difference_summary.png',dpi=320);fig.savefig(HERE/'figures/inlet_difference_summary.pdf');figs.append(fig)
    with PdfPages(HERE/'summary.pdf') as pdf:
        for fig in figs:pdf.savefig(fig);plt.close(fig)
    combined=fitz.open()
    for p in [HERE/'all_comparisons.pdf',HERE/'summary.pdf']:
        with fitz.open(p) as doc:combined.insert_pdf(doc)
    combined.save(HERE/'presentation_figures.pdf');combined.close()
    overview=[]
    for r in metrics:
        im=Image.open(HERE/'figures'/f"{r['case']}_comparison_row.png").convert('RGB');im.thumbnail((2000,380));overview.append(im)
    sheet=Image.new('RGB',(2040,sum(i.height+30 for i in overview)),'white');offset=15
    for im in overview:sheet.paste(im,(20,offset));offset+=im.height+30
    sheet.save(HERE/'overview.jpg',quality=92)
    environment={name:importlib.metadata.version(name) for name in ['numpy','scipy','jax','jaxlib','matplotlib','Pillow','PyMuPDF','scikit-fem','pytest']}
    (HERE/'environment_versions.json').write_text(json.dumps(dict(python=platform.python_version(),packages=environment),indent=2)+'\n')
    for p in (HERE/'figures').glob('*.png'):
        with Image.open(p) as im:
            if p.stem.endswith('_comparison_row'):assert im.size==(10368,1696)
            im.verify()
    manifest={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix not in ['.zip','.pyc'] and p.name!='SHA256SUMS.json'}
    (HERE/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'geometry_only_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and p.suffix not in ['.zip','.pyc']:z.write(p,arcname=str(Path(HERE.name)/p.relative_to(HERE)))
    print('Packaged',len(metrics),'cases; old files preserved:',verification['existing_files_verified'])

if __name__=='__main__':main()
