"""Editable RTL PowerPoint + matching self-contained HTML/PDF from one slide model."""
import base64,html,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from PIL import Image

HERE=Path(__file__).resolve().parent;A=HERE/'assets';W,H=1600,900
BG='#0b1424';PANEL='#15263b';WHITE='#f0f5fb';MUTED='#9fb3c8';TEAL='#40d9c4';GOLD='#ffc35a'
slides=[]


def font_path(name):
    bundled=HERE/'fonts'/name
    return bundled if bundled.exists() else Path('/usr/share/fonts/truetype/noto')/name


def add(kind,x,y,w,h,**kw):slides[-1]['blocks'].append(dict(kind=kind,x=x,y=y,w=w,h=h,**kw))
def rect(x,y,w,h,color=PANEL):add('rect',x,y,w,h,color=color)
def text(t,x,y,w,h,size=30,color=WHITE,bold=False,rtl=True):add('text',x,y,w,h,text=t,size=size,color=color,bold=bold,rtl=rtl)
def img(name,x,y,w,h):add('image',x,y,w,h,path=str(A/name))
def arrow(x,y,w=60,h=30):add('arrow',x,y,w,h,color=TEAL)
def card(x,y,w,h,label,value,note='',accent=TEAL):
    rect(x,y,w,h);text(label,x+18,y+12,w-36,45,24,MUTED)
    text(value,x+18,y+53,w-36,63,45,accent,True,False)
    if note:text(note,x+18,y+118,w-36,h-126,22)
def bullets(items,x=850,y=200,w=650,size=30,step=115):
    for i,item in enumerate(items):text('• '+item,x,y+i*step,w,step-8,size)
def slide(title,section,foot='',notes=''):
    slides.append(dict(title=title,section=section,notes=notes,blocks=[]))
    text(section,85,35,1420,35,18,TEAL,True,False)
    text(title,85,88,1430,88,43,WHITE,True)
    rect(1480,49,30,5,TEAL)
    text(foot,220,847,1290,35,17,MUTED)
    text(f'{len(slides):02d}',85,845,100,35,20,TEAL,True,False)


def content():
    metrics=json.loads((HERE/'data/forward_metrics.json').read_text())
    m={r['name']:r for r in metrics['all_predeclared_cases']}
    old=ROOT/'flow_reconstruction/examples/corner_launch_preconditioned/result'
    first=json.loads((old/'report.json').read_text());second=json.loads((HERE/'data/inverse_beta4/result/report.json').read_text())
    truth=json.loads((old/'truth_comparison.json').read_text());truth2=json.loads((HERE/'data/inverse_beta4/truth_comparison.json').read_text())
    pct=lambda x:f'{100*x:.2f}%'
    fa_pct=lambda x:pct(x).translate(str.maketrans('0123456789.%','۰۱۲۳۴۵۶۷۸۹٫٪'))
    slide('از پروفایل ورودی تا بازسازی جریان از مسیر ذرات','RESEARCH PRESENTATION · 07 NOV 2026',
          'گزارش پژوهشی قابل بازتولید | نسخهٔ مدل ثابت: گام ۹۴هزار | ارائهٔ چارچوب و شواهد عددی',
          'این ارائه دو بخش دارد: اپراتور مستقیم و حل معکوس. نتایج میدانِ کم‌خطا نمونه‌های منتخب‌اند؛ مدل عمومی هنوز تمام معیارها را پاس نکرده است.')
    text('DeepONet–PINN برای کانال دوبعدی\nو استنتاج معکوس با درگ استوکس',110,205,1390,160,55,WHITE,True)
    img('inverse_beta5.png',90,405,1420,320)
    text('پروفایل → میدان  |  مسیر ذرات → یک جریان سازگار',110,738,1370,60,32,TEAL,True)

    slide('دو نگاشت، یک مدل جریانِ مشترک','01 · RESEARCH QUESTION',notes='در معکوس، شبکهٔ جریان مجدداً آموزش نمی‌بیند؛ ورودی و سرعت میانگین برای هر مشاهده بهینه می‌شوند.')
    for x,title,body in [(90,'مسئلهٔ مستقیم','تابع ورودی f(y) و عدد Re معلوم‌اند؛ میدان u,v,p خواسته می‌شود.'),(830,'مسئلهٔ معکوس','دو مسیر زمان‌دار معلوم‌اند؛ f(y)، سرعت میانگین و میدان تخمین زده می‌شوند.')]:
        rect(x,205,680,270);text(title,x+25,225,630,60,37,TEAL,True);text(body,x+25,305,630,140,32)
    rect(90,530,1420,220);text('ایدهٔ اتصال',120,550,1350,50,32,GOLD,True)
    text('اپراتور جریانِ آموزش‌دیده، یک مدل جانشین منجمد در حلقهٔ استنتاج است. معیار نهاییِ سازگاری مسیر، حل مستقل معادلهٔ ذره در جریان تخمینی است.',120,612,1350,115,32)

    slide('محیط فیزیکی و شرایط مرزی','02 · FORWARD / GEOMETRY','کانال بی‌بعد ۵×۱؛ ورودی محوری است و v ورودی صفر است.')
    img('geometry.png',85,195,1430,365)
    card(85,585,440,195,'دامنهٔ آموزش','10 ≤ Re ≤ 100','بر پایهٔ سرعت میانگین ورودی')
    card(565,585,440,195,'ورودی تابعی','u(0,y)=f(y)','پروفایل صاف و غیرمنفی؛ دیواره‌ها صفر')
    card(1045,585,470,195,'خروجی میدان','u(x,y), v(x,y), p','فشار خارجی صفر؛ شرط خروجی از نوع traction')

    slide('معادلات بی‌بعد و قیدهای ساختاری','03 · FORWARD / PHYSICS','فشارِ تمام نقاط خروجی جداگانه صفر نشده؛ همان شرط traction در FEM و مدل استفاده می‌شود.')
    img('ns_equations.png',90,205,800,465)
    bullets(['پیوستگی و ناویر–استوکس پایا برای سیال تراکم‌ناپذیر.','تابع جریان: u=ψ_y و v=−ψ_x؛ پیوستگی در ساختار اعمال می‌شود.','قید سخت، کل spline ورودی را نگه می‌دارد؛ دیواره و دبی نیز ساختاری‌اند.'],930,210,560,30,170)

    slide('معماری اپراتور فیزیک‌آگاه','04 · FORWARD / ARCHITECTURE','دو DeepONet اسکالر مستقل برای تصحیح ψ و p؛ width=128، latent=64، depth=4.',
          'Branch مقادیر تابع روی ۲۵۷ حسگر داخلی را دریافت می‌کند. همهٔ گره‌های اصلی CSV در spline و lift حفظ می‌شوند. این نمایش، تفکیک محدود دارد و تضمین نوسان نامحدود نیست.')
    for x,y,w,h,label in [(90,210,290,135,'f(y), Re'),(90,445,290,135,'(x,y)'),(470,210,360,135,'Branch ψ\nBranch p'),(470,445,360,135,'Trunk ψ\nTrunk p'),(930,320,150,145,'⊙ , Σ'),(1180,270,330,245,'qψ , qp\nLift + correction\nψ , p → u , v , p')]:
        rect(x,y,w,h);text(label,x+12,y+12,w-24,h-24,30,TEAL,True,False)
    arrow(395,265);arrow(395,500);arrow(850,330);arrow(1097,370)
    text('مسیر مستقلِ شرط مرزی: انتگرالِ تمام تابع ورودی → lift تابع جریان',95,650,1400,70,32,GOLD,True)
    text('هیچ پروژکشن چهارمودی یا انتخاب از بانکِ پاسخ انجام نمی‌شود.',95,735,1400,55,29)

    slide('آموزش از فیزیک؛ مرجع برای پایش و انتخاب وزن','05 · FORWARD / TRAINING','۱۰هزار تابع ورودی از هفت مولد؛ بازنمونه‌گیری Re در batch؛ صفر برچسب میدان در objective.')
    img('loss.png',90,220,850,290)
    bullets(['مولدهای GRF، نوسانی، Beta، تخت، جت، spline و سهموی.','Loss از مومنتوم و traction خروجی ساخته می‌شود؛ مقیاس s فقط تابع ورودی و Re است.','FEM بیرون از گرادیان است؛ اما در پایش تکرارشونده و انتخاب checkpoint استفاده شده است.'],980,200,520,27,170)
    rect(90,580,850,175);text('تفکیک ضروری',120,597,790,50,30,GOLD,True)
    text('Validation مورد استفاده برای انتخاب مدل، آزمون کور نهایی محسوب نمی‌شود.',120,655,790,75,29)
    text('Adam · batch=64 · 48 PDE points/input · 100k steps',105,765,835,55,24,TEAL,False,False)

    slide('پروتکل مقایسهٔ مستقیم: وزن ثابت و شبکهٔ مشترک','06 · FORWARD / VALIDATION','هر چهار حالتِ از پیش مشخص‌شده و تمام خروجی‌ها ذخیره‌اند؛ هیچ تنظیم مجدد وزن انجام نشده است.')
    card(90,210,440,200,'وزن استفاده‌شده','94,000','SHA256: ccf6fa2b…',TEAL)
    card(565,210,440,200,'نمونه‌برداری مقایسه','7,081','نقاط مشترک با تراکم بیشتر نزدیک ورودی',TEAL)
    card(1040,210,470,200,'مرجع مستقل','P2 / P1','FEM تیلور–هود؛ کنترل ریزسازی دو شبکه',TEAL)
    text('تعریف خطا:  ‖پیش‌بینی − مرجع‖₂ / ‖مرجع‖₂',95,460,1400,65,35,WHITE,True)
    text('سرعت به‌صورت بردار (u,v) سنجیده می‌شود. فشار بدون جابه‌جایی پسینی مقایسه می‌شود.',95,550,1400,75,31)
    text('نمونهٔ «کم‌خطا در میدان»: خطای سرعت <۵٪ و فشار <۱۰٪؛ این شرط به معنی پاس‌شدن تمام معیارهای فیزیک نیست.',95,670,1400,105,31,GOLD)

    slide('نمونهٔ اول: ورودی Beta(5,5)، رینولدز ۳۰','07 · FORWARD / LOW-ERROR FIELD CASE','f(y)=630 y⁴(1−y)⁴؛ میانگین ورودی یک؛ همان وزن گام ۹۴هزار.')
    img('beta5_re30_profiles.png',90,190,920,300)
    card(1045,195,465,180,'خطای بردار سرعت',pct(m['beta5_re30']['velocity_relative_l2']),'در برابر FEM')
    text('پروفایل ورودی و برش‌های مقطعی، در کنار نقشهٔ u',1045,395,465,95,29)
    img('beta5_re30_u.png',90,515,1420,275)

    slide('همان ورودی: مقایسهٔ فشار و سرعت عرضی','08 · FORWARD / PRESSURE AND TRANSVERSE VELOCITY','مقیاس رنگ در دو نقشهٔ مرجع و شبکه یکسان است؛ ستون سوم اختلاف مستقیم است.')
    img('beta5_re30_p.png',90,190,1420,265)
    img('beta5_re30_v.png',90,480,1420,265)
    text(f"فشار: {pct(m['beta5_re30']['pressure_relative_l2'])}  |  خطای نسبی v: {pct(m['beta5_re30']['v_relative_l2'])}  |  RMSE(v): {m['beta5_re30']['v_rmse']:.5f} Umean",95,770,1400,50,27,TEAL,True)

    slide('نمونهٔ دوم: شکل ورودی متفاوت، Beta(4,4)','09 · FORWARD / SECOND INLET SHAPE','f(y)=140 y³(1−y)³؛ Re=30؛ همان checkpoint و بدون آموزش مجدد.')
    card(90,190,440,155,'خطای سرعت',pct(m['beta4_re30']['velocity_relative_l2']))
    card(570,190,440,155,'خطای فشار',pct(m['beta4_re30']['pressure_relative_l2']))
    card(1050,190,460,155,'خطای ریزسازی فشار',pct(m['beta4_re30']['mesh_change']['pressure_relative_l2']))
    img('beta4_re30_u.png',90,370,1420,205);img('beta4_re30_p.png',90,595,1420,205)

    slide('تغییر رینولدز، با حفظ پروفایل Beta(5,5)','10 · FORWARD / PARAMETRIC CHECK','این نقاط بررسی توسعه‌ای‌اند؛ نمودار، تعمیم آماری یا تضمین کل بازه را اثبات نمی‌کند.')
    img('reynolds_scan.png',90,190,1420,445)
    text('Re=20: سرعت ۰٫۹۸٪، فشار ۶٫۵۳٪  |  Re=30: سرعت ۰٫۹۱٪، فشار ۳٫۰۸٪',95,660,1400,60,30,TEAL,True)
    text('در Re=40، سرعت همچنان کم‌خطاست؛ فشار به ۱۱٫۷۶٪ می‌رسد و از آستانهٔ میدان عبور می‌کند.',95,730,1400,75,29,GOLD)

    slide('از مشاهدهٔ ذرات به یک جریان سازگار','11 · INVERSE / FRAMEWORK','مدل جانشین منجمد است؛ برای هر جفت مسیر جدید، تنها پارامترهای ورودی و سرعت میانگین بهینه می‌شوند.')
    labels=['دو مسیر زمان‌دار\n+ شرایط پرتاب','f(y), A\nپارامترهای مجهول','اپراتور منجمد\nu,v,p','معادلهٔ ذره\nحل مستقل ODE']
    for i,label in enumerate(labels):
        x=90+i*370;rect(x,230,310,170);text(label,x+15,255,280,125,30,WHITE,True)
        if i<3:arrow(x+320,300,40,24)
    rect(90,470,1420,230);text('حلقهٔ استنتاج',120,490,1360,50,34,TEAL,True)
    text('خطای مسیر + همواری پروفایل + فیزیک جریان → تنظیم f و A → تکرار؛ سپس بررسی حرکت ذرات در همان جریان.',120,560,1360,100,32)
    text('روش عمومیِ حل معکوس است؛ یک شبکهٔ معکوسِ آموزش‌دیده با یک forward-pass آنی نیست.',100,745,1400,65,27,GOLD)

    slide('سناریوی مشاهده: دو پرتاب از گوشه‌های مخالف','12 · INVERSE / GIVEN DATA','ورودی حل معکوس: زمان و مکان ذرات، سرعت اولیه، St و ویسکوزیتهٔ معلوم. میدان حقیقی وارد loss نمی‌شود.')
    img('observed_tracks.png',90,190,1420,325)
    card(90,545,440,215,'شرایط پرتاب','(6, −1.2) / (−6, 1.2)','از (۰،۱) و (۵،۰)؛ رو به مرکز هندسی',TEAL)
    card(570,545,440,215,'مشاهده','101 × 2','t تا ۱٫۲؛ نویز موقعیت ۰٫۰۰۰۲H',TEAL)
    card(1050,545,460,215,'ذرات نقطه‌ای','St = 0.42','درگ استوکس؛ اثر یک‌طرفهٔ سیال بر ذره',TEAL)

    slide('مدل ذره و فرم انتگرالیِ مقاوم‌تر به نویز','13 · INVERSE / PARTICLE PHYSICS','نمونهٔ ایده‌آل استوکس؛ برای ذرهٔ واقعی باید Re ذره کوچک باشد. زمان‌بندی مسیرها ضروری است.')
    img('particle_equations.png',90,190,1420,190)
    img('weak_equation.png',90,410,1420,235)
    text('فرم انتگرالی نیاز به مشتق‌گیری مرتبهٔ دوم از مسیر پرنویز را حذف می‌کند. حل ODE نهایی روی مسیرِ پیش‌بینی‌شدهٔ خود ذره انجام می‌شود.',95,690,1400,110,30,TEAL)

    slide('مجهول‌ها و قیدهای حل معکوس','14 · INVERSE / REGULARIZATION','تعداد گره‌ها و وزن priorها قابل تنظیم‌اند؛ این انتخاب‌ها جای اطلاعات مشاهده را نمی‌گیرند.')
    img('inverse_loss.png',90,220,800,270)
    bullets(['۶۵ گرهٔ spline؛ ۶۳ پارامتر آزاد شامل سرعت میانگین A.','پروفایل غیرمنفی با میانگین نرمال‌شده؛ Re=A Re_ref.','پیش‌شرط‌سازی وارون‌پذیر روی تمام گره‌ها؛ بدون حذف مود.','چند شروع و گزارش همگرایی؛ جواب نهایی الزاماً مینیمم سراسری نیست.'],940,200,560,29,140)
    rect(90,550,800,190);text('وزن‌های شبکه در تمام این مراحل ثابت می‌مانند.',120,585,740,100,34,TEAL,True)
    text('L-BFGS-B · 2–3 starts · 400–500 iterations',115,700,775,50,23,MUTED,False,False)

    slide('بازسازی سازگار با مسیر: آزمایش Beta(5,5)','15 · INVERSE / FLUID + PARTICLE RESULT','پس‌زمینه و خطوط سفید: سیال تخمینی؛ مسیر رنگی: ذرات در همان سیال؛ دایرهٔ توخالی: مشاهده.')
    img('inverse_beta5.png',90,190,1420,410)
    for i,(lab,tr) in enumerate(zip(['ذرهٔ بالا–چپ','ذرهٔ پایین–راست'],first['trajectories'])):
        card(90+480*i,625,440,165,lab,f"{100*(2**.5)*tr['position_rmse']:.4f}% H",'RMSE دوبعدی موقعیت',TEAL)
    card(1050,625,460,165,'سرعت میانگین تخمینی',f"{first['mean_speed']:.4f}",'مقدار مولد: ۱ Uref',GOLD)

    slide('تکرار چارچوب برای ورودی Beta(4,4)','16 · INVERSE / SECOND FLOW EXPERIMENT','دادهٔ مصنوعی از FEM مستقل تولید شد؛ تنها مسیرها و اطلاعات پرتاب به استنتاج داده شدند.')
    img('inverse_beta4.png',90,190,1420,410)
    for i,(lab,tr) in enumerate(zip(['ذرهٔ بالا–چپ','ذرهٔ پایین–راست'],second['trajectories'])):
        card(90+480*i,625,440,165,lab,f"{100*(2**.5)*tr['position_rmse']:.4f}% H",'RMSE دوبعدی موقعیت',TEAL)
    card(1050,625,460,165,'سرعت میانگین تخمینی',f"{second['mean_speed']:.4f}",'مقدار مولد: ۱ Uref',GOLD)

    slide('آیا جریان تخمینی، همان مسیر را بازتولید می‌کند؟','17 · INVERSE / TRAJECTORY CONSISTENCY','خطا در زمان‌های یکسان؛ درصدها نسبت به H هستند، نه درصد مسیر پیموده‌شده یا «درصد دقت مدل».')
    img('inverse_errors.png',90,190,1420,430)
    text('دو شکل ورودی و دو جفت مسیر بررسی شدند؛ این‌ها نمونه‌های توسعه‌ای‌اند، نه آزمون کور تعمیم.',95,655,1400,65,31)
    text('نتیجهٔ قابل ادعا: مدل، جریان‌هایی تخمین زده که حرکتِ ذرات در آن‌ها به مسیرهای داده‌شده نزدیک است.',95,735,1400,75,31,TEAL,True)

    slide('جمع‌بندی: دستاورد مشخص و محدودهٔ ادعا','18 · CONCLUSIONS','کد، وزن، ورودی‌ها، مراجع، شکل‌ها و فایل‌های ارائه در مخزن همراه نسخه و هش ذخیره می‌شوند.')
    for y,label,body,color in [(200,'مدل مستقیم','برای سه نمونهٔ مشخص، سرعت برداری زیر ۱٪ و فشار زیر ۷٪ اختلاف با FEM دارد.',TEAL),
                              (390,'مسئلهٔ معکوس','دو مسیر ذره به یک ورودی و میدان سازگار نسبت داده می‌شوند؛ بازشبیه‌سازی مستقل مسیر انجام شده است.',TEAL),
                              (580,'حد اعتبار','دقت عمومی و یکتایی بازسازی هنوز تأیید نشده‌اند؛ تطبیق مسیر با بازیابی دقیق کل میدان یکسان نیست.',GOLD)]:
        rect(90,y,1420,165);text(label,120,y+12,1340,50,34,color,True);text(body,120,y+70,1340,80,31)

    slide('پشتیبان: همهٔ حالت‌های اصلیِ وزن منتخب','A1 · FULL ORIGINAL VALIDATION','این جدولِ زمینه، از معرفی نمونه‌های منتخب به‌عنوان دقت تضمین‌شدهٔ اپراتور جلوگیری می‌کند.')
    img('all_original_cases.png',90,190,1420,410)
    text('در مجموعهٔ اصلی، سرعت، فشار و مومنتوم در همهٔ حالت‌ها هم‌زمان پذیرفته نشدند؛ حالت ۹نوسانی همچنان دشوار است.',95,645,1400,95,30,GOLD)
    text('دو ورودی ۵۱۳گرهیِ رزروشده در این ارائه استفاده نشده‌اند.',95,755,1400,45,27)

    slide('پشتیبان: مسیر خوب، لزوماً ورودی یکتا نمی‌دهد','A2 · IDENTIFIABILITY','رتبهٔ محلی، به مختصات پارامترها و آستانه وابسته است؛ پراکندگی شروع‌ها confidence interval نیست.')
    img('identifiability.png',90,190,1420,390)
    text(f"آزمایش اول: خطای ورودی {fa_pct(truth['inlet_relative_l2'])} و خطای کل سرعت {fa_pct(truth['velocity_relative_l2'])}، با وجود خطای مسیر نزدیک ۰٫۱ درصد ارتفاع کانال.",95,625,1400,90,31,GOLD)
    text(f"آزمایش دوم: خطای ورودی {fa_pct(truth2['inlet_relative_l2'])} و خطای کل سرعت {fa_pct(truth2['velocity_relative_l2'])}. این اختلاف‌ها در داده‌های ارائه محفوظ‌اند.",95,735,1400,75,29)

    slide('پشتیبان: بازتولید، نسخه‌ها و آزمون نرم‌افزار','A3 · REPRODUCIBILITY','چارچوب الهام‌گرفته از Wang, Wang & Perdikaris / Physics-informed DeepONets؛ benchmark کانال، پیاده‌سازی مستقل است.')
    bullets(['وزن مشترک همهٔ نمونه‌های این ارائه: گام ۹۴هزار؛ SHA256 با ccf6fa2b آغاز می‌شود.','FEM و کنترل ریزسازی مستقل؛ خطاهای میدان و تمام حالت‌های بررسی‌شده در JSON/NPZ محفوظ‌اند.','۶۸ تست نرم‌افزار: مشتق‌ها، قید مرزی، حرکت ذره، گرادیان معکوس و بارگذاری جواب؛ این عدد، تعداد آزمون موفق دقت میدان نیست.','کد تولید ارائه، PPTX قابل ویرایش، PDF، نسخهٔ HTML، یادداشت گوینده و دادهٔ شکل‌ها همراه ارائه‌اند.'],100,210,1400,31,145)
    return dict(forward_metrics=metrics,first_inverse=first,second_inverse=second,truth_comparisons=[truth,truth2])


def fit_image(path,x,y,w,h):
    iw,ih=Image.open(path).size;r=min(w/iw,h/ih);nw,nh=iw*r,ih*r
    return x+(w-nw)/2,y+(h-nh)/2,nw,nh


def add_speaker_notes():
    notes=[
    'هدف، ساخت دو نگاشت مرتبط است: حل مستقیم برای تابع ورودی و سپس استفاده از همان اپراتور در یک استنتاج معکوس. این نسخه گزارش چارچوب و شواهد توسعه‌ای است. مدل مبنا هنوز مجوز ادعای دقت برای تمام پروفایل‌ها را ندارد. زمان پیشنهادی: ۳۰ ثانیه.',
    'در مستقیم، شکل تابع ورودی و Re داده می‌شوند. در معکوس، تنها دو مسیر زمان‌دار و اطلاعات ذرات داده می‌شود. شبکهٔ جریان در معکوس منجمد است؛ این روش با بازآموزی PINN برای هر مسئله یا یک inverse network تک‌مرحله‌ای متفاوت است. زمان پیشنهادی: ۴۵ ثانیه.',
    'طول کانال پنج برابر ارتفاع است. ورودی فقط مؤلفهٔ محوری دارد و v ورودی صفر است؛ v در میدان داخلی می‌تواند غیرصفر باشد. فشار نقطه‌ای کل خروجی صفر نشده است؛ خروجی از نوع traction با فشار خارجی صفر است. FEM نیز همین مسئلهٔ مرزی را حل می‌کند. زمان: ۴۵ ثانیه.',
    'مقیاس سرعت در اپراتور مستقیم، سرعت میانگین ورودی است. از تابع جریان برای اعمال پیوستگی استفاده شده است. کل spline ورودی در lift مرزی وارد می‌شود؛ پاس‌شدن جرم و مرز ساختاری به‌تنهایی دقت میدان داخلی را ثابت نمی‌کند. زمان: ۴۵ ثانیه.',
    'دو شبکهٔ scalar مستقل برای تصحیح تابع جریان و فشار وجود دارد. هرکدام branch و trunk خود را دارند. branch از ۲۵۷ حسگر داخلی و Re و ویژگی مشتق‌شده از خود ورودی استفاده می‌کند. تمام نقاط CSV در قید مرزی حفظ می‌شوند. این نمایش یک تفکیک عددی محدود دارد، نه تضمین تمام توابع ممکن. زمان: ۶۰ ثانیه.',
    'بانک شامل فقط تابع‌های ورودی است، نه جواب‌های CFD. Loss مستقیم از باقیماندهٔ مومنتوم و traction ساخته می‌شود؛ s از انحنای ورودی و Re محاسبه شده است. مرجع FEM در گرادیان استفاده نشده، اما برای پایش تکرارشونده و انتخاب checkpoint استفاده شده است؛ بنابراین آن مراجع را آزمون کور ننامید. زمان: ۶۰ ثانیه.',
    'همهٔ نتیجه‌های مستقیم این ارائه از checkpoint گام ۹۴۰۰۰ گرفته شده‌اند. چهار حالت بررسی از پیش در prepare_results مشخص‌اند؛ سه مورد معیار میدانِ سرعت کمتر از ۵٪ و فشار کمتر از ۱۰٪ را دارند. این شرط جای تمام gateهای فیزیک را نمی‌گیرد. خطا روی ۷۰۸۱ نقطهٔ مشترک، با تراکم بیشتر نزدیک ورودی، محاسبه می‌شود. زمان: ۴۵ ثانیه.',
    'ورودی Beta(5,5) با میانگین یک است. خطوط ممتد برش‌ها FEM و خطوط خط‌چین شبکه‌اند. خطای بردار سرعت حدود ۰٫۹۱٪ است؛ نقشهٔ اختلاف با مقیاس متفاوت از دو نقشهٔ میدان نمایش داده می‌شود تا ساختار خطا دیده شود. این یک مثال عملکرد مناسب است، نه نتیجهٔ کل دامنه. زمان: ۶۰ ثانیه.',
    'فشار حدود ۳٫۰۸٪ خطا دارد. مؤلفهٔ عرضی از نظر اندازه کوچک‌تر است: خطای نسبی آن حدود ۹٫۲۶٪ و RMSE آن ۰٫۰۰۴۱۶ برابر سرعت میانگین است. کوچک‌بودن خطای سرعت برداری نباید به همان درصد برای تک‌تک مؤلفه‌ها تعبیر شود. فشار بدون alignment پسینی مقایسه شده است. زمان: ۵۰ ثانیه.',
    'اینجا شکل تابع نیز تغییر کرده و Beta(4,4) است. سرعت حدود ۰٫۷۶٪ و فشار حدود ۲٫۷۸٪ اختلاف دارند. تغییر فشار بین دو شبکهٔ FEM حدود ۰٫۰۲۴٪ است و بسیار کوچک‌تر از اختلاف مدل و مرجع است. وزن شبکه همان وزن مثال قبل است. این نمونه درون خانوادهٔ توابع مطالعه‌شده است و ادعای خارج‌توزیع نیست. زمان: ۵۰ ثانیه.',
    'حالت‌های Re=20،30،40 از یک پروفایل مشترک و یک checkpoint هستند. در Re=20 سرعت حدود ۰٫۹۸٪ و فشار حدود ۶٫۵۳٪ خطا دارند. در Re=40، سرعت همچنان حدود ۰٫۹۲٪ است ولی فشار حدود ۱۱٫۷۶٪ می‌شود. نشان‌دادن همین مورد مانع برداشت نادرستِ دقت یکنواخت در کل بازه است. زمان: ۵۰ ثانیه.',
    'مسیرها، نقطه و سرعت اولیه و زمان پاسخ ذره را می‌گیریم. پارامترهای شکل ورودی و سرعت میانگین بهینه می‌شوند و اپراتور منجمد هر بار میدان را می‌دهد. پس از استنتاج، حرکت ذرات مستقل و با میدان در موقعیت پیش‌بینی‌شدهٔ خود ذره حل می‌شود. زمان: ۵۰ ثانیه.',
    'مکان‌های شروع گوشه‌های مخالف‌اند و جهت پرتاب اولیه به سمت مرکز هندسی است. سرعت‌های اولیه ۶ و منفی ۶ در راستای x هستند؛ مؤلفه‌های y برابر منفی ۱٫۲ و مثبت ۱٫۲ است. St=0.42 معلوم فرض شده است. دو مسیر هرکدام ۱۰۱ نمونه با نویز مصنوعی ۰٫۰۰۰۲H دارند. این آزمایش عددی مصنوعی است، نه آزمایش واقعی ذرات. زمان: ۵۰ ثانیه.',
    'نیرو فقط درگ استوکس و اثر سیال بر ذره یک‌طرفه است. زمان‌بندی برای استنتاج لازم است؛ صرف منحنی هندسی کافی نیست. فرم انتگرالی دقیق، نیاز به مشتق دوم دادهٔ پرنویز را حذف می‌کند. quadrature روی مسیر مشاهده‌شده برای استنتاج استفاده می‌شود؛ سپس حل ODE مستقل از مسیر مشاهده‌شده کنترل نهایی است. زمان: ۶۰ ثانیه.',
    'شکل ورودی گره‌ای است؛ هیچ چهار مود ثابتی تحمیل نشده است. میانگین شکل نرمال می‌شود و A جداگانه برآورد می‌شود. با ویسکوزیتهٔ معلوم، Re=A Re_ref است. priorهای انحنا و فیزیک از اطلاعات معلوم استفاده می‌کنند و target میدان ندارند. شروع‌ها به سقف iteration رسیده‌اند؛ جواب، بهترین تخمین موجود است و مینیمم سراسری اثبات نشده است. زمان: ۶۰ ثانیه.',
    'این همان جریان بازسازی‌شده است، نه میدان حقیقی که به‌عنوان پاسخ در پس‌زمینه گذاشته شده باشد. رنگ‌ها سرعت سیال، خطوط سفید جهت سیال و رنگ‌های نارنجی و صورتی حرکت ذرات اینرسی‌دار هستند. RMSE دوبعدی مسیرها ۰٫۰۹۴۴ و ۰٫۰۶۸۲ درصد H است. A حدود ۱٫۰۰۲۴ در برابر مقدار مولد یک است. زمان: ۶۰ ثانیه.',
    'برای شکل Beta(4,4) یک مجموعه مسیر مستقل از FEM همین شکل ساخته شد. حل معکوس به میدان یا تابع ورودی مولد دسترسی نداشت. RMSE دوبعدی مسیرها حدود ۰٫۰۶۲۸ و ۰٫۰۵۳۳ درصد H است. مسیر نزدیک، شاهد سازگاری شرطی است؛ خطای کل میدان در پشتیبان گزارش شده است. زمان: ۵۰ ثانیه.',
    'نمودار، فاصلهٔ اقلیدسی موقعیت خروجی و ورودی در زمان یکسان را نشان می‌دهد. RMSE دوبعدی ریشهٔ میانگین dx²+dy² است و با RMSE مختصاتیِ گزارش‌های قبلی ضریب رادیکال دو تفاوت دارد. درصد H، درصد خطای موقعیت نسبت به ارتفاع کانال است؛ آن را درصد دقت شبکه ننامید. این مسیرها برای استنتاج استفاده شده‌اند و آزمون دیده‌نشده نیستند. زمان: ۵۰ ثانیه.',
    'دستاورد اول، اپراتور با نمونه‌های مشخصِ توافق خوب با FEM است. دستاورد دوم، اتصال عملی آن به استنتاج از ذرات و بازتولید مسیرهاست. مرز ادعا این است که دقت عمومی و یکتایی کل جریان هنوز ثابت نشده است. گام بعدی: بهبود دقت فیزیکی اپراتور و طراحی مشاهده‌های حساس‌تر یا دادهٔ بیشتر برای رفع نامعینی. زمان: ۶۰ ثانیه.',
    'این اسلاید برای سؤال دربارهٔ نتیجهٔ تمام benchmarkهاست. موارد ناموفق حذف نشده‌اند. در وزن منتخب، حالت ۹نوسانی خطای سرعت بزرگ‌تر از ۵٪ دارد و بعضی فشارها از ۱۰٪ عبور می‌کنند. تمام مومنتوم‌ها نیز زیر آستانهٔ اصلی ۰٫۰۵ نیستند. دو تابع رزروشدهٔ ۵۱۳گرهی در این کار مصرف نشده‌اند.',
    'برای سؤال دربارهٔ بازیابی ورودی: دو مسیر محدود، به‌ویژه در حضور خطای مدل و نویز، کل تابع را به‌طور مطمئن تعیین نمی‌کنند. در مثال اول خطای ورودی حدود ۴۳٪ و خطای کل سرعت حدود ۲۷٪ است. در مثال دوم نیز این خطاها صفر نیستند. رتبهٔ عددی Jacobian به پارامترگذاری وابسته است؛ این نتیجه تنها به محدودیت اطلاعات اشاره می‌کند.',
    '۶۸ تست، آزمون صحت پیاده‌سازی است؛ نه ۶۸ نتیجهٔ موفقِ دقت میدان. آزمون‌ها مشتق AD، حرکت تحلیلی ذره، مرز، checkpoint و گرادیان معکوس را بررسی می‌کنند. نسخهٔ ثابت وزن، داده‌ها، نمودارها و کد تولید ارائه در مخزن محفوظ‌اند. ارجاع: Wang, Wang & Perdikaris, Physics-informed DeepONets؛ کد رسمی PredictiveIntelligenceLab/Physics-informed-DeepONets.'
    ]
    assert len(notes)==len(slides)
    for s,n in zip(slides,notes):s['notes']=n


def build_html():
    font=base64.b64encode(font_path('NotoSansArabic-Regular.ttf').read_bytes()).decode()
    bold=base64.b64encode(font_path('NotoSansArabic-Bold.ttf').read_bytes()).decode()
    css=f'''@font-face{{font-family:NotoArabic;src:url(data:font/ttf;base64,{font})}}@font-face{{font-family:NotoArabic;src:url(data:font/ttf;base64,{bold});font-weight:700}}
    *{{box-sizing:border-box}}body{{margin:0;background:#111}}.slide{{position:relative;width:1600px;height:900px;background:{BG};overflow:hidden;page-break-after:always;break-after:page}}
    .txt{{position:absolute;font-family:NotoArabic,Arial,sans-serif;white-space:pre-wrap;line-height:1.45;overflow:hidden;unicode-bidi:plaintext}}
    .box{{position:absolute;border-radius:15px}}img{{position:absolute;background:white;border-radius:9px}}@page{{size:1600px 900px;margin:0}}@media print{{body{{background:white}}.slide{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}'''
    pages=[]
    for i,s in enumerate(slides):
        items=[]
        for j,b in enumerate(s['blocks']):
            x,y,w,h=[b[k] for k in ['x','y','w','h']];style=f'left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
            if b['kind']=='text':
                style+=f"font-size:{b['size']}px;color:{b['color']};font-weight:{700 if b['bold'] else 400};direction:{'rtl' if b['rtl'] else 'ltr'};text-align:{'right' if b['rtl'] else 'left'};"
                items.append(f'<div class="txt" data-slide="{i}" data-block="{j}" style="{style}">{html.escape(b["text"])}</div>')
            elif b['kind']=='image':
                x,y,w,h=fit_image(b['path'],x,y,w,h);enc=base64.b64encode(Path(b['path']).read_bytes()).decode()
                items.append(f'<img src="data:image/png;base64,{enc}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">')
            elif b['kind']=='rect':items.append(f'<div class="box" style="{style}background:{b["color"]}"></div>')
            else:items.append(f'<div style="position:absolute;{style}background:{b["color"]};clip-path:polygon(0 30%,65% 30%,65% 0,100% 50%,65% 100%,65% 70%,0 70%)"></div>')
        pages.append(f'<section class="slide" id="slide-{i+1}">'+''.join(items)+'</section>')
    (HERE/'defense_fa.html').write_text('<!doctype html><html lang="fa"><meta charset="utf-8"><style>'+css+'</style><body>'+''.join(pages)+'</body></html>')


def build_pptx():
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    scale=13.333333/1600
    def measure(v):return Inches(v*scale)
    def color(value):return RGBColor.from_string(value.lstrip('#'))
    for s in slides:
        sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=color(BG)
        for b in s['blocks']:
            x,y,w,h=[b[k] for k in ['x','y','w','h']]
            if b['kind']=='image':
                x,y,w,h=fit_image(b['path'],x,y,w,h);sl.shapes.add_picture(b['path'],measure(x),measure(y),measure(w),measure(h));continue
            if b['kind'] in ['rect','arrow']:
                shape=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if b['kind']=='rect' else MSO_SHAPE.RIGHT_ARROW,measure(x),measure(y),measure(w),measure(h))
                shape.fill.solid();shape.fill.fore_color.rgb=color(b['color']);shape.line.fill.background()
                if b['kind']=='rect':shape.adjustments[0]=.08
                continue
            shape=sl.shapes.add_textbox(measure(x),measure(y),measure(w),measure(h));tf=shape.text_frame
            tf.clear();tf.word_wrap=True;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.TOP
            for i,line in enumerate(b['text'].split('\n')):
                p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.alignment=PP_ALIGN.RIGHT if b['rtl'] else PP_ALIGN.LEFT
                pPr=p._p.get_or_add_pPr();pPr.set('rtl','1' if b['rtl'] else '0');p.line_spacing=1.3
                p.space_before=p.space_after=Pt(0)
                for run in p.runs:
                    run.font.name='Noto Sans Arabic';run.font.size=Pt(b['size']*.6);run.font.bold=b['bold'];run.font.color.rgb=color(b['color'])
                    from pptx.oxml.xmlchemy import OxmlElement
                    rp=run._r.get_or_add_rPr();rp.set('lang','fa-IR' if b['rtl'] else 'en-US')
                    cs=OxmlElement('a:cs');cs.set('typeface','Noto Sans Arabic');rp.append(cs)
        sl.notes_slide.notes_text_frame.text=s['notes'] or (s['title']+'\n'+next((b['text'] for b in s['blocks'] if b['kind']=='text' and b['y']==847),''))
        for paragraph in sl.notes_slide.notes_text_frame.paragraphs:
            paragraph.alignment=PP_ALIGN.RIGHT;paragraph._p.get_or_add_pPr().set('rtl','1')
            for run in paragraph.runs:run.font.name='Noto Sans Arabic';run.font.size=Pt(12)
    prs.save(HERE/'defense_fa.pptx')


def render():
    from playwright.sync_api import sync_playwright
    previews=HERE/'previews';previews.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport=dict(width=1600,height=900),device_scale_factor=1)
        page.set_content((HERE/'defense_fa.html').read_text(),wait_until='load');page.evaluate('document.fonts.ready')
        # Fit text based on actual Arabic browser shaping, then propagate the
        # resulting font sizes to the native editable PowerPoint text boxes.
        resized=page.evaluate('''() => {let changes=[];document.querySelectorAll('.txt').forEach(e=>{let n=0;while((e.scrollHeight>e.clientHeight+2||e.scrollWidth>e.clientWidth+2)&&n<60){e.style.fontSize=(parseFloat(e.style.fontSize)-.5)+'px';n++}if(n)changes.push({slide:+e.dataset.slide,block:+e.dataset.block,size:parseFloat(e.style.fontSize)});});return changes;}''')
        for change in resized:slides[change['slide']]['blocks'][change['block']]['size']=change['size']
        overflows=page.evaluate("() => [...document.querySelectorAll('.txt')].filter(e=>e.scrollHeight>e.clientHeight+2||e.scrollWidth>e.clientWidth+2).map(e=>({slide:e.dataset.slide,text:e.textContent}))")
        assert not overflows,overflows
        page.emulate_media(media='print');page.pdf(path=str(HERE/'defense_fa.pdf'),print_background=True,prefer_css_page_size=True)
        for i in range(len(slides)):page.locator('.slide').nth(i).screenshot(path=str(previews/f'slide_{i+1:02}.png'))
        browser.close()
    build_html()  # Persist the fitted sizes as well as the PDF-rendered DOM.
    (HERE/'layout_validation.json').write_text(json.dumps(dict(slides=len(slides),text_overflows=overflows,automatic_font_adjustments=resized),ensure_ascii=False,indent=2))
    # Contact sheets, four by three, for manual review.
    for first in range(0,len(slides),12):
        sheet=Image.new('RGB',(1600,3*255),'#172638')
        for j,index in enumerate(range(first,min(first+12,len(slides)))):
            im=Image.open(previews/f'slide_{index+1:02}.png');im.thumbnail((390,220));sheet.paste(im,((j%4)*400,(j//4)*255+20))
        sheet.save(previews/f'contact_{first//12+1}.jpg',quality=90)


def main():
    sources=content();add_speaker_notes();build_html();render();build_pptx()
    (HERE/'slide_model.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2))
    (HERE/'sources.json').write_text(json.dumps(dict(checkpoint_sha256=sources['forward_metrics']['checkpoint_sha256'],
        forward_case_metrics='data/forward_metrics.json',first_inverse_source='flow_reconstruction/examples/corner_launch_preconditioned/result',
        second_inverse_source='data/inverse_beta4/result',all_figures_from_saved_or_new_numeric_results=True,
        trained_model_ready=False,reserved_513_knot_tests_used=False),indent=2))
    notes=['# یادداشت‌های گوینده\n']
    for i,s in enumerate(slides,1):
        notes.append(f"## اسلاید {i}: {s['title']}\n\n"+(s['notes'] or 'هدف: '+s['title'])+'\n')
    (HERE/'speaker_notes.md').write_text('\n'.join(notes))
    fonts=HERE/'fonts';fonts.mkdir(exist_ok=True)
    for name in ['NotoSansArabic-Regular.ttf','NotoSansArabic-Bold.ttf']:
        source=font_path(name)
        if source.resolve()!=(fonts/name).resolve():shutil.copy2(source,fonts/name)
    license=Path('/usr/share/doc/fonts-noto-core/copyright')
    if license.exists():shutil.copy2(license,fonts/'COPYRIGHT.txt')
    print('Built',len(slides),'slides: PowerPoint, PDF, HTML, editable text, and numeric sources.')


if __name__=='__main__':main()
