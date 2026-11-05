# Conditional physics-informed channel-flow operator

## نسخهٔ اصلاح‌شده V2

**بازبینی طراحی:** [مقایسهٔ منطق پروژه با مقاله و تغییرات لازم برای پروفایل دلخواه](reports/operator_design_audit/README.md). V2 هنوز چهارمودی و هیبریدی است؛ بانک گستردهٔ جدید به trainer آن متصل نشده است و وجود آن بانک به معنی آموزش مدل جدید نیست.

مدل ۷۰هزارگامی و تمام داده‌ها، وزن‌ها و تاریخچهٔ خطا ذخیره شده‌اند. در ۸ آزمون نهاییِ شدیدتر، بیشینهٔ خطای بردار سرعت **۰٫۵۷٪** و فشار **۴٫۲۲٪** بود. معیار سخت‌گیرانهٔ مومنتوم هنوز پاس نشده است؛ جزئیات پنهان نشده‌اند.

- [راهنمای اجرای مدل جدید](README_V2.md)
- [تشخیص علت‌ها و نتایج کنترل‌شده](reports/v2/REPORT.md)
- [منحنی خطای حالات خارج از آموزش](reports/v2/learning_curves.png)
- [اطلس مقایسهٔ هر ۲۴ حالت: مدل، مرجع FEM و نقشهٔ اختلاف](reports/v2/comparisons/README.md)
- [آزمون محدودیت مدل با ورودی ۹نوسانی، در دامنهٔ سرعت و Re قبلی](reports/v2/nine_oscillations/README.md)
- [مولد ورودی گسترده بر اساس کد مقاله؛ ۱۰هزار ورودی پیشنهادی آموزش و هزار ورودی آزمون، بدون مدل آموزش‌دیدهٔ جدید](data/broad_inlets/README.md)
- [تمام چاپ‌های آموزش به CSV](reports/v2/iteration_errors.csv)
- وزن جدید: `models/v2/stream_operator.pt`؛ checkpoint کامل: `models/v2/jax_checkpoint.npz`.

```bash
python -m channel_operator.v2_predict --checkpoint models/v2/stream_operator.pt \
  --re 55 --coefficients 0.25 -0.1 0.05 0 --output prediction.npz
```

## آرشیو مستندات نسخهٔ نخست

یک مدل شرطی برای پیش‌بینی سرعت و فشار جریان آرام در کانال دوبعدی؛ ورودی مدل، **پروفایل سرعت ورودی و عدد رینولدز** است. پس از آموزش، برای شرایط جدید به آموزش مجدد نیاز نیست. این پروژه از ایدهٔ Physics-informed DeepONet استفاده می‌کند و پیاده‌سازی مستقل PyTorch است.

**وضعیت فعلی: نسخهٔ پژوهشی، نه حل عمومی تأییدشده.** مدل آموزش‌دیده و نتایج کامل در [گزارش آزمایش](reports/RESULTS.md) قرار دارند. روی هشت ورودی تازه، بیشینهٔ خطای سرعت حدود ۱٫۹٪ و فشار ۱٫۷٪ بود؛ اما برخی معیارهای فیزیکی پاس نشدند و خطای فشار در آزمون پروفایل‌های مرزیِ شدیدتر تا حدود ۵۸٪ رسید. این محدودیت را در استفاده از مدل لحاظ کنید.

برای اجرای وزن منتخبِ ذخیره‌شده در همین مخزن، بدون آموزش مجدد:

```bash
python -m channel_operator.predict --checkpoint models/channel_operator_candidate.pt \
  --re 55 --coefficients 0.25 -0.1 0.05 0 --output prediction.npz
```

این checkpoint برای inference است و optimizer ندارد؛ checkpointهای کاملِ آموزش در خروجی اجراهای محلی و خصوصی Kaggle نگه داشته می‌شوند. برنامه دربارهٔ معیارهای اعتبارسنجیِ پاس‌نشده هشدار می‌دهد.

## تعریف مسئله

کانال بی‌بعد `0 <= x <= 5`, `0 <= y <= 1` و سیال تراکم‌ناپذیر نیوتنی:

\[
\nabla\cdot\mathbf{u}=0,\qquad
(\mathbf{u}\cdot\nabla)\mathbf{u}=-\nabla p+\frac1{Re}\nabla^2\mathbf{u}.
\]

مقیاس طول ارتفاع کانال `H`، مقیاس سرعت میانگین ورودی `U`، و مقیاس فشار `rho*U^2` است. بنابراین `Re=U*H/nu` و انتگرال پروفایل بی‌بعد ورودی برابر یک است. برای تبدیل خروجی به واحد فیزیکی: `u_phys=U*u`, `v_phys=U*v`, `p_phys=p_ref+rho*U^2*p`. تغییر اندازهٔ سرعت ورودی با تغییر `Re` بازتاب می‌یابد؛ شکل پروفایل جداگانه به مدل داده می‌شود.

- ورودی: `u(0,y)=f(y)`, `v(0,y)=0`.
- دیواره‌ها: `u=v=0`؛ این شرایط و ورودی دقیقاً در معماری اعمال می‌شوند.
- خروجی: شرط طبیعی استاندارد do-nothing از فرم ضعیف با گرادیان نامتقارن: `u_x/Re - p = 0`, `v_x/Re = 0`. فشار نسبت به فشار خارجی صفر است؛ فشار نقطه‌ای کل خروجی جداگانه صفر تحمیل نمی‌شود. این شرط با تنش متقارن کامل متفاوت است و در مدل و حل مرجع یکسان پیاده‌سازی شده است.
- اعتبار هدف: `10 <= Re <= 100` و ورودی‌های صاف، مثبت و سازگار با عدم لغزش دیواره.

پروفایل‌های آموزش از `6*y*(1-y)*(1 + sum(a_k*cos(k*pi*y)))` با چهار مود و `sum(abs(a_k))<=0.6` ساخته و نرمال می‌شوند. ۳۳ نمونه از هر پروفایل و `log(Re)` ورودی branch و مختصات ورودی trunk هستند. اسپلاین مکعبی امکان ورود پروفایل نمونه‌برداری‌شدهٔ دیگر را می‌دهد، اما **دقت برای هر تابع دلخواه یا خارج از توزیع آموزش تضمین نمی‌شود**. برای خانوادهٔ وسیع‌تر باید دامنهٔ آموزش و آزمون گسترش یابد.

## آموزش و اعتبارسنجی

حالت پیش‌فرض **هیبریدی** است: ۴۸ حل مرجع اجزای محدود برای کمک به آموزش، همراه با باقی‌ماندهٔ معادلات، شرط خروجی و بقای دبی. در هر گام، شرط‌های جدید ورودی و رینولدز برای بخش فیزیک نمونه‌برداری می‌شوند. دادهٔ آزمون وارد آموزش یا انتخاب checkpoint نمی‌شود. گزینهٔ `--physics-only` جملهٔ داده را حذف می‌کند؛ همگرایی و دقت آن جداگانه باید سنجیده شود.

حل مرجع مستقل از شبکه، Taylor–Hood P2/P1 با تکرار Picard است. اعتبارسنجی شامل هشت شرط دیده‌نشده، مقایسهٔ دو شبکهٔ اجزای محدود، حل دقیق پوازی، خطای سرعت و فشار، باقی‌ماندهٔ معادلات روی نقاط جدید و خطای دبی است. حد پذیرش سرعت ۵٪، فشار ۱۰٪، دبی ۲٪ است؛ شرایط تکمیلی در `metrics.json` ثبت می‌شوند. پایان آموزش به معنی عبور از این معیارها نیست. پاسخ پوازی در مدل به‌عنوان خط مبنا وجود دارد؛ آزمون پوازی مدل، صحت معادلات را می‌سنجد و شاهد یادگیری تعمیم نیست.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --output artifacts --steps 12000
```

آموزش از GPU موجود استفاده می‌کند و در غیر این صورت روی CPU اجرا می‌شود. تولید حل‌های اجزای محدود روی CPU است. خروجی‌ها: `model.pt`, `training.jsonl`, `training_summary.json`, `metrics.json`, `reference_metadata.json`, `heldout_fields.png` و فایل‌های NPZ. برای اجرای مرحله‌ای از `--stage data|train|evaluate` استفاده کنید؛ ازسرگیری وزن‌ها و optimizer با `--resume artifacts/model.pt` ممکن است، اما دنبالهٔ تصادفی دقیق اجرای قبلی بازسازی نمی‌شود.

صفر بودن کد خروج فرمان یا وضعیت Complete در Kaggle فقط پایان موفق اجرای برنامه را نشان می‌دهد؛ پذیرش علمی با `passed` و تک‌تک `gates` در `metrics.json` مشخص می‌شود.

### نسخهٔ دارای ویژگی‌های ناحیهٔ ورودی

گزینهٔ `--entrance-features` چهار ویژگی برای طول‌های کوتاه نزدیک ورودی به trunk اضافه می‌کند و وزن پیش‌فرض جملهٔ داده را از ۲۰ به ۲۰۰ افزایش می‌دهد. وزن داده با `--data-weight` قابل تعیین است؛ وزن‌های بقای جرم، مومنتوم، خروجی و دبی به‌ترتیب ۱۰، ۱، ۰٫۱ و ۱۰ باقی می‌مانند. تنظیم ضرایب loss، جایگزین آزمون مستقل نیست.

اجرای توسعهٔ محلی شامل آموزش ۴۰۰۰ گامی روی ۲۴ شرط، سپس ادامهٔ همان آموزش عمومی تا گام ۱۰۰۰۰ روی ۱۲۸ شرط با وزن دادهٔ ۱۰۰۰ است. این ادامهٔ آموزش برای کل خانواده انجام می‌شود؛ در زمان پیش‌بینی برای هیچ ورودی مجدداً آموزش انجام نمی‌شود.

```bash
python run_experiment.py --output artifacts/entrance --steps 4000 \
  --train-cases 24 --test-cases 4 --entrance-features
python run_experiment.py --output artifacts/refined --stage data \
  --train-cases 128 --test-cases 4
python run_experiment.py --output artifacts/refined --stage train --steps 10000 \
  --resume artifacts/entrance/model.pt --entrance-features --data-weight 1000
python run_experiment.py --output artifacts/refined --stage evaluate
```

چهار حالت این اجرای محلی برای بررسی توسعه استفاده شده‌اند؛ نباید به‌عنوان آزمون نهایی دست‌نخورده معرفی شوند. برای آزمون مستقلِ سخت‌تر، ۱۶ ترکیب ثابت از ضریب‌های مرزی `+/-0.6` در چهار مود و رینولدزهای ۱۰ و ۱۰۰ در دسترس است:

```bash
python scripts/stress_test.py --checkpoint artifacts/refined/model.pt \
  --output artifacts/refined/stress_test.json
```

در مرحلهٔ بعد، ۶۴ شرط آموزشی با دامنهٔ بیشتر و رینولدزهای ۵۰، ۷۵ و ۱۰۰ به همان مجموعهٔ آموزش افزوده و آموزش عمومی تا گام ۱۶۰۰۰ ادامه داده می‌شود. این فرمان داده‌های آزمون را تغییر نمی‌دهد و از افزودن تکراری جلوگیری می‌کند:

```bash
python scripts/augment_training.py --output artifacts/refined --count 64
python run_experiment.py --output artifacts/refined --stage train --steps 16000 \
  --resume artifacts/refined/model.pt --entrance-features --data-weight 4000
python run_experiment.py --output artifacts/refined --stage evaluate
python scripts/final_benchmark.py --checkpoint artifacts/refined/model.pt \
  --output artifacts/final-benchmark
```

`final_benchmark.py` یک مجموعهٔ هشت‌حالته با seed تازهٔ ۶۴۰۵۲۹ تولید می‌کند که در بررسی توسعه استفاده نشده است. گزارش چهارحالتهٔ توسعه، آزمون هشت‌حالتهٔ تازه و آزمون ۱۶حالتهٔ مرزی باید جداگانه تفسیر شوند.

## پیش‌بینی بدون آموزش مجدد

```bash
python -m channel_operator.predict --checkpoint artifacts/model.pt --re 55 \
  --coefficients 0.25 -0.1 0.05 0 --output prediction.npz
```

برای ورودی نمونه‌برداری‌شده، به‌جای ضرایب از `--inlet-csv inlet.csv` استفاده کنید. فایل CSV سربرگ `y,u`، حداقل چهار سطر، مختصات صعودی از صفر تا یک و سرعت صفر در دو انتها داشته باشد. سرعت میانگین به یک نرمال می‌شود؛ `Re` باید بر پایهٔ میانگین فیزیکی همان پروفایل وارد شود. ورودی منفی یا اسپلاین دارای سرعت منفی رد می‌شود. پروفایل‌های دارای بازگشت جریان و جریان آشفته در این مدل پشتیبانی نشده‌اند.

کنار فایل پیش‌بینی، فایل JSON تشخیصی شامل خطای دبی و باقی‌ماندهٔ معادلات نوشته می‌شود. اگر checkpoint همهٔ معیارها را پاس نکرده باشد یا گزارش اعتبارسنجی هم‌گام موجود نباشد، برنامه هشدار می‌دهد. این تشخیص‌ها جانشین خطای اندازه‌گیری‌شده در برابر CFD نیستند.

## Kaggle و ذخیره‌سازی

```bash
python scripts/make_kaggle.py --destination kaggle-run
kaggle kernels push -p kaggle-run --accelerator NvidiaTeslaT4
kaggle kernels status mohrezsayeh/channel-pi-deeponet-re10-100
kaggle kernels output mohrezsayeh/channel-pi-deeponet-re10-100 -p artifacts/kaggle
```

نوت‌بوک خصوصی است و خودِ کد را بسته‌بندی می‌کند؛ کلید حساب در آن قرار نمی‌گیرد. داده‌های تولیدشده و checkpointهای کامل از Git مستثنی هستند؛ فقط checkpoint کوچکِ inference منتخب در `models/channel_operator_candidate.pt` ثبت شده است. نتایج GPU در خروجی خصوصی Kaggle نگه داشته می‌شوند.

در محیط‌های دارای شبکهٔ محدود، API به `api.kaggle.com` و `www.kaggle.com` نیاز دارد و دریافت فایل‌های خروجی به `www.kaggleusercontent.com` نیز نیاز دارد. موفقیت آموزش مستقل از مجازبودن دانلود فایل‌هاست؛ گزارش عددی را می‌توان از لاگ API نیز دریافت کرد.

## مرجع علمی

Wang, Wang & Perdikaris (2021), *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets*, Science Advances 7(40), eabi8605. DOI: [10.1126/sciadv.abi8605](https://doi.org/10.1126/sciadv.abi8605).
Reference repository: https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets (reviewed at `77e7c747d653b34e79e43df7d88bad87de5c27d8`). Its examples do not directly solve this steady Navier–Stokes channel problem; this implementation follows the operator-learning idea rather than copying those examples.
