# Conditional physics-informed channel-flow operator

یک مدل شرطی برای پیش‌بینی سرعت و فشار جریان آرام در کانال دوبعدی؛ ورودی مدل، **پروفایل سرعت ورودی و عدد رینولدز** است. پس از آموزش، برای شرایط جدید به آموزش مجدد نیاز نیست. این پروژه از ایدهٔ Physics-informed DeepONet استفاده می‌کند و پیاده‌سازی مستقل PyTorch است.

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

## پیش‌بینی بدون آموزش مجدد

```bash
python -m channel_operator.predict --checkpoint artifacts/model.pt --re 55 \
  --coefficients 0.25 -0.1 0.05 0 --output prediction.npz
```

برای ورودی نمونه‌برداری‌شده، به‌جای ضرایب از `--inlet-csv inlet.csv` استفاده کنید. فایل CSV سربرگ `y,u`، حداقل چهار سطر، مختصات صعودی از صفر تا یک و سرعت صفر در دو انتها داشته باشد. سرعت میانگین به یک نرمال می‌شود؛ `Re` باید بر پایهٔ میانگین فیزیکی همان پروفایل وارد شود. ورودی منفی یا اسپلاین دارای سرعت منفی رد می‌شود. پروفایل‌های دارای بازگشت جریان و جریان آشفته در این مدل پشتیبانی نشده‌اند.

## Kaggle و ذخیره‌سازی

```bash
python scripts/make_kaggle.py --destination kaggle-run
kaggle kernels push -p kaggle-run --accelerator NvidiaTeslaT4
kaggle kernels status mohrezsayeh/channel-pi-deeponet-re10-100
kaggle kernels output mohrezsayeh/channel-pi-deeponet-re10-100 -p artifacts/kaggle
```

نوت‌بوک خصوصی است و خودِ کد را بسته‌بندی می‌کند؛ کلید حساب در آن قرار نمی‌گیرد. مدل و دادهٔ تولیدشده از Git مستثنی هستند و در خروجی خصوصی Kaggle ذخیره می‌شوند. دسترسی نوشتن GitHub مستقل از خواندن است و باید عملاً بررسی شود.

## مرجع علمی

Wang, Wang & Perdikaris (2021), *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets*, Science Advances 7(40), eabi8605. DOI: [10.1126/sciadv.abi8605](https://doi.org/10.1126/sciadv.abi8605).
Reference repository: https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets (reviewed at `77e7c747d653b34e79e43df7d88bad87de5c27d8`). Its examples do not directly solve this steady Navier–Stokes channel problem; this implementation follows the operator-learning idea rather than copying those examples.
