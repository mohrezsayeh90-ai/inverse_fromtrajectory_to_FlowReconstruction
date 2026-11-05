# اپراتور جریان کانال با پروفایل ورودی کامل

کد فعال از نو نوشته شده است. **تمام نسخه‌های قبلی با داده‌ها، وزن‌ها و گزارش‌ها در [`old_verssion/`](old_verssion/) نگه داشته شده‌اند**؛ برای هر فایل قدیمی، هش پیش از انتقال در [manifest آرشیو](old_verssion/ARCHIVE_MANIFEST.json) ثبت شده است. اجرای کد قدیمی باید از داخل همان پوشه باشد.

## وضعیت واقعی

مدل جدید شرط ورودی را به چهار مود تبدیل نمی‌کند. ورودی ۹ و ۲۵نوسانی، تخت، نامتقارن و Beta در تست‌های مرزی حفظ شده‌اند. آموزش جدید فقط از معادلات و شرایط مرزی استفاده می‌کند؛ هیچ پاسخ CFD وارد objective نمی‌شود.

**آموزش و ارزیابی مدل جدید در حال انجام است؛ هنوز checkpoint تأییدشدهٔ آمادهٔ استفاده منتشر نشده است.** پاس‌شدن تست کد یا پایان اجرای GPU با دقت میدان یکسان نیست. وضعیت ماشین‌خوان در [STATUS.json](STATUS.json) قرار دارد.

- [اجرای بررسیِ GPU](https://www.kaggle.com/code/mohrezsayeh/channel-general-physics-v3-probe)
- [آزمایش فعلیِ شاخهٔ مستقیم](https://www.kaggle.com/code/mohrezsayeh/channel-general-physics-v3-sensor)
- [آزمایش اپراتورهای مستقل فشار و تابع جریان](https://www.kaggle.com/code/mohrezsayeh/channel-general-physics-v3-pressure)
- [گزارش تطبیق مبنای بازسازی](docs/operator_design_audit/README.md)
- [نگاشت ایرادهای قبلی به تغییرهای کد جدید](docs/IMPLEMENTATION.md)

## منطق مدل

ورودی، خودِ تابع سرعت محوری روی عرض کانال است؛ CSV با تعداد نمونه و مختصات دلخواه پذیرفته می‌شود. همهٔ نمونه‌های آن در یک spline پیوستهٔ C2 حفظ می‌شوند. هیچ برازش چهارمودی یا انتخاب از بانک پاسخ انجام نمی‌شود. دو نوع branch برای مقایسه پیاده شده است: دریافت مستقیم مقادیر تابع روی حسگرهای داخلیِ ثبت‌شده در config، مطابق منطق مقاله؛ و encoder تجمیعی با quadrature. در هر دو، تمام گره‌های اصلی CSV در شرط مرزی حفظ می‌شوند. در نوع مستقیم، تعداد حسگرهای داخلی بخشی از معماریِ آموزش‌دیده است؛ تعداد گره‌های CSV ورودی می‌تواند متفاوت باشد.

شرط مرزی از **انتگرال کل پروفایل** ساخته می‌شود. خروجی streamfunction و فشار است و `u=psi_y`, `v=-psi_x`؛ ورودی، دیواره و بقای جرم در ساختار اعمال می‌شوند. ویژگی‌های Fourier داخل شبکه فقط embedding مختصات‌اند و تابع ورودی را قطع یا فیلتر نمی‌کنند.

آموزش با ۱۰هزار نمونهٔ ورودی از هفت مولد GRF، نوسانی، Beta، تخت، جت، spline تصادفی و سهموی انجام می‌شود. Re در هر batch دوباره نمونه‌برداری می‌شود. این بانک **فقط ورودی** است؛ ورودی کاربر هم لازم نیست عضو این بانک یا همین مولدها باشد. دقت برای شکل تازه باید جداگانه سنجیده شود.

کانال بی‌بعد ۵×۱ و بازهٔ آموزش Re=۱۰ تا ۱۰۰ است. برای ورودی فیزیکی، `U=mean(u_in)`, `Re=UH/nu`, `u_phys=U*u*`, `p_phys=p_external+rho*U²*p*`. CSV باید y/H از صفر تا یک، سرعت صفر در دو انتها و میانگین مثبت داشته باشد. برگشت موضعی جریان در رابط ورودی حذف نمی‌شود، ولی هنوز در آموزش/اعتبارسنجی این نسخه پوشش تأییدشده ندارد.

یک نمایش با اطلاعات محدود نمی‌تواند دقت را برای نوسان نامحدود تضمین کند. پذیرش شکل دلخواه به معنی تغییرندادن آن به یک خانوادهٔ ساده‌تر است؛ کافی‌بودن نمونه‌های CSV و دقت پاسخ مدل همچنان باید بررسی شوند.

## نصب و آزمون

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

برای GPU کگل از JAX/CUDA موجود استفاده کنید؛ بستهٔ CPU را جایگزین نصب GPU نکنید. `requirements-validation.txt` فقط برای ساخت مرجع مستقل اجزای محدود است و trainer آن را لازم ندارد.

## آموزش بدون پاسخ مرجع

```bash
python -m flow_operator.train --output runs/train --steps 50000 \
  --pool-count 10000 --batch 64 --points 48 --width 128 --latent 64 --depth 4 \
  --encoder-kind sensor --lift oseen --physics-normalization input_curvature_sqrt
```

برای این فرمان هیچ فایل دادهٔ حل‌شده یا solver اجزای محدود لازم نیست. نمونه‌های ورودی در همان اجرا ساخته می‌شوند. checkpoint شامل وزن‌ها، Adam، RNG، پیکربندی و هش بانک ورودی است.

```bash
python -m flow_operator.train --output runs/continued --steps 100000 \
  --pool-count 10000 --batch 64 --points 48 --width 128 --latent 64 --depth 4 \
  --encoder-kind sensor --lift oseen --physics-normalization input_curvature_sqrt \
  --resume runs/train/checkpoint.npz
```

برای پایش خطا با مرجعِ جدا، `--reference-monitor validation/references` اضافه کنید. مقایسه در logging انجام می‌شود؛ targetهای مرجع، آمار پاسخ یا errorها به loss و batch آموزش وارد نمی‌شوند. این داده‌ها validation تکرارشونده‌اند و در انتخاب بهترین checkpoint هم استفاده می‌شوند؛ نباید آزمون کور نهایی نامیده شوند.

گزینهٔ `--separate-pressure-trunk` دو branch/trunk مستقل برای فشار و تابع جریان می‌سازد. با `--devices 2` می‌توان همان batch سراسری را با میانگین‌گیری گرادیان روی دو GPU اجرا کرد. این تنظیم‌ها آزمایشی‌اند؛ جزئیات و آزمون تطبیق update در [سند پیاده‌سازی](docs/IMPLEMENTATION.md) آمده است.

## ساخت مرجع و ارزیابی وزن ثابت

```bash
pip install -r requirements-validation.txt
python -m validation.prepare
python -m flow_operator.evaluate --checkpoint runs/train/checkpoint.npz \
  --references validation/references --output runs/evaluation
```

آستانه‌ها در [configs/acceptance.json](configs/acceptance.json) ثبت شده‌اند. ارزیاب نقشه‌های u،v،p مرجع/مدل/اختلاف، داده‌های عددی، خطای مرز، دبی و residual را ذخیره می‌کند. هیچ جابه‌جایی پسینی فشار انجام نمی‌شود. مرجع از همان تابع spline و همان خروجی طبیعی استفاده می‌کند.

## پیش‌بینی بدون آموزش مجدد

پس از داشتن checkpoint آموزش‌دیده و گزارش پذیرش منطبق با هش وزن و امضای پیاده‌سازی:

```bash
python -m flow_operator.predict --checkpoint models/checkpoint.npz \
  --inlet-csv examples/nine_oscillations.csv --re 55 --output prediction.npz
```

فرمان فقط وزن ثابت را اجرا می‌کند؛ optimizer یا CFD فراخوانی نمی‌شود. اولین فراخوانی JAX زمان compile دارد؛ این زمان هم در گزارش ثبت می‌شود. در نبود گزارش پذیرش، فرمان پیش‌فرض متوقف می‌شود. `--allow-unvalidated` فقط برای مشاهدهٔ صریح یک خروجی آزمایشی است و آن را به مدل معتبر تبدیل نمی‌کند.

## نتایج آزمایش‌ها

[جدول رد/قبول، منحنی خطای مرجع و residual واقعی](reports/experiments/README.md). دستورهای بالا مسیر آزمایشی فعلی‌اند و تضمین همگرایی نیستند. lift تحلیلی Oseen فقط یک مقداردهی اولیه بر اساس معادلات خطی‌شده است؛ objective همچنان NS غیرخطی کامل است. مقیاس‌بندی اختیاری residual از مشتق‌های خود ورودی و Re محاسبه می‌شود و به پاسخ CFD نیاز ندارد.

برای هر حالت، نقشهٔ FEM/مدل/اختلاف در پوشهٔ آزمایش مربوطه ذخیره شده است. وزن‌های ردشده در `models/candidates/` با گزارش رد نگه داشته می‌شوند؛ مسیر `models/checkpoint.npz` تا زمان پذیرش مدل ساخته نمی‌شود.

## منبع

Wang, Wang & Perdikaris، [Physics-informed DeepONets](https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets)، کامیت بررسی‌شده `77e7c747d653b34e79e43df7d88bad87de5c27d8`. مولد GRF و ایدهٔ branch/trunk و modified MLP از آن الهام گرفته‌اند. نمایش کامل spline، encoder با quadrature متغیر و lift مرزی، سازگارکردن مستقل برای کانال NS هستند؛ این پروژه بازتولید عین benchmarkهای مقاله نیست.
