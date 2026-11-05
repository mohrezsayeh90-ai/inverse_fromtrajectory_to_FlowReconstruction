# استفاده از نسخهٔ اصلاح‌شده V2

مدل ثابتِ ۷۰هزارگامی، پروفایل ورودی و Re را به u,v,p بی‌بعد نگاشت می‌کند. روش هیبریدی است: ۲۸۸ حل CFD برای آموزش، همراه با معادلات NS؛ ۱۶ حالت مستقل فقط برای پایش و ۸ حالت دیگر فقط برای آزمون نهایی. [گزارش کامل](reports/v2/REPORT.md) و [منحنی خطا](reports/v2/learning_curves.png) را ببینید.

## مقایسهٔ تصویری تمام حالت‌ها

[اطلس نتایج](reports/v2/comparisons/README.md) شامل مقایسهٔ مدل و مرجع برای تمام ۲۴ حالت است: میدان‌های u، v، فشار و اندازهٔ سرعت، نقشهٔ اختلاف، مقاطع کانال و جدول خطاهای هر حالت.

- [PDF شامل ۴۹ صفحه](reports/v2/comparisons/comparison_atlas.pdf)
- [داشبورد تعاملی آفلاین؛ دانلود و در مرورگر باز کنید](reports/v2/comparisons/dashboard.html)
- [جدول خلاصهٔ خطاها](reports/v2/comparisons/summary.csv)
- [مقادیر نقطه‌ای مدل، مرجع و اختلاف به CSV](reports/v2/comparisons/pointwise_fields.csv)
- [تمام میدان‌ها برای پردازش با Python](reports/v2/comparisons/all_fields.npz)

این خروجی‌ها بدون آموزش مجدد و روی همان نقاط مرجع ایجاد می‌شوند:

```bash
python scripts/render_v2_comparisons.py
```

### آزمایش ورودی با ۹ نوسان کامل

[مقایسهٔ تشخیصی با مرجع FEM ریزشده](reports/v2/nine_oscillations/README.md): دامنهٔ سرعت و Re در محدودهٔ قبلی است، اما فرکانس ورودی بیرون از خانوادهٔ چهارمودی آموزش است. شبکهٔ ثابت شرط ورودی ۹نوسانی را اعمال نمی‌کند؛ این گزارش آزمون محدودیت مدل است و پشتیبانی از این خانواده را ادعا نمی‌کند.

## پیش‌بینی بدون آموزش مجدد

```bash
pip install -r requirements.txt
python -m channel_operator.v2_predict --checkpoint models/v2/stream_operator.pt \
  --re 55 --coefficients 0.25 -0.1 0.05 0 --output prediction.npz
```

یا به‌جای ضرایب از `--inlet-csv examples/inlet_example.csv` استفاده کنید. CSV سربرگ y,u و نقاط صعودی از y=0 تا 1 دارد؛ سرعت در دو انتها صفر است. میانگین سرعت نرمال می‌شود. انطباق با خانوادهٔ چهارمودی کنترل می‌شود و خطای انطباق بالای ۰٫۱٪ رد می‌شود. دامنهٔ ضرایب sum(abs(a))<=0.8 و بازهٔ Re=[10,100] است.

کانال 0<=x/H<=5 و 0<=y/H<=1 است. Re=U_mean*H/nu و p*=(p-p_ref)/(rho*U_mean^2). بنابراین u_phys=U_mean*u و p_phys=p_ref+rho*U_mean^2*p. برای تغییر اندازهٔ فیزیکی سرعت، Re را بر پایهٔ میانگین همان ورودی تنظیم کنید.

u=psi_y و v=-psi_x هستند؛ مشتق مکانی برای inference لازم است، اما وزن‌ها تغییر نمی‌کنند و CFD در زمان پیش‌بینی اجرا نمی‌شود. برنامه علاوه بر میدان NPZ، فایل JSON تشخیصی می‌نویسد. هشدار معیار مومنتومِ پاس‌نشده عمدی است.

## تکرار ارزیابی مدل ثابت

```bash
python scripts/evaluate_v2.py --checkpoint models/v2/stream_operator.pt \
  --data data/v2/monitor.npz --training-data data/v2/train.npz \
  --output artifacts/v2-monitor --role repeated_monitor_validation
python scripts/evaluate_v2.py --checkpoint models/v2/stream_operator.pt \
  --data data/v2/blind.npz --training-data data/v2/train.npz \
  --output artifacts/v2-blind --role untouched_final_blind_test
```

استفاده از دادهٔ blind برای تنظیم مدل بعدی، آن را از آزمون کور به مجموعهٔ توسعه تبدیل می‌کند؛ برای ارزیابی نسخه‌های بعدی باید نمونهٔ مستقل تازه ساخت.

## تکرار آموزش

برای CPU، `pip install -r requirements-jax.txt` کافی است. برای GPU در Kaggle از JAX/CUDA موجود و تست‌شدهٔ محیط استفاده شده است؛ بستهٔ CPU را جایگزین پیکربندی GPU نکنید.

```bash
python run_v2_jax.py --data data/v2 --output artifacts/v2-main \
  --steps 50000 --batch 16 --points 96 --pressure-mode convective --relative-data
python run_v2_jax.py --data data/v2 --output artifacts/v2-refine \
  --steps 70000 --batch 16 --points 96 --pressure-mode convective --relative-data \
  --resume artifacts/v2-main/jax_checkpoint.npz --adaptive-physics --restart-lr 0.0003
```

هر ۲۵۰ گام خطای تک‌تک حالات خارج از آموزش و هر ۱۰۰۰ گام تشخیص فیزیکی چاپ می‌شود. برای ادامه از وزن منتشرشده، در فرمان دوم از `--resume models/v2/jax_checkpoint.npz` و تعداد گام هدف بالاتر از ۷۰۰۰۰ استفاده کنید.

نوت‌بوک نهایی خصوصی: https://www.kaggle.com/code/mohrezsayeh/channel-stream-jax-refine-v2 . checkpoint اصلی، optimizer، RNG، لاگ و متادیتای source در خروجی آن ذخیره شده‌اند. آزمایش‌های کنترل نیز در گزارش فهرست شده‌اند.
