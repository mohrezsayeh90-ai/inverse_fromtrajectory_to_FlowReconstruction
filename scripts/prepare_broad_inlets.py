"""Build input-only train/test banks from the official paper's GRF approach."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from channel_operator.broad_profiles import LENGTH_SCALES, SOURCE_URL, SOURCE_COMMIT, sample_inlets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/broad_inlets')
    parser.add_argument('--train-count', type=int, default=10000)
    parser.add_argument('--test-count', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    banks, summary = {}, {}
    for name, count, seed in [('train', args.train_count, 9703101), ('heldout', args.test_count, 9703199)]:
        bank = sample_inlets(count, seed)
        np.savez_compressed(args.output/f'{name}.npz', **bank)
        y, profiles = bank['y'], bank['profiles']
        peaks = np.array([len(find_peaks(p, prominence=.01)[0]) for p in profiles])
        assert np.all(profiles[:, [0, -1]] == 0) and profiles.min() >= 0
        assert profiles.max() <= 2.875
        assert np.max(abs(np.trapezoid(profiles.astype(float), y, axis=1)-1)) < 1e-6
        hashes = {hashlib.sha256(p.tobytes()).hexdigest() for p in profiles}
        assert len(hashes) == count
        banks[name] = dict(bank=bank, peaks=peaks, hashes=hashes)
        unique, frequency = np.unique(peaks, return_counts=True)
        summary[name] = dict(count=count, unique_profiles=len(hashes), seed=seed,
            sensors=len(y), min_speed=float(profiles.min()), max_speed=float(profiles.max()),
            maximum_flux_quadrature_error=float(np.max(abs(np.trapezoid(profiles.astype(float), y, axis=1)-1))),
            peak_count_range=[int(peaks.min()), int(peaks.max())],
            peak_histogram=dict(zip(map(str, unique), map(int, frequency))),
            profiles_with_at_least_nine_peaks=int((peaks >= 9).sum()),
            rejected_speed_draws=int(bank['rejected_speed_draws']))
        print(name, json.dumps(summary[name]), flush=True)
    assert not banks['train']['hashes'] & banks['heldout']['hashes']
    metadata = dict(source=SOURCE_URL, source_commit=SOURCE_COMMIT,
        official_method='RBF Gaussian random field; Cholesky sample on 512 points; linear sensor interpolation',
        adaptations=['NumPy implementation', 'six length scales instead of one',
                     'positive channel envelope 6*y*(1-y)*(1+a*tanh(g))',
                     'unit mean by trapezoid quadrature', 'reject draws with max speed >2.875',
                     '257 input sensors', 'Re log-uniform in [10,100]'],
        length_scales=list(LENGTH_SCALES), peak_definition='Interior maxima with prominence >=0.01 U_mean',
        response_labels_present=False, cfd_calls=0, model_training_performed=False,
        compatible_with_existing_v2=False, heldout_exact_profile_overlap=0, splits=summary)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'text.color':'#142b49', 'axes.labelcolor':'#142b49'})
    train = banks['train']['bank']; profiles, y = train['profiles'], train['y']
    fig, axes = plt.subplots(2,3, figsize=(15,9), layout='constrained')
    for ax, scale in zip(axes.flat, LENGTH_SCALES):
        indices = np.flatnonzero(train['length_scales']==scale)[:6]
        for i, color in zip(indices, plt.cm.viridis(np.linspace(.05,.9,len(indices)))):
            ax.plot(y, profiles[i], color=color, lw=1.4,
                    label=f"{banks['train']['peaks'][i]} peaks")
        ax.set(title=f'GRF length scale = {scale:g}', xlabel='y / H', ylabel='u_in / U_mean',
               xlim=(0,1), ylim=(0,2.9))
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('PAPER-BASED GAUSSIAN-FIELD INLET GENERATOR\n'
                 'Input-only bank: 10,000 train + 1,000 held-out | no CFD labels | not yet used to train an operator',
                 fontsize=16,fontweight='bold')
    fig.savefig(args.output/'examples.png',dpi=150)
    fig.savefig(args.output/'examples.pdf');plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,4.5),layout='constrained')
    for name, color in [('train','#008b8b'),('heldout','#e3584f')]:
        counts = banks[name]['peaks']
        bins=np.arange(max(banks['train']['peaks'].max(),banks['heldout']['peaks'].max())+2)-.5
        ax.hist(counts,bins=bins,weights=np.ones(len(counts))/len(counts),histtype='step',lw=2,
                color=color,label=f'{name}: {len(counts):,}')
    ax.set(xlabel='Number of interior peaks (prominence >= 0.01 U_mean)',ylabel='Fraction of profiles',
           title='Complexity coverage of generated inlet functions')
    ax.legend();ax.grid(alpha=.2);fig.savefig(args.output/'peak_distribution.png',dpi=150);plt.close(fig)
    text=f'''# مولد گستردهٔ ورودی بر اساس کد رسمی مقاله

**این فایل‌ها فقط ورودی‌اند؛ مدل جدیدی با آن‌ها آموزش ندیده است. وزن V2 از آن‌ها پشتیبانی نمی‌کند.**

مولد از [کد رسمی مقاله]({SOURCE_URL}) سازگار شده است: کرنل RBF، نمونه‌گیری میدان تصادفی گاوسی با Cholesky روی ۵۱۲ نقطه و نمونه‌برداری در حسگرها.
در notebook رسمی FF مقدار `length_scale=0.01` و در مثال ساده‌تر `0.2` است؛ این نسخه از شش مقیاس {LENGTH_SCALES} استفاده می‌کند.
تعداد نمونه‌های این بانک، انتخاب ماست؛ عدد پیش‌فرض notebook رسمی FF در نسخهٔ بررسی‌شده ۵۰۰۰ است و مقاله برای آزمایش مربوطه ۱۰هزار گزارش می‌کند.

برای سازگاری با کانال، میدان گاوسی g به `6*y*(1-y)*(1+a*tanh(g))` تبدیل و میانگین آن یک می‌شود؛ `a` بین ۰٫۱۵ و ۰٫۷۵ است.
این تبدیلِ نامنفی، صفرکردن سرعت روی دو دیواره، محدودیت سرعت و مخلوط‌کردن مقیاس‌ها، تغییرات ما نسبت به مولد مقاله‌اند.
هیچ تصویرکردن روی چهار مود یا بریدن نوسان‌ها انجام نمی‌شود. این بازتولید داده‌های عیناً یکسان مقاله نیست، بلکه استفاده از همان روش تولید است.

| مجموعه | تعداد شکل یکتا | حسگر | بازهٔ تعداد قله | تعداد با حداقل ۹ قله |
|---|---:|---:|---:|---:|
| آموزش پیشنهادی | {summary['train']['unique_profiles']} | 257 | {summary['train']['peak_count_range']} | {summary['train']['profiles_with_at_least_nine_peaks']} |
| آزمون کنارگذاشته‌شده | {summary['heldout']['unique_profiles']} | 257 | {summary['heldout']['peak_count_range']} | {summary['heldout']['profiles_with_at_least_nine_peaks']} |

قله با برجستگی حداقل ۰٫۰۱ سرعت میانگین شمرده می‌شود. هیچ پروفایل یکسانی بین دو مجموعه وجود ندارد؛ این بررسی به معنی جدایی توزیع‌ها نیست و هر دو از یک مولدند.
Re در بازهٔ ۱۰ تا ۱۰۰، سرعت نمونه‌ها نامنفی و حداکثر ۲٫۸۷۵، و میانگین ذوزنقه‌ای روی حسگرها برابر یک است.
این حدود روی نمونه‌های ذخیره‌شده کنترل شده‌اند؛ مدل آینده باید نمایش پیوسته و دقت حسگرهای آن را جداگانه بررسی کند.

![نمونه‌های مولد در مقیاس‌های مختلف](examples.png)

![توزیع پیچیدگی شکل‌ها](peak_distribution.png)

- [ورودی‌های آموزش پیشنهادی](train.npz)
- [ورودی‌های آزمون مستقل](heldout.npz)
- [متادیتا و شمارش‌ها](metadata.json)
- [PDF نمونه‌ها](examples.pdf)

NPZ فقط شامل `y, profiles, reynolds, length_scales, amplitudes, rejected_speed_draws` است؛ میدان سرعت/فشار حل‌شده در آن وجود ندارد.

```bash
python scripts/prepare_broad_inlets.py
```

این مرحله فقط آماده‌سازی ورودی برای آموزش آیندهٔ یک اپراتور بدون برچسب CFD است؛ دربارهٔ همگرایی یا دقت چنین مدلی نتیجه‌ای ارائه نمی‌کند.
'''
    (args.output/'README.md').write_text(text)


if __name__=='__main__':
    main()
