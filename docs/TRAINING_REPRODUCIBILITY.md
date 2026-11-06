# Continuing the independent-pressure experiment

The 5,000-step pilot is a candidate experiment. Its validation does not certify the requested general operator. The longer run keeps the same model configuration, input pool, Adam state, and optimizer RNG; it tests whether additional training improves the measured errors.

```bash
python -m scripts.make_kaggle \
  --output notebooks/training/pressure_long \
  --id mohrezsayeh/channel-general-physics-v3-pressure-long \
  --steps 100000 \
  --resume-source mohrezsayeh/channel-general-physics-v3-pressure \
  --include-references --include-heldout --final-evaluation \
  --reuse-validation-from-resume \
  --encoder-kind sensor --lift oseen \
  --physics-normalization input_curvature_sqrt \
  --batch 64 --points 48 --width 128 --latent 64 --depth 4 \
  --separate-pressure-trunk --devices 2
kaggle kernels push -p notebooks/training/pressure_long --accelerator NvidiaTeslaT4
```

Target steps are global: this command resumes step 5,000 and trains through step 100,000. The source snapshot and SHA256 manifest are committed beside the notebook. The input pool has 10,000 input functions and no solved response fields. The runtime regenerates it from the recorded seed and refuses a resume if its hash changes.

Kaggle limits notebook source to 1 MB. The four validation-reference files are reused from the attached completed pilot, with SHA256 checks before loading. Only the two reserved references are embedded in the new notebook. `output/reference_manifest.json` records every reference hash and origin. No precision reduction or numerical-reference change is used to fit the size limit.

Each 250 steps prints unscaled physical diagnostics. Each 1,000 steps also prints the four numerical-reference validation errors. The current physical monitor uses 1,024 points per diagnostic profile, independently of optimizer sampling. The checkpoint records its definition. If this definition differs from the parent run, the old selection score is not compared directly with the new scores.

The reference solutions are excluded from the objective and residual scales. They are used for repeated validation and checkpoint selection. The two reserved 513-knot test profiles are excluded from that monitor. `scripts/finalize_run.py` evaluates the last and best saved checkpoints, selects using validation, and runs the reserved test only if validation passes. Its `output/acceptance_status.json` distinguishes training completion, validation acceptance, and readiness after both suites.

The source files do not contain Kaggle or GitHub credentials. The notebook is private, requires GPUs, and uses Kaggle's preinstalled scientific libraries without internet access. A two-device parity check runs before training. Saved weights have no device axis and can be evaluated on a single CPU or GPU.

Outputs to retain after the run:

- `checkpoint.npz`, `best_checkpoint.npz`, `selected_checkpoint.npz` and matching validation reports.
- `training.jsonl`, `summary.json`, `input_pool_metadata.json`, `source_manifest.json`.
- `acceptance_status.json`, `validation_checkpoint/`, `validation_best_checkpoint/`, and `heldout_test/` if actually evaluated.

No outcome is asserted in advance. The inference API rejects a checkpoint without a passing report that matches its weight hash and implementation signature, unless provisional use is explicitly requested.

## Saved-run reproduction without relying on Kaggle storage

The completed last, best, and selected weights, including Adam and RNG states, are all in `reports/experiments/pressure_long/output/`. The 5,000-step parent is also retained at `models/candidates/pressure/checkpoint.npz`. Sources and seeds are retained in the run snapshot. To reproduce the continuation on a suitable two-GPU machine from repository files:

```bash
python -m flow_operator.train --output runs/reproduced_pressure \
  --resume models/candidates/pressure/checkpoint.npz --steps 100000 \
  --pool-count 10000 --pool-seed 9703107 --batch 64 --points 48 \
  --width 128 --latent 64 --depth 4 --encoder-kind sensor --lift oseen \
  --physics-normalization input_curvature_sqrt --separate-pressure-trunk \
  --devices 2 --monitor-points 1024 \
  --reference-monitor validation/references --reference-every 1000
```

Input functions are regenerated from the saved seed; resume checks their hash before updating weights. A changed software environment that changes that hash is rejected. The pinned tested CPU package versions are in `requirements-cpu-tested.txt`; keep an appropriate CUDA JAX installation for GPU training. Bitwise identical retraining across hardware, Python, and library versions is not promised. Frozen-weight evaluation is independently replayed in the saved reports. `python -m scripts.verify_reproducibility` checks the actual repository artifacts rather than assuming their presence.
