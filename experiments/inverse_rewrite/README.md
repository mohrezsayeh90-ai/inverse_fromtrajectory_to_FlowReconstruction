# Inverse redesign experiment

The frozen SOAP 314k checkpoint is checked against its published SHA-256. The fluid references are the saved FEM fields from the prior particle count experiments. Tracks include coordinate noise with standard deviation 0.0002 in channel height units. Training, validation, and final replay test tracks use separate seeds. All fitting and selection use observations and known initial conditions. Relative L2 inlet errors are calculated after selection.

The recorded design uses the old noisy observations. Two independently seeded runs compare the randomized and mixed designs. Both share their first two random launches. The random pool samples position, speed, and angle and retains trajectories with a full observation horizon. The mixed pool alternates two stratified near-inlet launches with one additional random launch. The near-inlet trajectories use a random streamwise position, lower speed, random transverse velocity, and a randomly shifted low-discrepancy height sequence. The common geometric count schedule is 2, 4, 8, 16, 32, 48. Every displayed count is refit.

The full-channel inverse represents log inlet shape with cosine coefficients and a cubic spline on 65 knots. It compares coarse and fine fits, smoothing strengths and near-inlet residual weights, and the previous count's solution evaluated against the enlarged data. The inverse cost omits the learned field's PDE residual. The coarse solution initializes the fine solution. An integral-position score on 16 disjoint validation trajectories ranks the candidates. That score uses observed paths as integration coordinates. Saved replays at the final count integrate the fitted ODE from the initial conditions for both the training paths and eight additional test paths.

The boundary estimate uses early portions of near-inlet tracks. It integrates the particle equation with a nonnegative spline for inlet speed and an optional quadratic streamwise correction. Candidate resolutions, training regions and roughness penalties are ranked on the same local region of independent validation tracks. For the nine-cycle case, an extended run adds 64 near-inlet observations to the original pool of 32. Full-domain replay with the frozen SOAP forward operator is a separate diagnostic.

From Main_Proposal, use the Python environment at /workspace/setup/adaptive-env/bin/python and the FLOW checkout or export at revision f6b8690059716e809c2dac27c8715e65fcb1d267. The individual scripts accept --help.

```sh
export REWRITE_PYTHON=/workspace/setup/adaptive-env/bin/python
export REWRITE_FLOW=/workspace/revision-sources/flow-soap
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/diagnose_old.py --flow "$REWRITE_FLOW"
profiles=(beta5 skew_b3 flat30 twin_b3_r30 shoulder mixed waves3_r30 lobes4_r30 nine30)
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/run.py --flow "$REWRITE_FLOW" "${profiles[@]}"
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/run.py --flow "$REWRITE_FLOW" --seed 53129 "${profiles[@]}" --designs random hybrid
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/run_local.py --flow "$REWRITE_FLOW" --extend nine30
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/run_local.py --flow "$REWRITE_FLOW" --seed 53129 --extend nine30
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/deliver.py
for seed in 39421 53129; do
  "$REWRITE_PYTHON" -B experiments/inverse_rewrite/run_local.py --flow "$REWRITE_FLOW" --seed "$seed" --extend --fine-grid nine30
done
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/report.py --thesis
. /workspace/setup/activate.sh
latexmk -e '$max_repeat=12;' -xelatex -outdir=build/final -interaction=nonstopmode -halt-on-error -file-line-error thesis.tex
cp build/final/thesis.pdf thesis.pdf
"$REWRITE_PYTHON" -B experiments/inverse_rewrite/report.py
```

Workers can take disjoint lists of profiles and be pinned with --cpu. Intermediate reports are reusable. Data and figures are under New_Adaptive_Particle_Results/inverse_rewrite.
