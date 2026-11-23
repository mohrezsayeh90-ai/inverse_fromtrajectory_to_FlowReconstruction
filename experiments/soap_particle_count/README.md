# Reproduce the adaptive particle study with SOAP 314k

Use an export or checkout of FLOW revision `f6b8690059716e809c2dac27c8715e65fcb1d267`. The final checkpoint is `reports/experiments/soap_cosine_200k/final_checkpoint.npz`. The runner checks its published SHA-256. The new study reads the existing 94k study's launch schedules, noisy tracks and archived FEM reference files. All fitted profiles, independent replays and plots are local to Main_Proposal.

Run from Main_Proposal using the Python 3.12 environment and dependency versions in `experiments/adaptive_particle_count/requirements.txt`.

```sh
export SOAP_FLOW=/workspace/revision-sources/flow-soap
export SOAP_PYTHON=/workspace/setup/adaptive-env/bin/python
"$SOAP_PYTHON" experiments/soap_particle_count/run.py --flow "$SOAP_FLOW" --parity
"$SOAP_PYTHON" experiments/soap_particle_count/check_forward.py --flow "$SOAP_FLOW"
"$SOAP_PYTHON" experiments/soap_particle_count/run.py --flow "$SOAP_FLOW" --variant soap_314k_near --near-inlet --parity nine30
"$SOAP_PYTHON" experiments/soap_particle_count/extend_near.py --flow "$SOAP_FLOW"
"$SOAP_PYTHON" experiments/soap_particle_count/run.py --flow "$SOAP_FLOW" --variant soap_314k_near_extended --near-inlet --extend-near nine30
"$SOAP_PYTHON" experiments/soap_particle_count/deliver.py
. /workspace/setup/activate.sh
latexmk -e '$max_repeat=12;' -xelatex -outdir=build/final -interaction=nonstopmode -halt-on-error -file-line-error thesis.tex
cp build/final/thesis.pdf thesis.pdf
"$SOAP_PYTHON" experiments/soap_particle_count/report.py
```

You can pass disjoint profile lists and separate available CPU IDs with `--cpu` to parallelize the primary study. A completed fit can be resumed using its saved report. Each adaptive stop considers the relative inlet error and whether each independent replay reached its end time. An early stop can be followed by display-only fits so the figure still shows five actual stages. These extra stages may cross back above the threshold.

The extended run preserves the first 24 observations from the previous focused design byte for byte. It samples the extra positions and velocities using seed 284914 and the same height sequence and time grid. Previously computed SOAP fits at 24 or fewer tracks can be reused because they use the identical prefixes and inverse settings. Comparisons to the 94k weight appear for counts available in both studies. Neither models' weights are trained in these experiments.
