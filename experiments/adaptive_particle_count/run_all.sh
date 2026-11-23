#!/usr/bin/env bash
# Run from Main_Proposal. Existing per-profile locks prevent duplicate writes.
set -euo pipefail
adaptive_python=${ADAPTIVE_PYTHON:-/workspace/setup/adaptive-env/bin/python}
flow_checkout=${1:-/workspace/revision-sources/flow}
mapfile -t adaptive_cpus < <("$adaptive_python" -c 'import os; print("\n".join(map(str,sorted(os.sched_getaffinity(0))[:5])))')
adaptive_pids=()
for adaptive_cpu in "${adaptive_cpus[@]}"; do
 (
  for profile in beta5 skew_b3 flat30 twin_b3_r30 shoulder mixed waves3_r30 lobes4_r30 nine30; do
   "$adaptive_python" experiments/adaptive_particle_count/run.py --flow "$flow_checkout" --profile "$profile" --cpu "$adaptive_cpu"
  done
 ) > "/tmp/adaptive-worker-${adaptive_cpu}.log" 2>&1 &
 adaptive_pids+=("$!")
done
adaptive_status=0
for adaptive_pid in "${adaptive_pids[@]}"; do wait "$adaptive_pid" || adaptive_status=1; done
if [ "$adaptive_status" -ne 0 ]; then exit "$adaptive_status"; fi
"$adaptive_python" experiments/adaptive_particle_count/summarize.py
"$adaptive_python" experiments/adaptive_particle_count/audit.py
adaptive_hard_profile=$("$adaptive_python" -c 'import csv; from pathlib import Path; rows=list(csv.DictReader(Path("New_Adaptive_Particle_Results/continuation/summary.csv").open())); r=max(rows,key=lambda r:float(r["final_error_percent"])); print(r["profile"] if float(r["final_error_percent"])>5 else "")')
if [ -n "$adaptive_hard_profile" ]; then
 "$adaptive_python" experiments/adaptive_particle_count/run.py --flow "$flow_checkout" --profile "$adaptive_hard_profile" --variant near_inlet --smoothness 1e-8 --physics-weight 0 --launch-design near-inlet --max-iterations 400
 ADAPTIVE_VARIANT=near_inlet "$adaptive_python" experiments/adaptive_particle_count/summarize.py
 "$adaptive_python" experiments/adaptive_particle_count/audit.py --variant near_inlet --expected-profiles 1
fi
