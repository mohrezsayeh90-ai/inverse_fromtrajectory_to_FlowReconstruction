"""Fixed, previously unused coefficient-boundary cases. Does not retrain the model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from channel_operator.profiles import from_coefficients
from channel_operator.reference import solve_channel
from channel_operator.model import load_model, residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint)
    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    gx, gy = np.meshgrid(np.linspace(.01, 4.99, 49), np.linspace(.01, .99, 21), indexing="ij")
    xy = np.stack([gx, gy], -1).reshape(-1, 2)
    records = []
    for mode in range(4):
        for sign in [-1, 1]:
            a = np.zeros(4)
            a[mode] = sign*.6
            profile = from_coefficients(a)
            for re in [10., 100.]:
                reference, info = solve_channel(profile, re, nx=64, ny=24)
                truth = reference(xy)
                with torch.no_grad():
                    prediction = model(tensor(profile[None]), tensor([re]), tensor(xy[None])).numpy()[0]
                coords = np.random.default_rng(724).uniform([.005, .005], [4.995, .995], (1, 1024, 2))
                mass, mx, my = residuals(model, tensor(profile[None]), tensor([re]), tensor(coords))
                error = prediction-truth
                record = {"coefficients": a.tolist(), "reynolds": re,
                          "velocity_relative_l2": float(np.linalg.norm(error[:, :2])/np.linalg.norm(truth[:, :2])),
                          "pressure_relative_l2": float(np.linalg.norm(error[:, 2])/np.linalg.norm(truth[:, 2])),
                          "divergence_rms": float(mass.square().mean().sqrt().detach()),
                          "momentum_rms": float((mx.square().mean()+my.square().mean()).sqrt().detach()),
                          "reference": info}
                records.append(record)
                print(json.dumps(record), flush=True)
    report = {"checkpoint_step": checkpoint["step"], "note": "Fixed boundary-of-training-domain cases, evaluated without refitting. Reference is 64x24 P2/P1; see main report for mesh convergence checks.", "cases": records,
              "max_velocity_relative_l2": max(r["velocity_relative_l2"] for r in records),
              "max_pressure_relative_l2": max(r["pressure_relative_l2"] for r in records),
              "passed_velocity_pressure": all(r["velocity_relative_l2"]<.05 and r["pressure_relative_l2"]<.1 for r in records)}
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
