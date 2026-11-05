import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import load_model, residuals
from .profiles import from_coefficients, load_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--re", type=float, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coefficients", type=float, nargs=4)
    group.add_argument("--inlet-csv")
    parser.add_argument("--output", default="prediction.npz")
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=81)
    args = parser.parse_args()
    if not 10 <= args.re <= 100:
        parser.error("Validated Reynolds interval is [10,100]")
    profile = load_csv(args.inlet_csv) if args.inlet_csv else from_coefficients(args.coefficients)
    if args.coefficients and sum(abs(x) for x in args.coefficients) > .6:
        print("WARNING: profile is outside the training coefficient domain")
    if args.inlet_csv:
        print("WARNING: custom profiles may be outside the training family; validate against CFD")
    model, checkpoint = load_model(args.checkpoint)
    metrics_path = Path(args.checkpoint).parent/"metrics.json"
    validation = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    validation_current = bool(validation and validation.get("checkpoint_step") == checkpoint["step"])
    if not validation_current or not validation.get("passed"):
        print("WARNING: this checkpoint has not passed all documented validation gates")
    x, y = np.meshgrid(np.linspace(0, 5, args.nx), np.linspace(0, 1, args.ny), indexing="ij")
    xy = np.stack([x, y], -1)
    with torch.no_grad():
        values = model(torch.tensor(profile[None], dtype=torch.float32), torch.tensor([args.re]),
                       torch.tensor(xy.reshape(1, -1, 2), dtype=torch.float32)).numpy().reshape(args.nx, args.ny, 3)
    np.savez_compressed(args.output, x=x, y=y, u=values[..., 0], v=values[..., 1], p=values[..., 2], inlet=profile, reynolds=args.re)
    rng = np.random.default_rng(224)
    points = torch.tensor(rng.uniform([.005, .005], [4.995, .995], (1, 1024, 2)), dtype=torch.float32)
    mass, mx, my = residuals(model, torch.tensor(profile[None], dtype=torch.float32), torch.tensor([args.re]), points)
    flux = np.trapezoid(values[..., 0], y, axis=-1)
    diagnostics = {"output": args.output, "reynolds": args.re, "trained_step": checkpoint["step"], "retraining": False,
                   "checkpoint_passed_validation": bool(validation_current and validation.get("passed")),
                   "max_flux_error": float(np.max(np.abs(flux-1))),
                   "divergence_rms": float(mass.square().mean().sqrt().detach()),
                   "momentum_rms": float((mx.square().mean()+my.square().mean()).sqrt().detach()),
                   "note": "Residuals are diagnostics, not a substitute for independent CFD error estimates."}
    Path(args.output).with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
