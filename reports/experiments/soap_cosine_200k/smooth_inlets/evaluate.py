"""Evaluate a fixed set of smooth inlets using independent refined FEM fields."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from flow_operator.checkpoint import load
from flow_operator.evaluate import evaluate_fields
from flow_operator.profiles import Inlet
from flow_operator.validation import read_cases
from validation.fem import solve_reference


def parabola(y):
    return 6*y*(1-y)


SUITES = {
    "initial": (
        ("parabolic_re20", 20., "6*y*(1-y)", parabola),
        ("parabolic_re80", 80., "6*y*(1-y)", parabola),
        ("slightly_skewed_re40", 40., "6*y*(1-y)*(1+0.05*(2*y-1))", lambda y: parabola(y) * (1 + .05*(2*y-1))),
        ("broad_symmetric_re60", 60., "6*y*(1-y)*(1+0.05*(2*y-1)**2)", lambda y: parabola(y) * (1 + .05*(2*y-1)**2)),
    ),
    "extended": (
        ("sine_re35", 35., "sin(pi*y)", lambda y: np.sin(np.pi*y)),
        ("broad_cosine_re50", 50., "6*y*(1-y)*(1+0.12*cos(2*pi*(y-0.5)))", lambda y: parabola(y)*(1 + .12*np.cos(2*np.pi*(y-.5)))),
        ("rounded_shoulder_re65", 65., "1-(2*y-1)^4", lambda y: 4*y*(1-y)*(1+(2*y-1)**2)),
        ("gentle_off_center_re75", 75., "6*y*(1-y)*(1+0.16*exp(-((y-0.65)/0.24)^2))", lambda y: parabola(y)*(1 + .16*np.exp(-((y-.65)/.24)**2))),
    ),
    "diverse": (
        ("half_cosine_re15", 15., "sin(pi*y)^2", lambda y: np.sin(np.pi*y)**2),
        ("wide_sextic_re30", 30., "1-(2*y-1)^6", lambda y: 1-(2*y-1)**6),
        ("smooth_tilt_re45", 45., "6*y*(1-y)*(1+0.15*tanh((y-0.5)/0.22))",
         lambda y: parabola(y)*(1+.15*np.tanh((y-.5)/.22))),
        ("center_bell_re55", 55., "6*y*(1-y)*(1+0.3*exp(-((y-0.5)/0.22)^2))",
         lambda y: parabola(y)*(1+.3*np.exp(-((y-.5)/.22)**2))),
        ("shallow_center_dip_re70", 70., "6*y*(1-y)*(1-0.22*exp(-((y-0.5)/0.18)^2))",
         lambda y: parabola(y)*(1-.22*np.exp(-((y-.5)/.18)**2))),
        ("broad_skew_re90", 90., "(1-(2*y-1)^4)*(1-0.12*(2*y-1))",
         lambda y: (1-(2*y-1)**4)*(1-.12*(2*y-1))),
    ),
}


def prepare_references(directory, definitions):
    directory.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 1, 257)
    xx, yy = np.meshgrid(np.r_[np.linspace(.002,.3,25), np.linspace(.4,4.99,48)],
                         np.linspace(.002,.998,97), indexing="ij")
    xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    for name, reynolds, formula, profile in definitions:
        path = directory / f"{name}.npz"
        values = profile(y)
        if (not np.all(np.isfinite(values)) or np.min(values) < -1e-12
                or not np.allclose(values[[0, -1]], 0., atol=1e-12)):
            raise ValueError(f"Invalid no-slip inlet: {name}")
        values[[0, -1]] = 0
        inlet = Inlet.from_samples(y, values)
        if path.exists():
            with np.load(path, allow_pickle=False) as reference:
                if (float(reference["reynolds"]) != reynolds
                        or not np.array_equal(reference["inlet_y"], y)
                        or not np.allclose(reference["inlet_values"], inlet.values, rtol=0, atol=1e-14)
                        or not np.array_equal(reference["xy"], xy)):
                    raise ValueError(f"Stored reference definition differs: {path}")
            continue
        predictions, meshes = [], []
        for nx, ny in [(32, 24), (48, 36)]:
            print(f"FEM {name} on {nx}x{ny}", flush=True)
            field, stats = solve_reference(inlet, reynolds, nx, ny)
            predictions.append(np.concatenate([field(block) for block in np.array_split(xy, 14)]))
            meshes.append(stats)
        coarse, fine = predictions
        velocity = np.linalg.norm((coarse-fine)[:,:2]) / np.linalg.norm(fine[:,:2])
        pressure = np.linalg.norm((coarse-fine)[:,2]) / np.linalg.norm(fine[:,2])
        if velocity > .01 or pressure > .02:
            raise RuntimeError(f"FEM mesh refinement failed for {name}: {velocity}, {pressure}")
        metadata = dict(
            role="smooth_inlet_diagnostic", unnormalized_profile=formula,
            meshes=meshes, mesh_change=dict(velocity_relative_l2=velocity, pressure_relative_l2=pressure),
            pressure_alignment=False,
        )
        np.savez_compressed(path, inlet_y=y, inlet_values=inlet.values, reynolds=reynolds,
                            xy=xy, reference=fine, metadata=json.dumps(metadata))
        print(f"Mesh change for {name}: velocity {velocity:.3%}, pressure {pressure:.3%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acceptance", type=Path, default=Path("configs/acceptance.json"))
    parser.add_argument("--suite", choices=SUITES, default="diverse")
    args = parser.parse_args()
    definitions = SUITES[args.suite]
    references = args.output / "references"
    prepare_references(references, definitions)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    params, config, metadata, _ = load(args.checkpoint)
    threshold_config = json.loads(args.acceptance.read_text())
    thresholds = {k: v for k, v in threshold_config.items() if isinstance(v, (int, float))}
    cases = read_cases(references)
    if {case["name"] for case in cases} != {item[0] for item in definitions}:
        raise ValueError("Unexpected cases in the reference directory")
    with jax.enable_x64():
        double = jax.tree.map(lambda x: x.astype(np.float64), params)
        results = evaluate_fields(double, config, cases, thresholds)
    for row, case in zip(results["cases"], cases, strict=True):
        row["gates"] = {key: row[key] < limit for key, limit in thresholds.items()}
        row["reference_definition"] = case["metadata"]
    results.update(checkpoint_sha256=checkpoint_hash, nominal_step=metadata["step"],
                   updates=metadata["cosine_updates"], role="smooth_inlet_diagnostic", suite=args.suite,
                   cfd_training_labels=metadata["CFD_response_labels_used"])
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
