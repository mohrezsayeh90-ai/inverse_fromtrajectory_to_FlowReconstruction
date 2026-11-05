"""Generate independent converged FEM labels and a fixed held-out split."""
import json
from pathlib import Path
import numpy as np
from .profiles import sample_profiles, from_coefficients
from .reference import solve_channel


def generate(output, train_cases=48, test_cases=8, seed=1731):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train_profiles, train_coeff = sample_profiles(rng, train_cases)
    train_re = np.exp(rng.uniform(np.log(10), np.log(100), train_cases))
    train_re[:2] = [10, 100]
    # Disjoint deterministic test conditions, never used for model selection.
    test_profiles, test_coeff = sample_profiles(np.random.default_rng(seed+999), test_cases)
    test_re = np.linspace(12, 98, test_cases)
    x, y = np.meshgrid(np.linspace(.015, 4.985, 49), np.linspace(.01, .99, 21), indexing="ij")
    xy = np.column_stack([x.ravel(), y.ravel()])
    metadata = {"seed": seed, "train_cases": train_cases, "test_cases": test_cases, "test_used_for_training": False, "cases": []}
    for name, profiles, coeff, res in [("train", train_profiles, train_coeff, train_re), ("test", test_profiles, test_coeff, test_re)]:
        fields, mesh_differences = [], []
        for i, (profile, re) in enumerate(zip(profiles, res)):
            evaluate, info = solve_channel(profile, re, nx=40 if name=="train" else 64, ny=16 if name=="train" else 24)
            values = evaluate(xy)
            fields.append(values)
            if name == "test":
                coarse, _ = solve_channel(profile, re, nx=40, ny=16)
                other = coarse(xy)
                mesh_differences.append({
                    "velocity_relative_l2": float(np.linalg.norm(other[:, :2]-values[:, :2])/np.linalg.norm(values[:, :2])),
                    "pressure_relative_l2": float(np.linalg.norm(other[:, 2]-values[:, 2])/np.linalg.norm(values[:, 2]))})
            metadata["cases"].append(dict(split=name, index=i, reynolds=float(re), **info))
            print(f"FEM {name} {i+1}/{len(res)} Re={re:.2f} iterations={info['iterations']}", flush=True)
        np.savez_compressed(output/f"{name}.npz", profiles=profiles, coefficients=coeff, reynolds=res, xy=xy, fields=np.array(fields))
        if name == "test":
            metadata["test_mesh_comparison"] = mesh_differences
    # Exact solution verifies FEM formulation and pressure reference independently.
    re = 40.
    exact = np.column_stack([6*xy[:, 1]*(1-xy[:, 1]), np.zeros(len(xy)), 12/re*(5-xy[:, 0])])
    evaluate, info = solve_channel(from_coefficients(np.zeros(4)), re, nx=20, ny=8)
    metadata["poiseuille_max_abs_error"] = float(np.max(np.abs(evaluate(xy)-exact)))
    (output/"reference_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
