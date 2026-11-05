"""Fresh test seed, independent of the four development cases; no training here."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from channel_operator.profiles import sample_profiles
from channel_operator.reference import solve_channel
from channel_operator.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory to preserve earlier evaluations")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, output/"model.pt")
    seed = 640529
    profiles, coefficients = sample_profiles(np.random.default_rng(seed), 8)
    res = np.linspace(12, 98, 8)
    x, y = np.meshgrid(np.linspace(.015, 4.985, 49), np.linspace(.01, .99, 21), indexing="ij")
    xy = np.column_stack([x.ravel(), y.ravel()])
    fields, mesh_differences = [], []
    info = []
    for i, (profile, re) in enumerate(zip(profiles, res)):
        fine, detail = solve_channel(profile, re, nx=64, ny=24)
        coarse, _ = solve_channel(profile, re, nx=40, ny=16)
        values, other = fine(xy), coarse(xy)
        fields.append(values)
        info.append(detail)
        mesh_differences.append({"velocity_relative_l2": float(np.linalg.norm(other[:, :2]-values[:, :2])/np.linalg.norm(values[:, :2])),
                                 "pressure_relative_l2": float(np.linalg.norm(other[:, 2]-values[:, 2])/np.linalg.norm(values[:, 2]))})
        print(f"Fresh benchmark FEM {i+1}/8 Re={re:.2f}", flush=True)
    np.savez_compressed(output/"test.npz", profiles=profiles, coefficients=coefficients, reynolds=res, xy=xy, fields=np.array(fields))
    parabolic = 6*np.linspace(0, 1, 33)*(1-np.linspace(0, 1, 33))
    exact_solver, _ = solve_channel(parabolic, 40., nx=20, ny=8)
    exact = np.column_stack([6*xy[:, 1]*(1-xy[:, 1]), np.zeros(len(xy)), 12/40*(5-xy[:, 0])])
    (output/"reference_metadata.json").write_text(json.dumps({"seed": seed, "test_used_for_training": False,
        "test_mesh_comparison": mesh_differences, "reference_cases": info,
        "poiseuille_max_abs_error": float(np.abs(exact_solver(xy)-exact).max())}, indent=2))
    evaluate(output)


if __name__ == "__main__":
    main()
