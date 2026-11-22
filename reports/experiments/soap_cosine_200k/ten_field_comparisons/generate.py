"""Generate frozen SOAP predictions, FEM comparisons, and exportable field values."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

import jax
import numpy as np
from PIL import Image

from flow_operator.checkpoint import load
from flow_operator.evaluate import evaluate_fields

EVALUATION_ROOT = Path("reports/experiments/soap_cosine_200k")
INPUTS = (
    ("01_wall_soft_quartic", 12., "[4*y*(1-y)]^2", lambda y: (4*y*(1-y))**2),
    ("02_left_skewed_quadratic", 24., "6*y*(1-y)*(1-0.20*(2*y-1))",
     lambda y: 6*y*(1-y)*(1-.20*(2*y-1))),
    ("03_flat_core_octic", 32., "1-(2*y-1)^8", lambda y: 1-(2*y-1)**8),
    ("04_sine_blend", 39., "0.6*sin(pi*y)+0.4*sin(pi*y)^2",
     lambda y: .6*np.sin(np.pi*y)+.4*np.sin(np.pi*y)**2),
    ("05_right_broad_bump", 47., "6*y*(1-y)*(1+0.3*exp(-((y-0.68)/0.22)^2))",
     lambda y: 6*y*(1-y)*(1+.3*np.exp(-((y-.68)/.22)**2))),
    ("06_shallow_twin_shoulders", 53., "6*y*(1-y)*(1-0.28*exp(-((y-0.5)/0.20)^2))",
     lambda y: 6*y*(1-y)*(1-.28*np.exp(-((y-.5)/.20)**2))),
    ("07_smooth_left_bias", 61., "(1-(2*y-1)^4)*(1-0.18*sin(pi*(y-0.5)))",
     lambda y: (1-(2*y-1)**4)*(1-.18*np.sin(np.pi*(y-.5)))),
    ("08_broad_gaussian_core", 72., "6*y*(1-y)*exp(-((y-0.5)/0.45)^2)",
     lambda y: 6*y*(1-y)*np.exp(-((y-.5)/.45)**2)),
    ("09_gently_wavy", 83., "6*y*(1-y)*(1+0.075*cos(4*pi*y))",
     lambda y: 6*y*(1-y)*(1+.075*np.cos(4*np.pi*y))),
    ("10_tilted_flat_core", 95., "(1-(2*y-1)^6)*(1+0.16*(2*y-1))",
     lambda y: (1-(2*y-1)**6)*(1+.16*(2*y-1))),
)


def export_field_tables(fields, cases):
    for case in cases:
        name = case["name"]
        with np.load(fields / f"{name}.npz", allow_pickle=False) as saved:
            prediction, reference = saved["prediction"], saved["reference"]
            error = prediction-reference
            table = np.column_stack((saved["xy"], prediction, reference, error))
            if table.shape != (7081, 11) or not np.isfinite(table).all():
                raise ValueError(f"Invalid field values for {name}")
            np.savetxt(fields / f"{name}.csv", table, delimiter=",", fmt="%.15g", comments="",
                       header="x_over_H,y_over_H,u_model,v_model,p_model,u_FEM,v_FEM,p_FEM,u_error,v_error,p_error")
            case["field_point_count"] = len(table)
            case["component_rmse"] = dict(zip(("u", "v", "p"), np.sqrt(np.mean(error**2, axis=0)), strict=True))


def export_overview(output, cases):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), layout="constrained", sharex=True, sharey=True)
    names = {name: formula for name, _, formula, _ in INPUTS}
    for ax, case in zip(axes.flat, cases, strict=True):
        name = case["name"]
        with np.load(output/"references"/f"{name}.npz", allow_pickle=False) as data:
            ax.plot(data["inlet_y"], data["inlet_values"], color="#216477", linewidth=2)
        ax.set_title(name[3:].replace("_", " ").title(), fontsize=9)
        ax.text(.04, .9, f"Re={case['reynolds']:g}", transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=.2)
        case["inlet_formula"] = names[name]
    fig.supxlabel("Cross-channel position y/H")
    fig.supylabel("Inlet velocity / mean inlet velocity")
    fig.savefig(output/"inlet_profiles.png", dpi=160)
    plt.close(fig)
    images = []
    for path in [output/"inlet_profiles.png", *(output/"fields"/f"{c['name']}.png" for c in cases)]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    images[0].save(output/"field_comparisons.pdf", save_all=True, append_images=images[1:], resolution=155)
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=EVALUATION_ROOT/"final_checkpoint.npz")
    args = parser.parse_args()
    output = args.output
    fields = output/"fields"
    fields.mkdir(parents=True, exist_ok=True)
    reference_tools = runpy.run_path(str(EVALUATION_ROOT/"smooth_inlets/evaluate.py"))
    reference_tools["prepare_references"](output/"references", INPUTS)
    cases = reference_tools["read_cases"](output/"references")
    if {c["name"] for c in cases} != {n for n, *_ in INPUTS}:
        raise ValueError("The evaluation requires exactly the ten declared profiles")
    params, config, metadata, _ = load(args.checkpoint)
    thresholds = {k:v for k,v in json.loads(Path("configs/acceptance.json").read_text()).items()
                  if isinstance(v,(int,float))}
    with jax.enable_x64():
        params = jax.tree.map(lambda x: x.astype(np.float64), params)
        result = evaluate_fields(params, config, cases, thresholds, out=fields)
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    result.update(checkpoint_sha256=checkpoint_sha, nominal_step=metadata["step"],
                  cfd_training_labels=metadata["CFD_response_labels_used"],
                  pressure_convention="Same outlet convention as FEM. No fitted gauge shift.",
                  units="x/H, y/H, u/U_mean, v/U_mean, p/(rho*U_mean^2)")
    export_field_tables(fields, result["cases"])
    export_overview(output, result["cases"])
    for row, case in zip(result["cases"], cases, strict=True):
        row["reference_definition"] = case["metadata"]
        row["gates"] = {k: bool(row[k] < v) for k,v in thresholds.items()}
    (output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (output/"summary.csv").open("w", newline="") as stream:
        columns = ("case", "Re", "velocity_L2_percent", "pressure_L2_percent",
                   "u_RMSE", "v_RMSE", "p_RMSE", "momentum_RMS", "fields_pass")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in result["cases"]:
            writer.writerow([row["name"], row["reynolds"], 100*row["velocity_relative_l2"],
                             100*row["pressure_relative_l2"], *row["component_rmse"].values(),
                             row["momentum_rms"], row["gates"]["velocity_relative_l2"] and row["gates"]["pressure_relative_l2"]])


if __name__ == "__main__":
    main()
