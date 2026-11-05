"""Held-out FEM comparison plus fresh PDE, boundary and flux diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import load_model, residuals, outlet_residuals
from .profiles import from_coefficients, Y


def evaluate(output):
    output = Path(output)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(output/"model.pt", device)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    with np.load(output/"test.npz") as d:
        profiles, res, xy, truth = [d[k] for k in ["profiles", "reynolds", "xy", "fields"]]
    p, r = tensor(profiles), tensor(res)
    with torch.no_grad():
        prediction = model(p, r, tensor(xy)[None].expand(len(p), -1, -1)).cpu().numpy()
    cases = []
    rng = np.random.default_rng(84853)
    for i in range(len(p)):
        pp, rr = p[i:i+1], r[i:i+1]
        coords = rng.uniform([.005, .005], [4.995, .995], (1, 2048, 2))
        m, mx, my = residuals(model, pp, rr, tensor(coords))
        gx, gy = np.meshgrid(np.linspace(0, 5, 31), np.linspace(0, 1, 257), indexing="ij")
        grid = np.stack([gx, gy], -1)
        with torch.no_grad():
            uvp = model(pp, rr, tensor(grid.reshape(1, -1, 2))).cpu().numpy().reshape(31, 257, 3)
        flux = np.trapezoid(uvp[..., 0], gy, axis=-1)
        outlet_xy = np.stack([np.full(257, 5.), np.linspace(0, 1, 257)], -1)
        on, ot = outlet_residuals(model, pp, rr, tensor(outlet_xy[None]))
        error = prediction[i]-truth[i]
        record = {"case": i, "reynolds": float(res[i]),
                  "velocity_relative_l2": float(np.linalg.norm(error[:, :2])/np.linalg.norm(truth[i, :, :2])),
                  "pressure_relative_l2": float(np.linalg.norm(error[:, 2])/np.linalg.norm(truth[i, :, 2])),
                  "v_absolute_rmse": float(np.sqrt(np.mean(error[:, 1]**2))),
                  "divergence_rms": float(m.square().mean().sqrt().detach()),
                  "momentum_rms": float((mx.square().mean()+my.square().mean()).sqrt().detach()),
                  "max_flux_error": float(np.max(np.abs(flux-1))),
                  "scaled_outlet_rms": float((on.square().mean()+ot.square().mean()).sqrt().detach())}
        cases.append(record)
    meta = json.loads((output/"reference_metadata.json").read_text())
    # Deliberately explicit acceptance gates; a completed training is not a pass.
    criteria = {"velocity_relative_l2_max": .05, "pressure_relative_l2_max": .10,
                "max_flux_error": .02, "divergence_rms_max": .03,
                "momentum_rms_max": .05, "scaled_outlet_rms_max": .10,
                "reference_mesh_velocity_max": .01, "reference_mesh_pressure_max": .02}
    gates = {"velocity": max(c["velocity_relative_l2"] for c in cases)<.05,
             "pressure": max(c["pressure_relative_l2"] for c in cases)<.10,
             "flux": max(c["max_flux_error"] for c in cases)<.02,
             "divergence": max(c["divergence_rms"] for c in cases)<.03,
             "momentum": max(c["momentum_rms"] for c in cases)<.05,
             "outlet": max(c["scaled_outlet_rms"] for c in cases)<.10,
             "reference_mesh": all(c["velocity_relative_l2"]<.01 and c["pressure_relative_l2"]<.02 for c in meta["test_mesh_comparison"]),
             "reference_poiseuille": meta["poiseuille_max_abs_error"]<1e-7}
    report = {"checkpoint_step": ckpt["step"], "training_mode": ckpt["mode"], "test_cases": cases,
              "acceptance_criteria": criteria, "gates": gates, "passed": all(gates.values()),
              "reference_mesh_comparison": meta["test_mesh_comparison"],
              "scope": "Held-out smooth positive no-slip inlets within the four-mode L1<=0.6 family, Re in [10,100]; no extrapolation guarantee."}
    (output/"metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/"heldout_predictions.npz", xy=xy, profiles=profiles, reynolds=res, reference=truth, prediction=prediction)
    i = int(np.argmax([c["velocity_relative_l2"] for c in cases]))
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), constrained_layout=True)
    for j, name in enumerate(["u", "v", "p"]):
        for k, (title, field) in enumerate([("FEM", truth), ("operator", prediction), ("error", prediction-truth)]):
            ax = axes[j, k]
            z = field[i, :, j].reshape(49, 21).T
            im = ax.imshow(z, origin="lower", extent=[.015, 4.985, .01, .99], aspect="auto", cmap="coolwarm")
            ax.set_title(f"{name}: {title}")
            fig.colorbar(im, ax=ax)
    fig.suptitle(f"Worst held-out velocity case: Re={res[i]:.1f}")
    fig.savefig(output/"heldout_fields.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    return report
