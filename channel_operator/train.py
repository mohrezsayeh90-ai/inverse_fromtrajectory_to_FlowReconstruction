"""Train one operator for the entire inlet/Re family; never refit at inference."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .model import ChannelDeepONet, residuals, outlet_residuals
from .profiles import sample_profiles


def train(output, steps=12000, seed=42, batch=8, points=96, width=64, latent=32, physics_only=False, resume=None, entrance_features=False, data_weight=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChannelDeepONet(width=width, latent=latent, entrance_features=entrance_features).to(device)
    data_weight = (200 if entrance_features else 20) if data_weight is None else data_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history, first_step = [], 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model.config:
            raise ValueError("Resume model architecture mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step = checkpoint["step"]
        if steps <= first_step:
            raise ValueError(f"Target steps must exceed resumed checkpoint step {first_step}")
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    data = None
    if not physics_only:
        with np.load(output/"train.npz") as d:
            data = {k: tensor(d[k]) for k in ["profiles", "reynolds", "xy", "fields"]}
    start = time.time()
    log = open(output/"training.jsonl", "a", buffering=1)

    def save(step):
        state = {"state_dict": model.state_dict(), "model_config": model.config, "optimizer": optimizer.state_dict(),
                 "step": step, "seed": seed, "mode": "physics_only" if physics_only else "hybrid_physics_informed",
                 "training_domain": {"length": 5, "height": 1, "reynolds": [10, 100], "cosine_modes": 4, "coefficient_l1_radius": .6},
                 "torch_version": torch.__version__}
        state["loss_config"] = dict(mass=10, momentum=1, outlet=.1, flux=10, data=data_weight)
        temporary = output/"checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(output/"model.pt")

    for step in range(first_step+1, steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = step/steps
        lr = 2e-4 + .5*(2e-3-2e-4)*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Every step samples NEW inlet/Re conditions, not just the FEM cases.
        profiles, _ = sample_profiles(rng, batch)
        profiles = tensor(profiles)
        re = tensor(np.exp(rng.uniform(np.log(10), np.log(100), batch)))
        xy = torch.rand(batch, points, 2, device=device)
        # Half of x locations focus on the developing entrance region.
        xy[..., 0] = 5*xy[..., 0]**2
        mass, mx, my = residuals(model, profiles, re, xy)
        mass_loss = mass.square().mean()
        momentum = mx.square().mean()+my.square().mean()
        outlet = torch.rand(batch, 32, 2, device=device)
        outlet[..., 0] = 5
        on, ot = outlet_residuals(model, profiles, re, outlet)
        bc_loss = on.square().mean()+ot.square().mean()
        # Integrated mass balance on randomly selected cross-sections.
        gx = torch.rand(batch, 3, 1, device=device)*5
        gy = torch.linspace(0, 1, 49, device=device)
        grid = torch.stack((gx.expand(-1, -1, len(gy)), gy[None, None, :].expand(batch, 3, -1)), -1).reshape(batch, -1, 2)
        u = model(profiles, re, grid)[..., 0].reshape(batch, 3, -1)
        flux_loss = (torch.trapezoid(u, gy, dim=-1)-1).square().mean()
        supervised = tensor(0.)
        if data is not None:
            ci = torch.randint(len(data["profiles"]), (batch,), device=device)
            pi = torch.randint(len(data["xy"]), (192,), device=device)
            prediction = model(data["profiles"][ci], data["reynolds"][ci], data["xy"][pi][None].expand(batch, -1, -1))
            target = data["fields"][ci[:, None], pi[None, :]]
            scale = torch.ones(batch, 1, 3, device=device)
            scale[..., 2] = data["reynolds"][ci, None]/60
            supervised = ((prediction-target)*scale).square().mean()
        # FEM warm-start remains part of the loss, physics is active from step 1.
        loss = 10*mass_loss + momentum + .1*bc_loss + 10*flux_loss + data_weight*supervised
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            record = {"step": step, "loss": float(loss.detach()), "divergence_mse": float(mass_loss.detach()),
                      "momentum_mse": float(momentum.detach()), "outlet_mse": float(bc_loss.detach()),
                      "flux_mse": float(flux_loss.detach()), "supervised_mse": float(supervised.detach()),
                      "seconds": time.time()-start, "lr": lr}
            log.write(json.dumps(record)+"\n")
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % 1000 == 0 or step == steps:
            save(step)
    log.close()
    (output/"training_summary.json").write_text(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name() if device.type=="cuda" else None,
        "steps": steps, "seconds": time.time()-start, "mode": "physics_only" if physics_only else "hybrid_physics_informed", "last": history[-1] if history else None}, indent=2))
    return model
