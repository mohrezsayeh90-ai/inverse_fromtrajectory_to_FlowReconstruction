"""Cosine continuation with full-state snapshots and verified portable archives."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy

from .checkpoint import load
from .model import make_functions
from .soap_checkpoint import restore_state, save_state
from .train_soap import (
    SOAP_SETTINGS, finite, make_gradient, make_verified_pool, measure,
    fixed_physics, same_tree, sha, soap, tree_norm, write_json,
)
from .train import batch
from .validation import read_cases


def learning_rate(index, total):
    """First and last parameter updates attain the requested endpoints."""
    phase = jnp.minimum(jnp.asarray(index) / (total - 1), 1.)
    return 1e-5 + .5 * (1e-3 - 1e-5) * (1 + jnp.cos(jnp.pi * phase))


def make_update(functions, total):
    tx = soap(learning_rate=1., **SOAP_SETTINGS)
    gradient = make_gradient(functions, microbatches=4)

    @jax.jit
    def update(params, state, index, draw):
        (loss, terms), grad = gradient(params, draw)
        norm = tree_norm(grad)
        clipped = jax.tree.map(lambda g: g * jnp.minimum(1., 100 / (norm + 1e-12)), grad)
        updates, state = tx.update(clipped, state, params)
        lr = learning_rate(index, total)
        updates = jax.tree.map(lambda u: u * lr, updates)
        result = optax.apply_updates(params, updates)
        relative = tree_norm(updates) / jnp.maximum(tree_norm(params), 1e-30)
        return result, state, (loss, terms, norm, relative, lr), finite((result, state, grad))
    return update, tx


def package(root, checkpoint, index):
    target = root / f"resume_{index:06d}.zip"
    if target.exists():
        return
    pending = target.with_suffix(".tmp")
    files = [(checkpoint, "checkpoint.npz")]
    files += [(root / name, name) for name in
              ("config.json", "training.jsonl", "status.json", "source_bundle.zip", "RESUME.txt")]
    manifest = {}
    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path, name in files:
            manifest[name] = sha(path)
            archive.write(path, name)
        archive.writestr("SHA256.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(pending) as archive:
        for name, expected in manifest.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise RuntimeError(f"Archive validation failed for {name}")
    pending.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--references", default="validation/references", type=Path)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.references = args.references.resolve()
    if args.steps < 2 or not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("Use GPU JAX and at least two updates")
    if jax.config.x64_enabled:
        raise ValueError("Training parameters and objective must stay float32")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    parent_params, config, parent, _ = load(args.parent)
    if parent["optimizer"] != "soap":
        raise ValueError("Expected the complete pretrained SOAP checkpoint")
    functions = make_functions(config)
    runner, tx = make_update(functions, args.steps)
    initial, _, initial_state, initial_rng, verified_parent = restore_state(args.parent, tx.init)
    if not same_tree(initial, parent_params) or int(initial_state[0].count) < 2:
        raise ValueError("SOAP continuation state is inconsistent")
    settings = dict(parent_sha256=sha(args.parent), schedule_updates=args.steps,
                    schedule="cosine", initial_lr=1e-3, final_lr=1e-5,
                    parent_updates=parent["updates"], parent_optimizer_count=int(initial_state[0].count),
                    optimizer="soap", optimizer_settings=SOAP_SETTINGS,
                    pool_sha256=parent["pool_sha256"], physics_normalization="input_curvature_sqrt",
                    global_batch=64, microbatches=4, interior_points=48, outlet_points=48,
                    snapshots_every=30000, latest_every=250,
                    jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__, optax=optax.__version__,
                    dependencies=dict(chex="0.1.92", jaxtyping="0.3.11", absl_py="2.3.1",
                                      toolz="1.1.0", typing_extensions="4.15.0", wadler_lindig="0.1.7"),
                    source_sha256={p.name:sha(p) for p in Path(__file__).parent.rglob("*.py")})
    existing = root / "config.json"
    if existing.exists() and json.loads(existing.read_text()) != settings:
        raise ValueError("Run configuration differs from the saved continuation")
    write_json(existing, settings)
    source = Path(__file__).resolve().parents[1]
    bundle = root / "source_bundle.zip"
    if not bundle.exists():
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            paths = [p for p in (source / "flow_operator").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts]
            paths += list(args.references.glob("*.npz"))
            paths += [source / "configs/acceptance.json"]
            archive.write(args.parent, "parent.npz")
            for path in paths:
                archive.write(path, str(path.relative_to(source)))
    (root / "RESUME.txt").write_text(
        "Unzip source_bundle.zip into a project directory. Retain GPU-enabled JAX.\n"
        "Use NumPy 2.3.5, SciPy 1.17.0 and the pinned SOAP dependencies in config.json.\n"
        "Extract checkpoint.npz from the latest resume archive into the run directory.\n"
        "The complete parent checkpoint is included as parent.npz in source_bundle.zip.\n"
        "Run from the project root with the same numeric environment and source.\n"
        "python -m flow_operator.train_soap_cosine --parent /path/to/parent.npz "
        "--output /path/to/run --resume /path/to/checkpoint.npz\n"
        "The cosine index, optimizer moments/bases and production RNG are restored.\n"
    )
    checkpoint = args.resume or root / "latest.npz"
    if checkpoint.exists():
        params, restored_config, state, rng, metadata = restore_state(checkpoint, tx.init)
        if restored_config != config or metadata["cosine_config"] != settings:
            raise ValueError("Invalid resumed schedule configuration")
        index = metadata["cosine_updates"]
        elapsed = metadata["train_seconds"]
    else:
        params, state, rng = initial, initial_state, initial_rng
        index = 0
        elapsed = 0.
    if int(state[0].count) != settings["parent_optimizer_count"] + index:
        raise ValueError("SOAP counter does not match the cosine update index")
    pool, _ = make_verified_pool()
    if tuple(p.stem for p in sorted(args.references.glob("*.npz"))) != ("asymmetric", "beta", "flat", "nine"):
        raise ValueError("Use only the four repeated-validation cases")
    cases = read_cases(args.references)
    thresholds = {k:v for k,v in json.loads((source/"configs/acceptance.json").read_text()).items()
                  if isinstance(v, (int, float))}
    started = time.monotonic()

    def status(phase):
        write_json(root/"status.json",dict(state=phase, cosine_updates=index,
                   target=args.steps, nominal_step=parent["step"]+index,
                   optimizer_count=int(state[0].count), train_seconds=elapsed))

    def record(value):
        with (root/"training.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False)+"\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def save(snapshot=False):
        metadata = dict(verified_parent)
        for key in ("format","config","optimizer_present","state_format","structures",
                    "step","rng_state","array_hashes","parameter_dtype","soap_state_present",
                    "CFD_response_labels_used"):
            metadata.pop(key, None)
        metadata.update(optimizer="soap", learning_rate=float(learning_rate(max(index-1,0),args.steps)),
                        updates=parent["updates"]+index, cosine_updates=index, cosine_config=settings,
                        optimizer_count=int(state[0].count), train_seconds=elapsed, arm="soap_cosine")
        save_state(root/"latest.npz",params,config,state,rng,metadata)
        if snapshot:
            path = root/f"checkpoint_{index:06d}.npz"
            if not path.exists():
                os.link(root/"latest.npz",path)
            p,_,s,r,m=restore_state(path,tx.init)
            if not same_tree((params,state),(p,s)) or r.bit_generator.state!=rng.bit_generator.state:
                raise RuntimeError("Snapshot roundtrip failed")
            status("running")
            package(root,path,index)

    status("compiling")
    disposable = batch(np.random.default_rng(918399),pool,64,48)
    compilation = time.monotonic()
    runner = runner.lower(params,state,jnp.asarray(index),disposable).compile()
    record(dict(event="compiled", seconds=time.monotonic()-compilation, cosine_updates=index))
    expected = runner(params,state,jnp.asarray(index),disposable)
    if not bool(expected[3]):
        save(snapshot=True)
        status("nonfinite_preflight")
        raise RuntimeError("Nonfinite disposable preflight")
    status("running")
    while index < args.steps:
        before_rng = rng.bit_generator.state
        begin = time.monotonic()
        proposed, new_state, metrics, ok = runner(params,state,jnp.asarray(index),batch(rng,pool,64,48))
        if not bool(ok):
            rng.bit_generator.state = before_rng
            save(snapshot=True)
            status("nonfinite")
            raise RuntimeError("Nonfinite update rejected; last finite state saved")
        params, state = proposed, new_state
        index += 1
        elapsed += time.monotonic()-begin
        if index % 50 == 0 or index == 1:
            loss,terms,norm,relative,lr=jax.device_get(metrics)
            record(dict(event="train", updates=index, cosine_updates=index,
                        nominal_step=parent["step"]+index, optimizer_count=int(state[0].count),
                        loss=float(loss), normalized_momentum_mse=float(terms[0]), outlet_mse=float(terms[1]),
                        gradient_norm=float(norm), relative_update_norm=float(relative), learning_rate=float(lr),
                        train_seconds=elapsed))
        if index % 250 == 0:
            save(); status("running")
        if index % 1000 == 0 or index == args.steps:
            validation = measure(params,config,cases,thresholds,functions)
            physics = fixed_physics(params,functions,config)
            record(dict(event="validation", updates=index, fields=validation, fixed_physics=physics))
        if index % 30000 == 0 or index == args.steps:
            save(snapshot=True)
    status("complete")
    write_json(root/"summary.json",dict(updates=index, train_seconds=elapsed,
               invocation_wall_seconds=time.monotonic()-started, optimizer_count=int(state[0].count)))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
