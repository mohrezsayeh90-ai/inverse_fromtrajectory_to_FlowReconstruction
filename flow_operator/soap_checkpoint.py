"""Atomic, pickle-free continuation checkpoints compatible with inference load()."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import FORMAT, load
from .model import initialize

STATE_FORMAT = "inlet_optimizer_state_v1"


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _save(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, array_hashes={k: _digest(v) for k, v in arrays.items()})
    payload = json.dumps(metadata, sort_keys=True, allow_nan=False)
    arrays["metadata"] = np.array(payload)
    arrays["metadata_sha256"] = np.array(hashlib.sha256(payload.encode()).hexdigest())
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            np.savez(tmp, **arrays)
            tmp.flush()
            os.fsync(tmp.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_state(path, params, config, state, rng, metadata):
    kind = metadata["optimizer"]
    arrays = {}
    trees = {}
    for prefix, tree in (("parameter", params), ("state", state)):
        leaves, structure = jax.tree_util.tree_flatten(tree)
        trees[prefix] = str(structure)
        arrays.update({f"{prefix}_{i}": np.asarray(a) for i, a in enumerate(leaves)})
    _save(path, arrays, dict(
        metadata, format=FORMAT, config=config.json(), optimizer_present=False,
        state_format=STATE_FORMAT, structures=trees,
        step=metadata["base_adam_count"] + metadata["updates"],
        rng_state=rng.bit_generator.state, soap_state_present=kind == "soap",
        parameter_dtype="float32", CFD_response_labels_used=0,
    ))


def restore_state(path, state_template):
    params, config, metadata, _ = load(path)
    if metadata["state_format"] != STATE_FORMAT:
        raise ValueError("Unsupported continuation checkpoint")
    state_template = state_template(params)
    with np.load(path, allow_pickle=False) as data:
        payload = str(data["metadata"])
        if hashlib.sha256(payload.encode()).hexdigest() != str(data["metadata_sha256"]):
            raise ValueError("Corrupt continuation metadata")
        for key, expected in metadata["array_hashes"].items():
            if _digest(data[key]) != expected:
                raise ValueError(f"Corrupt continuation array {key}")

        def restore(prefix, template):
            leaves, structure = jax.tree_util.tree_flatten(template)
            if str(structure) != metadata["structures"][prefix]:
                raise ValueError(f"Incompatible {prefix} tree")
            values = []
            for i, expected in enumerate(leaves):
                value = data[f"{prefix}_{i}"]
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f"Incompatible {prefix} array {i}")
                values.append(jnp.asarray(value))
            return jax.tree_util.tree_unflatten(structure, values)

        params = restore("parameter", initialize(config))
        state = restore("state", state_template)
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata["rng_state"]
    return params, config, state, rng, metadata
