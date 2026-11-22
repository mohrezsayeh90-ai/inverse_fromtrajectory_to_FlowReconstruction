"""Matched Adam/SOAP continuation from the audited inlet checkpoint."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._vendor.soap_jax.soap import soap
from .checkpoint import load
from .evaluate import evaluate_fields
from .model import Config, make_functions
from .sampling import make_pool, diagnostic_profiles
from .soap_checkpoint import restore_state, save_state
from .train import adam_update, batch, make_loss, replicate, shard_batch
from .validation import compare, read_cases

BASE_SHA = "ccf6fa2b3e179a22c443384083e2dec338fc497f9562a567ed7e0955d3c67bf8"
POOL_SHA = "0894ca0065947a7de6979c13ae3c1d543aa81d19118f0f8c15cdc8bcf56c4103"
BASE_STEP = 94000
POOL_SEED = 9703107
POOL_COUNT = 10000
NORMALIZATION = "input_curvature_sqrt"
SCREEN = 5000
FINAL = 20000
CASE_NAMES = ("asymmetric", "beta", "flat", "nine")
GUARDS = {"velocity_relative_l2": 1e-10, "pressure_relative_l2": 1e-10, "v_rmse": 1e-10}
STRUCTURE = ("divergence_rms", "boundary_max_error", "flux_max_error")
SOAP_SETTINGS = dict(b1=.95, b2=.95, shampoo_beta=-1, eps=1e-8, weight_decay=0.,
                     precondition_frequency=10, max_precond_dim=1024, precondition_1d=False)
ARMS = {"adam_control": None, "soap_lr_2e5": 2e-5, "soap_lr_5e5": 5e-5, "soap_lr_1e4": 1e-4}
VENDOR = Path(__file__).parent / "_vendor"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def same_tree(actual, expected):
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        return False
    return all(np.array_equal(a, b) for a, b in
               zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))


def verify_base(path):
    if sha(path) != BASE_SHA:
        raise ValueError("Expected the exact selected 94000-step checkpoint")
    params, config, meta, state = load(path, optimizer=True)
    expected = dict(step=BASE_STEP, pool_seed=POOL_SEED, pool_count=POOL_COUNT,
                    pool_sha256=POOL_SHA, physics_normalization=NORMALIZATION,
                    global_batch=64, physics_points_per_condition=48, encoder_points=257)
    if config != Config(separate_pressure_trunk=True) or any(meta[k] != v for k, v in expected.items()):
        raise ValueError("Base training specification differs")
    if sum(p.size for p in jax.tree.leaves(params)) != 513024:
        raise ValueError("Unexpected base parameter count")
    if any(p.dtype != np.float32 or not np.isfinite(p).all() for p in jax.tree.leaves((params, state))):
        raise ValueError("Invalid base weights or Adam moments")
    return params, config, meta, state


def make_verified_pool():
    pool, meta = make_pool(POOL_COUNT, POOL_SEED)
    digest = hashlib.sha256()
    for array in pool.values():
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    if digest.hexdigest() != POOL_SHA:
        raise ValueError(f"Regenerated input pool differs: {digest.hexdigest()}")
    held, _, names = diagnostic_profiles()
    for name, coeff in zip(names, held["coeff"], strict=True):
        if name != "poiseuille" and np.any(np.all(pool["coeff"] == coeff, axis=(1, 2))):
            raise ValueError(f"Repeated diagnostic profile in training pool: {name}")
    return pool, meta


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(x, x).real for x in jax.tree.leaves(tree)))


def finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def make_gradient(functions, devices=(), microbatches=1):
    """Average unmodified gradients over the full production batch before clipping."""
    value_grad = jax.value_and_grad(make_loss(functions, NORMALIZATION), has_aux=True)

    def gradient(params, data):
        if microbatches == 1:
            result = value_grad(params, data)
        else:
            chunks = shard_batch(data, microbatches)
            chunk = lambda i: jax.tree.map(lambda x: x[i], chunks)
            initial = value_grad(params, chunk(0))

            def accumulate(i, total):
                return jax.tree.map(jnp.add, total, value_grad(params, chunk(i)))

            result = jax.tree.map(lambda x: x / microbatches,
                                  jax.lax.fori_loop(1, microbatches, accumulate, initial))
        if devices:
            result = jax.lax.pmean(result, "training_devices")
        return result

    return gradient


def make_update(functions, rate, devices=(), microbatches=1):
    tx = soap(learning_rate=rate, **SOAP_SETTINGS) if rate is not None else None
    gradient = make_gradient(functions, devices, microbatches)

    def update(params, state, count, data):
        (loss, terms), grad = gradient(params, data)
        norm = tree_norm(grad)
        clipped = jax.tree.map(lambda g: g * jnp.minimum(1., 100 / (norm + 1e-12)), grad)
        if tx is None:
            new_params, m, v, lr = adam_update(params, *state, count, clipped)
            new_state = (m, v)
            internal = count
        else:
            updates, new_state = tx.update(clipped, state, params)
            new_params = optax.apply_updates(params, updates)
            lr = jnp.asarray(rate, dtype=jnp.float32)
            internal = new_state[0].count
        delta = tree_norm(jax.tree.map(jnp.subtract, new_params, params))
        relative = delta / jnp.maximum(tree_norm(params), 1e-30)
        metrics = (loss, terms, norm, relative, lr, internal)
        return new_params, new_state, metrics, finite((norm, grad, new_params, new_state, metrics))

    compiled = jax.pmap(update, axis_name="training_devices", devices=devices) if devices else jax.jit(update)
    return compiled, tx


def fixed_physics(params, functions, config):
    profiles, reynolds, names = diagnostic_profiles()
    xy = np.random.default_rng(918390).uniform(size=(len(names), 1024, 2)).astype(np.float32)
    xy[..., 0] = config.length * xy[..., 0] ** 2
    y = np.broadcast_to(np.linspace(.001, .999, 96, dtype=np.float32), (len(names), 96))
    residual = np.asarray(functions["residual"](params, profiles, reynolds, xy))
    outlet = np.asarray(functions["outlet"](params, profiles, reynolds, y))
    return [dict(name=name, reynolds=float(re), momentum_rms=float(np.sqrt(np.mean(np.sum(r[:, :2]**2, axis=-1)))),
                 divergence_rms=float(np.sqrt(np.mean(r[:, 2]**2))),
                 outlet_traction_rms=float(np.sqrt(np.mean(np.sum(o**2, axis=-1)))))
            for name, re, r, o in zip(names, reynolds, residual, outlet, strict=True)]


def measure(params, config, cases, thresholds, functions):
    with jax.enable_x64():
        double = jax.tree.map(lambda array: array.astype(jnp.float64), params)
        report = evaluate_fields(double, config, cases, thresholds, functions=functions)
    report["score"] = max(c[k] / v for c in report["cases"] for k, v in thresholds.items())
    report["field_tiebreak"] = max(c[k] / thresholds[k] for c in report["cases"]
                                   for k in ("velocity_relative_l2", "pressure_relative_l2"))
    return report


def guardrails(candidate, baseline, factor):
    flags = []
    for a, b in zip(candidate["cases"], baseline["cases"], strict=True):
        if a["name"] != b["name"]:
            raise ValueError("Unmatched validation cases")
        for field, tolerance in GUARDS.items():
            ceiling = b[field] + tolerance if b[field] <= tolerance else factor * b[field]
            if a[field] > ceiling:
                flags.append(dict(case=a["name"], metric=field, value=a[field], ceiling=ceiling))
    return flags


@dataclass
class Progress:
    params: object
    state: object
    rng: np.random.Generator
    updates: int = 0
    initialized: int = 0
    train_seconds: float = 0.
    clipped: int = 0
    checkpoint_seconds: float = 0.


class Experiment:
    def __init__(self, args, devices, started):
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = started + args.wall_seconds - args.reserve_seconds
        self.final_deadline = started + args.wall_seconds
        self.started = started
        self.base, self.config, self.meta, self.adam_state = verify_base(args.checkpoint)
        self.pool, pool_meta = make_verified_pool()
        self.devices = devices if len(devices) > 1 else ()
        self.microbatches = 1 if self.devices else 4
        self.functions = make_functions(self.config)
        paths = sorted(Path(args.references).glob("*.npz"))
        if tuple(p.stem for p in paths) != CASE_NAMES:
            raise ValueError("Provide only the four repeated-validation reference files")
        self.cases = read_cases(args.references)
        self.thresholds = {k: v for k, v in json.loads(Path(args.acceptance).read_text()).items()
                           if isinstance(v, (float, int))}
        if len(self.thresholds) != 7:
            raise ValueError("Expected seven original acceptance thresholds")
        vendor = json.loads((VENDOR / "upstream.json").read_text())
        for name, digest in vendor["unmodified_files"].items():
            if sha(VENDOR / name) != digest:
                raise ValueError(f"Vendored SOAP source differs: {name}")
        self.provenance = dict(
            parent_sha256=BASE_SHA, base_adam_count=BASE_STEP, dtype="float32",
            pool_sha256=POOL_SHA, pool_seed=POOL_SEED, pool_count=POOL_COUNT,
            encoder_points=257, physics_normalization=NORMALIZATION,
            global_batch=64, interior_points=48, outlet_points=48, gradient_clip=100.,
            optimizer_vendor=vendor, reference_sha256={p.name: sha(p) for p in paths},
            source_sha256={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        )
        self.settings = dict(self.provenance, geometry=self.config.json(), updates_budget=50000,
                             wall_seconds=args.wall_seconds, reserve_seconds=args.reserve_seconds,
                             devices=[str(d) for d in devices], microbatches=self.microbatches,
                             screen=SCREEN, final=FINAL, arms=ARMS, soap_settings=SOAP_SETTINGS,
                             jax=jax.__version__, numpy=np.__version__, optax=optax.__version__,
                             guard_absolute_tolerance=GUARDS)
        config_path = self.out / "config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            for key in ("source_sha256", "reference_sha256", "pool_sha256", "parent_sha256",
                        "devices", "microbatches", "jax", "numpy", "optax"):
                if previous[key] != self.settings[key]:
                    raise ValueError(f"Resume setting differs: {key}")
        write_json(config_path, self.settings)
        write_json(self.out / "input_pool_metadata.json", pool_meta)
        base_copy = self.out / "base_checkpoint.npz"
        if not base_copy.exists():
            pending = base_copy.with_suffix(".tmp")
            shutil.copyfile(args.checkpoint, pending)
            pending.replace(base_copy)
        if sha(base_copy) != BASE_SHA:
            raise ValueError("Run's preserved base checkpoint differs")
        self.updates = {}
        self.status = {}
        self.validation_seconds = 0.
        self.preflight_seconds = 0.

    def cursor(self):
        rng = np.random.default_rng()
        rng.bit_generator.state = self.meta["rng_state"]
        return rng

    def output_tree(self, tree):
        return jax.tree.map(lambda x: x[0], tree) if self.devices else tree

    def inputs(self, params, state, count, draw):
        if not self.devices:
            return params, state, jnp.asarray(count), draw
        n = len(self.devices)
        return replicate(params, n), replicate(state, n), jnp.full((n,), count), shard_batch(draw, n)

    def inspect_checkpoint(self, path, state_template, progress):
        p, c, state, rng, metadata = restore_state(path, state_template)
        if (c != self.config or not same_tree((p, state), (progress.params, progress.state))
                or rng.bit_generator.state != progress.rng.bit_generator.state
                or metadata["updates"] != progress.updates):
            raise RuntimeError(f"Checkpoint roundtrip failed: {path}")
        draw = batch(np.random.default_rng(0), self.pool, 1, 1)
        inputs = (draw["profiles"], draw["reynolds"], draw["xy"])
        np.testing.assert_array_equal(self.functions["fields"](p, *inputs),
                                      self.functions["fields"](progress.params, *inputs))

    def save(self, arm, progress, rate, init, validate=False):
        folder = self.out / arm
        path = folder / "latest.npz"
        count = BASE_STEP + progress.updates if rate is None else progress.initialized + progress.updates
        metadata = dict(self.provenance, arm=arm, optimizer="adam" if rate is None else "soap",
                        learning_rate=rate, optimizer_settings=(
                            dict(b1=.9, b2=.999, eps=1e-8, lr_formula="max(2e-5, 1e-3 * .95**(count/2000))")
                            if rate is None else SOAP_SETTINGS),
                        updates=progress.updates, basis_initializations=progress.initialized,
                        optimizer_count=count, stage="screen" if progress.updates <= SCREEN else "extension",
                        train_seconds=progress.train_seconds, clipped=progress.clipped,
                        checkpoint_seconds=progress.checkpoint_seconds, soap_init_seed=918391)
        started = time.monotonic()
        save_state(path, progress.params, self.config, progress.state, progress.rng, metadata)
        snap = folder / f"update_{progress.updates:05d}.npz"
        if validate or progress.updates in (100, 1000, 2500, SCREEN) or (
                progress.updates > SCREEN and progress.updates % 2500 == 0):
            if not snap.exists():
                os.link(path, snap)
        if validate:
            self.inspect_checkpoint(path, init, progress)
        progress.checkpoint_seconds += time.monotonic() - started

    def read_progress(self, arm, rate, init):
        path = self.out / arm / "latest.npz"
        if not path.exists():
            path = self.out / arm / "initialized.npz"
        if not path.exists():
            return Progress(self.base, init(self.base), self.cursor())
        params, cfg, state, rng, meta = restore_state(path, init)
        if (cfg != self.config or meta["arm"] != arm or meta["learning_rate"] != rate
                or any(meta[k] != v for k, v in self.provenance.items())):
            raise ValueError(f"Incompatible run history in {path}")
        updates, initialized = meta["updates"], meta["basis_initializations"]
        if rate is None:
            valid = initialized == 0 and meta["optimizer_count"] == BASE_STEP + updates
        else:
            valid = initialized == 1 and int(state[0].count) == meta["optimizer_count"] == updates + initialized
        if not valid or not 0 <= updates <= FINAL:
            raise ValueError(f"Invalid optimizer counters in {path}")
        return Progress(params, state, rng, updates, initialized, meta["train_seconds"],
                        meta["clipped"], meta["checkpoint_seconds"])

    def log(self, arm, record):
        with (self.out / arm / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(arm, json.dumps(record, allow_nan=False), flush=True)

    def endpoint(self, arm, progress, tag):
        started = time.monotonic()
        report = measure(progress.params, self.config, self.cases, self.thresholds, self.functions)
        report["fixed_physics"] = fixed_physics(progress.params, self.functions, self.config)
        elapsed = time.monotonic() - started
        self.validation_seconds += elapsed
        snap = self.out / arm / f"update_{progress.updates:05d}.npz"
        if arm == "base":
            snap = self.out / "base_checkpoint.npz"
        report.update(updates=progress.updates, training_seconds=progress.train_seconds,
                      checkpoint_seconds=progress.checkpoint_seconds, evaluation_seconds=elapsed,
                      clip_fraction=(progress.clipped / progress.updates if progress.updates else 0.),
                      rng_state=progress.rng.bit_generator.state,
                      checkpoint=str(snap.relative_to(self.out)), checkpoint_sha256=sha(snap))
        write_json(self.out / arm / f"{tag}.json", report)
        return report

    def run_arm(self, arm, target):
        rate = ARMS[arm]
        folder = self.out / arm
        folder.mkdir(exist_ok=True)
        stage = "screen" if target == SCREEN else "final"
        endpoint = folder / f"{stage}.json"
        if endpoint.exists():
            saved = json.loads(endpoint.read_text())
            if saved["updates"] != target or sha(self.out / saved["checkpoint"]) != saved["checkpoint_sha256"]:
                raise ValueError(f"Invalid recorded endpoint: {endpoint}")
            return saved, "complete"
        if time.monotonic() >= self.deadline:
            return None, "deadline"

        if arm not in self.updates:
            self.updates[arm] = make_update(self.functions, rate, self.devices, self.microbatches)
        runner, tx = self.updates[arm]
        init = tx.init if tx is not None else lambda _: self.adam_state
        progress = self.read_progress(arm, rate, init)
        if progress.updates > target:
            raise ValueError(f"Resume passed missing {stage} endpoint for {arm}")

        def data(rng):
            return batch(rng, self.pool, 64, 48, self.config.length)

        compile_seconds = 0.
        if progress.updates < target:
            test_inputs = self.inputs(progress.params, progress.state, BASE_STEP + progress.updates + 1,
                                      data(np.random.default_rng(918391)))
            start = time.monotonic()
            runner = runner.lower(*test_inputs).compile()
            compile_seconds = time.monotonic() - start
        self.log(arm, dict(event="start", stage=stage, updates=progress.updates,
                           compile_seconds=compile_seconds))

        params, state = progress.params, progress.state
        if self.devices:
            params, state = replicate(params, len(self.devices)), replicate(state, len(self.devices))

        def update(count, draw):
            if self.devices:
                n = len(self.devices)
                return runner(params, state, jnp.full((n,), count), shard_batch(draw, n))
            return runner(params, state, jnp.asarray(count), draw)

        if rate is not None and progress.initialized == 0 and time.monotonic() < self.deadline:
            start = time.monotonic()
            new, state, m, ok = update(1, data(np.random.default_rng(918391)))
            if (not bool(self.output_tree(ok)) or not same_tree(new, params)
                    or int(self.output_tree(m)[-1]) != 1):
                self.log(arm, dict(event="basis_failure", updates=0))
                return None, "nonfinite_or_invalid_basis"
            progress.state = self.output_tree(state)
            progress.initialized = 1
            self.save(arm, progress, rate, init)
            initialized_path = folder / "initialized.npz"
            os.link(folder / "latest.npz", initialized_path)
            preflight = self.out / "production_preflight.json"
            if not preflight.exists() and time.monotonic() < self.deadline:
                preflight_started = time.monotonic()
                p, _, st, cursor, _ = restore_state(initialized_path, init)
                original_draw, restored_draw = data(self.cursor()), data(cursor)
                if not same_tree(original_draw, restored_draw):
                    raise ValueError("Production draw changed after full-state restore")
                expected = update(BASE_STEP + 1, original_draw)
                restored = runner(*self.inputs(p, st, BASE_STEP + 1, restored_draw))
                if not bool(self.output_tree(expected[3])) or not same_tree(expected, restored):
                    raise ValueError("Production update changed after full-state restore")
                if same_tree(params, expected[0]):
                    raise ValueError("First counted SOAP update leaves parameters unchanged")
                self.preflight_seconds += time.monotonic() - preflight_started
                write_json(preflight, dict(arm=arm, production_batch=64, verified_update=1,
                                          separate_preflight_calls=2, seconds=self.preflight_seconds,
                                          next_draw_matches=True, next_update_matches=True))
            self.log(arm, dict(event="basis_initialization", updates=0, rng_seed=918391,
                               optimizer_count=1, seconds=time.monotonic() - start))

        disposition = "deadline"
        while progress.updates < target and time.monotonic() < self.deadline:
            saved_rng = progress.rng.bit_generator.state
            count = BASE_STEP + progress.updates + 1
            start = time.monotonic()
            proposed, proposed_state, metrics, ok = update(count, data(progress.rng))
            if not bool(self.output_tree(ok)):
                progress.rng.bit_generator.state = saved_rng
                disposition = "nonfinite"
                self.log(arm, dict(event=disposition, updates=progress.updates, proposed_step=count))
                break
            progress.train_seconds += time.monotonic() - start
            params, state = proposed, proposed_state
            progress.updates += 1
            loss, terms, norm, relative, lr, internal = jax.device_get(self.output_tree(metrics))
            if int(internal) != (count if rate is None else progress.updates + progress.initialized):
                raise ValueError("Optimizer count advanced unexpectedly")
            progress.clipped += int(norm > 100.)
            n = progress.updates
            if n % 50 == 0 or n == 1:
                record = dict(event="train", updates=n, nominal_step=count, optimizer_count=int(internal),
                              loss=float(loss), normalized_momentum_mse=float(terms[0]),
                              outlet_mse=float(terms[1]), divergence_mse=float(terms[2]),
                              gradient_norm=float(norm), clipping_fraction=progress.clipped / n,
                              relative_update_norm=float(relative), learning_rate=float(lr),
                              train_seconds=progress.train_seconds, seconds_per_update=progress.train_seconds / n)
                if n == 100:
                    stats = [d.memory_stats() or {} for d in jax.local_devices()]
                    record["first_100_updates"] = True
                    record["device_peak_bytes"] = [s.get("peak_bytes_in_use") for s in stats]
                    record["estimated_training_seconds_at_this_rate"] = record["seconds_per_update"] * target
                self.log(arm, record)
            if n % 250 == 0 or n == 100:
                progress.params, progress.state = self.output_tree((params, state))
                self.save(arm, progress, rate, init)
                self.status[arm] = dict(updates=n, stage=stage, state="running")
                self.write_status()
            if n % 500 == 0 and n < target and time.monotonic() < self.deadline:
                start = time.monotonic()
                current = self.output_tree(params)
                record = dict(event="validation", updates=n,
                              fixed_physics=fixed_physics(current, self.functions, self.config))
                if n % 1000 == 0:
                    with jax.enable_x64():
                        double = jax.tree.map(lambda x: x.astype(jnp.float64), current)
                        record["fields"] = compare(double, self.functions, self.cases)
                record["seconds"] = time.monotonic() - start
                self.validation_seconds += record["seconds"]
                self.log(arm, record)

        progress.params, progress.state = self.output_tree((params, state))
        if progress.updates or progress.initialized:
            self.save(arm, progress, rate, init, validate=True)
        if progress.updates == target:
            return self.endpoint(arm, progress, stage), "complete"
        if progress.updates and disposition == "nonfinite" and time.monotonic() < self.final_deadline:
            self.endpoint(arm, progress, "rejected")
        return None, disposition

    def write_status(self):
        write_json(self.out / "status.json", dict(
            arms=self.status, wall_seconds=time.monotonic() - self.started,
            validation_seconds=self.validation_seconds, preflight_seconds=self.preflight_seconds,
        ))

    def run(self):
        self.write_status()
        base_dir = self.out / "base"
        base_dir.mkdir(exist_ok=True)
        base_path = base_dir / "metrics.json"
        if base_path.exists():
            base = json.loads(base_path.read_text())
        else:
            base = self.endpoint("base", Progress(self.base, self.adam_state, self.cursor()), "metrics")
        screens = {}
        for arm in ARMS:
            report, status = self.run_arm(arm, SCREEN)
            self.status[arm] = dict(self.status.get(arm, {}), screen=status)
            self.write_status()
            if report is not None:
                screens[arm] = report
        if screens and any(r["rng_state"] != next(iter(screens.values()))["rng_state"] for r in screens.values()):
            raise ValueError("Screen production RNG endpoints differ")
        failures = {arm: guardrails(report, base, 1.20) for arm, report in screens.items() if ARMS[arm] is not None}
        ranked = [arm for arm, flags in failures.items() if not flags]
        all_screens_decided = all(self.status[a]["screen"] != "deadline" for a in ARMS)
        winner = min(ranked, key=lambda a: (screens[a]["score"], screens[a]["field_tiebreak"],
                                             screens[a]["training_seconds"])) if ranked and all_screens_decided else None
        write_json(self.out / "selection.json", dict(winner=winner, screen_guard_failures=failures,
                                                    all_screens_decided=all_screens_decided,
                                                    screen_rng_equal=len(screens) == len(ARMS)))
        final = {}
        if "adam_control" in screens and all_screens_decided:
            for arm in [a for a in ("adam_control", winner) if a]:
                report, status = self.run_arm(arm, FINAL)
                self.status[arm]["extension"] = status
                self.write_status()
                if report is not None:
                    final[arm] = report
        decision = "partial" if not all_screens_decided else "inconclusive"
        gates = {}
        if winner and winner in final and "adam_control" in final:
            a, s = final["adam_control"], final[winner]
            if a["rng_state"] != s["rng_state"]:
                raise ValueError("Final production RNG endpoints differ")
            gates = dict(score_improved_5pct=s["score"] <= .95 * a["score"],
                         guard_failures=guardrails(s, a, 1.05),
                         structural_limits_passed=all(s["gates"][k] for k in STRUCTURE))
            decision = ("promising" if gates["score_improved_5pct"] and not gates["guard_failures"]
                        and gates["structural_limits_passed"] else "mixed")
        if any(item.get("extension") == "deadline" for item in self.status.values()):
            decision = "partial"
        updates = {}
        for name in ARMS:
            path = self.out / name / "latest.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    updates[name] = json.loads(str(archive["metadata"]))["updates"]
        write_json(self.out / "summary.json", dict(decision=decision, gates=gates, winner=winner,
                   updates_by_arm=updates, updates=sum(updates.values()), budget=50000,
                   optimizer_comparison_complete=winner in final and "adam_control" in final,
                   status=self.status, wall_seconds=time.monotonic() - self.started,
                   validation_seconds=self.validation_seconds, preflight_seconds=self.preflight_seconds,
                   normalization=NORMALIZATION))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--acceptance", default="configs/acceptance.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wall-seconds", type=int, default=28800)
    parser.add_argument("--reserve-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1800 <= args.reserve_seconds < args.wall_seconds <= 28800:
        parser.error("Use a runtime cap of at most eight hours with at least 30 minutes reserved")
    devices = tuple(d for d in jax.local_devices() if d.platform == "gpu")[:2]
    if not devices or jax.config.x64_enabled:
        raise RuntimeError("This experiment requires CUDA JAX with float32 training defaults")
    Experiment(args, devices, started).run()


if __name__ == "__main__":
    main()
