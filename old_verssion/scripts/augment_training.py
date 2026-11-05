"""Add high-Re, large-amplitude conditions to TRAINING only; preserve test data."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from channel_operator.profiles import from_coefficients
from channel_operator.reference import solve_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    if (output/"augmentation.json").exists():
        raise ValueError("Training data was already augmented; refusing duplicate additions")
    with np.load(output/"train.npz") as data:
        original = {k: data[k] for k in data.files}
    rng = np.random.default_rng(90841)
    a = rng.normal(size=(args.count, 4))
    a *= rng.uniform(.4, .6, (args.count, 1))/np.abs(a).sum(axis=1, keepdims=True)
    profiles = from_coefficients(a)
    res = rng.choice([50., 75., 100.], size=args.count, p=[.2, .2, .6])
    fields, records = [], []
    for i, (p, re) in enumerate(zip(profiles, res)):
        reference, info = solve_channel(p, re, nx=40, ny=16)
        fields.append(reference(original["xy"]))
        records.append(info)
        print(f"Additional TRAIN FEM {i+1}/{args.count} Re={re:.0f}", flush=True)
    updated = dict(xy=original["xy"])
    for key, extra in [("profiles", profiles), ("coefficients", a), ("reynolds", res), ("fields", np.array(fields))]:
        updated[key] = np.concatenate([original[key], extra])
    np.savez_compressed(output/"train-augmented.npz", **updated)
    (output/"train.npz").rename(output/"train-before-augmentation.npz")
    (output/"train-augmented.npz").rename(output/"train.npz")
    (output/"augmentation.json").write_text(json.dumps({"seed": 90841, "added_train_cases": args.count,
        "total_train_cases": len(updated["profiles"]), "reynolds": res.tolist(), "reference": records,
        "test_data_changed": False}, indent=2))


if __name__ == "__main__":
    main()
