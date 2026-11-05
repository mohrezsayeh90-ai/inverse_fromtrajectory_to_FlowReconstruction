import argparse
from pathlib import Path
from channel_operator.data import generate
from channel_operator.train import train
from channel_operator.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--train-cases", type=int, default=48)
    parser.add_argument("--test-cases", type=int, default=8)
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--entrance-features", action="store_true")
    parser.add_argument("--data-weight", type=float)
    parser.add_argument("--stage", choices=["all", "data", "train", "evaluate"], default="all")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.steps < 1 or args.train_cases < 2 or args.test_cases < 1:
        parser.error("Need positive steps, >=2 training cases, >=1 test case")
    if args.data_weight is not None and args.data_weight < 0:
        parser.error("Data weight cannot be negative")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.stage in ["all", "data"]:
        generate(args.output, args.train_cases, args.test_cases)
    if args.stage in ["all", "train"]:
        train(args.output, steps=args.steps, physics_only=args.physics_only, resume=args.resume, entrance_features=args.entrance_features, data_weight=args.data_weight)
    if args.stage in ["all", "evaluate"]:
        evaluate(args.output)


if __name__ == "__main__":
    main()
