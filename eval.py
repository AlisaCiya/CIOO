"""
Single-model evaluation entrypoint.

Usage:
  python eval.py --weights weights/0-14n.pt --nc_steps [15] --name n
  python eval.py --weights weights/0-14-19n.pt --nc_steps [15,5] --name n
"""

import argparse
import ast
import os

from trainer import IncrementalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Single-model evaluation")

    parser.add_argument("--weights", type=str, required=True,
                        help="Checkpoint to evaluate")
    parser.add_argument("--name", default="n", type=str,
                        help="Model scale (n/s/m/l/x)")
    parser.add_argument("--nc_steps", type=ast.literal_eval, default=[15],
                        help="Class steps, e.g. [15] or [15,5]")
    parser.add_argument("--data-dir", type=str, default="VOC")
    parser.add_argument("--inp-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--disjoint", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    args.eval_weights = args.weights
    args.out = os.path.splitext(args.weights)[0] + ".pt"
    args.path = "weights"
    if args.nc_steps:
        total = sum(args.nc_steps)
        args.allowed_classes = set(range(total))
    return args


def main():
    args = parse_args()
    print(f"评估类别范围: 0-{sum(args.nc_steps) - 1}, disjoint={args.disjoint}")
    trainer = IncrementalTrainer(args)
    overall, per_class = trainer.evaluate(model_path=args.weights, save_csv=True)

    print("\nOverall:")
    print({
        "precision": overall[0],
        "recall": overall[1],
        "mAP50": overall[2],
        "mAP50-95": overall[3],
    })


if __name__ == "__main__":
    main()
