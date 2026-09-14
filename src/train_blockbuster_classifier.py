"""Compatibility CLI for the calibrated blockbuster artifact.

Blockbuster selection shares the temporal folds and feature contract with the
main training run, so this command intentionally invokes the governed trainer.
"""
import argparse
from pathlib import Path
from src.train_model import PRE_RELEASE_FEATURES, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/movies_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/reduced"))
    args = parser.parse_args()
    train(PRE_RELEASE_FEATURES, args.input_path, args.output_dir)


if __name__ == "__main__": main()
