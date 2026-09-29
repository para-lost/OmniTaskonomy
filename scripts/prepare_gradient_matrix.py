"""Prepare the 19 I2I and 25 I2T pools consumed by analyze_gradients.py matrix."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitaskonomy.gradients.prepare_matrix import prepare_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/gradients/transfer.json"))
    args = parser.parse_args()
    print(prepare_matrix(args.data_root, args.output))


if __name__ == "__main__":
    main()
