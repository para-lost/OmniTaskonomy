"""Cache the released recipe images and train/val manifests."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitaskonomy.data.recipe import TASKS, prepare_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(dict.fromkeys(TASKS.values())))
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    args = parser.parse_args()
    for task in args.tasks:
        for split in args.splits:
            directory = args.output_dir / task / split
            paths = prepare_recipe(task, {kind: directory / f"{kind}.jsonl" for kind in ("i2i", "i2t")}, split=split)
            for path in paths.values():
                print(path)


if __name__ == "__main__":
    main()
