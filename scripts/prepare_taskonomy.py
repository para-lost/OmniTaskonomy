"""Download official Taskonomy sources and restore the published training selections."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitaskonomy.data.taskonomy import RELEASE_REVISION, TASKS, prepare_taskonomy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/taskonomy"),
                        help="Official extracted sources; missing archives are downloaded here")
    parser.add_argument("--revision", default=RELEASE_REVISION, help="Hugging Face dataset revision")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for task in args.tasks:
        path = prepare_taskonomy(task, args.output_dir / task / "train.jsonl", raw_root=args.raw_root,
                                 revision=args.revision, workers=args.workers)
        print(path)


if __name__ == "__main__":
    main()
