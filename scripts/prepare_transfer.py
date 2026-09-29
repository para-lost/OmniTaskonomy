"""Prepare seven public I2I transfer pools and the fixed LLaVA 50k pool."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitaskonomy.data.transfer import TASKS, prepare_llava, prepare_transfer_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=[*TASKS, "llava"], default=[*TASKS, "llava"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--llava-json", type=Path, help="Original LLaVA-Instruct-150K JSON; downloaded when omitted")
    parser.add_argument("--coco-root", type=Path, default=Path("data/raw/coco"),
                        help="Directory containing train2017/; missing images are extracted from the official archive")
    args = parser.parse_args()
    for task in args.tasks:
        output = args.output_dir / task / "train.jsonl"
        if task == "llava":
            path = prepare_llava(output, llava_json=args.llava_json, coco_root=args.coco_root)
        else:
            path = prepare_transfer_task(task, output)
        print(path)


if __name__ == "__main__":
    main()
