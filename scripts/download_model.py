#!/usr/bin/env python3
"""Download BAGEL weights and record the resolved Hugging Face revision."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/BAGEL-7B-MoT"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
    revision = HfApi().model_info(repo_id, revision=args.revision).sha
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=args.output_dir)
    (args.output_dir / "download.json").write_text(
        json.dumps({"repo_id": repo_id, "revision": revision}, indent=2) + "\n"
    )
    print(f"Downloaded {repo_id}@{revision} to {args.output_dir}")


if __name__ == "__main__":
    main()
