#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/train.txt -r requirements/eval.txt
python -m pip install -r requirements/flash-attn.txt --no-build-isolation
python -m pip install -e '.[analysis]'
python -m pip install -e ./VLMEvalKit --no-deps
