#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $1 == --help || $1 == -h ]]; then
    echo "Usage: $0 TRAINING_ROOT EVALUATION_ROOT [evaluate.py options]"
    echo "Seeds default to 42 43 44; append --seeds to override."
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then exit 0; fi
    exit 2
fi

training_root=$1
evaluation_root=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shopt -s nullglob
count=0

for run in "$training_root"/*/checkpoints.json; do
    job=$(basename -- "$(dirname -- "$run")")
    [[ $job == *_stage1 ]] && continue
    python "$script_dir/evaluate.py" --run-file "$run" --seeds 42 43 44 \
        --output-dir "$evaluation_root/$job" "$@"
    count=$((count + 1))
done

if [[ $count == 0 ]]; then
    echo "No final checkpoints.json found under $training_root" >&2
    exit 1
fi
