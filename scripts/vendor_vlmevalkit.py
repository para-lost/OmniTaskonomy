#!/usr/bin/env python3
"""Export the experimental gen4und VLMEvalKit working tree for release."""

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitaskonomy.data.common import sha256


ROOT_FILES = {"LICENSE", "requirements.txt", "run.py", "setup.py"}
SOURCE_DIRS = {"vlmeval", "assets", "requirements"}
BAGEL_ROOT = Path("vlmeval/vlm/bagel")
BAGEL_FILES = {"__init__.py", "data/__init__.py", "data/data_utils.py",
               "data/transforms.py", "data/configs/example.yaml"}
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".cache", "outputs", "results",
                 "your_results", "checkpoints", "weights", "wandb", "lmudata"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pkl", ".pickle", ".xlsx", ".pt", ".pth",
                     ".safetensors", ".bin", ".ckpt", ".ipynb", ".pem", ".key", ".md"}


def include_file(path):
    if any(part.lower() in EXCLUDED_DIRS or part.startswith(".env") for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.is_relative_to(BAGEL_ROOT):
        relative = path.relative_to(BAGEL_ROOT)
        return str(relative) in BAGEL_FILES or (relative.parts[0] == "modeling" and path.suffix == ".py")
    return path.parts[0] in SOURCE_DIRS or str(path) in ROOT_FILES


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Research source changed; expected one patch anchor: {old!r}")
    return text.replace(old, new, 1)


def release_config(text):
    assignments = {node.targets[0].id: node for node in ast.parse(text).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    start, end = assignments["bagel_series"].lineno, assignments["o1_key"].lineno
    lines = text.splitlines(keepends=True)
    lines[start - 1:end - 1] = [
        'bagel_series = {"BAGEL-7B-MoT": partial(BAGEL, model_path=BAGEL_ROOT or "checkpoints/BAGEL-7B-MoT")}\n\n'
    ]
    return "".join(lines) + (
        "\n# Explicit class configuration supplies portable base and checkpoint paths.\n"
        "import vlmeval.vlm as vlm\n"
        "from omnitaskonomy.vlmeval_adapter import OmniTaskonomyBAGEL\n"
        "vlm.OmniTaskonomyBAGEL = OmniTaskonomyBAGEL\n"
        "from omnitaskonomy.umm_vlmeval_adapter import OmniTaskonomyUMM\n"
        "vlm.OmniTaskonomyUMM = OmniTaskonomyUMM\n"
    )


def release_runner(text):
    text = replace_once(text, "from vlmeval.smp import *\n",
                        "from vlmeval.smp import *\nfrom omnitaskonomy.eval_status import record_evaluation\n")
    text = replace_once(text, '        eval_id = f"T{date}_G{commit_id}"',
                        '        eval_id = os.environ.get("OMNITASKONOMY_EVAL_ID", f"T{date}_G{commit_id}")')
    text = replace_once(text, "                if RANK == 0 and len(prev_pred_roots):",
                        "                if RANK == 0 and args.reuse and len(prev_pred_roots):")
    text = replace_once(text, "            try:\n                pred_format = get_pred_file_format()",
                        "            if RANK == 0:\n"
                        "                record_evaluation(pred_root, dataset_name, 'running')\n"
                        "            try:\n                pred_format = get_pred_file_format()")
    text = replace_once(text, "                    if args.mode == 'infer':\n                        continue",
                        "                    if args.mode == 'infer':\n"
                        "                        record_evaluation(pred_root, dataset_name, 'done', skip_reason='mode_infer')\n"
                        "                        continue")
    text = replace_once(text, "                    eval_results = dataset.evaluate(result_file, **judge_kwargs)\n",
                        "                    eval_results = dataset.evaluate(result_file, **judge_kwargs)\n"
                        "                    record_evaluation(pred_root, dataset_name, 'done',\n"
                        "                                      judge=judge_kwargs.get('model'), scores=eval_results)\n")
    return replace_once(text, "            except Exception as e:\n                logger.exception(",
                        "            except Exception as e:\n"
                        "                if RANK == 0:\n"
                        "                    record_evaluation(pred_root, dataset_name, 'failed', error_message=str(e))\n"
                        "                logger.exception(")


def release_bagel_adapter(text):
    start = text.index("        # config (llm_config.json")
    end = text.index("        self.use_cpu_offload =", start)
    return (text[:start] + "        self.ori_model_path = os.environ.get('BAGEL_ORI_MODEL_PATH', model_path)\n"
            + text[end:])


def release_setup(text):
    text = replace_once(text, 'with open(\'README.md\', encoding="utf-8") as f:\n    readme = f.read()\n\n\n', "")
    return replace_once(text, "        long_description=readme,\n        long_description_content_type='text/markdown',\n", "")


def export_source(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    repository = Path(subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True).strip())
    prefix = source.relative_to(repository)
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "-z", "--", str(prefix)]
    ).decode().split("\0")
    paths = sorted(Path(name).relative_to(prefix) for name in tracked if name)
    selected = [path for path in paths if include_file(path)]
    if not ROOT_FILES.issubset({str(path) for path in selected}):
        raise ValueError("Source is missing required tracked VLMEvalKit files")
    patches = {"vlmeval/config.py": release_config, "run.py": release_runner, "setup.py": release_setup,
               "vlmeval/vlm/bagel_vlm.py": release_bagel_adapter}
    patched = {path: patch((source / path).read_text()) for path, patch in patches.items()}
    records = []
    for path in selected:
        origin, target = source / path, destination / path
        if origin.is_symlink() or not origin.is_file():
            raise ValueError(f"Expected a regular source file: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if str(path) in patched:
            target.write_text(patched[str(path)])
        records.append({"path": str(path), "source_sha256": sha256(origin),
                        "release_sha256": sha256(target)})
    manifest = destination / "SOURCE_MANIFEST.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "source": "gen4und/VLMEvalKit", "source_kind": "tracked working-tree files",
        "upstream": "https://github.com/open-compass/VLMEvalKit", "revision": revision,
        "revision_scope": "gen4und repository; source hashes also capture uncommitted changes",
        "source_manifest": manifest.name, "source_manifest_sha256": sha256(manifest),
        "license": "LICENSE", "license_sha256": sha256(destination / "LICENSE"),
        "changes": {
            "vlmeval/config.py": "Replace machine-specific BAGEL experiment registry with a portable entry and register BAGEL and custom UMM release adapters.",
            "run.py": "Record completion/failures and isolate run directories for multi-seed aggregation; inference and scoring logic are retained.",
            "vlmeval/vlm/bagel_vlm.py": "Use the supplied model path as the default base directory instead of a private filesystem path.",
            "setup.py": "Use package metadata without a separate vendored README; release instructions live in the root README.",
        },
        "selection": "Tracked runtime source, assets, packaging, and historical BAGEL inference modules.",
        "excluded": "Documentation, Markdown documents, environment/credential files, results, predictions, weights, caches, Git metadata, notebooks, cluster launchers, and BAGEL training/evaluation extras.",
    }
    (destination / "VENDORED.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local gen4und/VLMEvalKit working tree")
    parser.add_argument("--destination", type=Path, required=True, help="A new directory; never overwrite an existing one")
    args = parser.parse_args()
    export_source(args.source, args.destination)
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
