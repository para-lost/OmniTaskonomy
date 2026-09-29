"""Extract gradients and reproduce the paper's gradient analyses."""

import argparse
import csv
import json
from pathlib import Path

from omnitaskonomy.analysis.gradient_transfer import associate, import_estimates
from omnitaskonomy.data.common import sha256
from omnitaskonomy.gradients.artifacts import load_artifact, validate_rows, write_json
from omnitaskonomy.gradients.manifest import REFERENCE_TASKS, freeze, verify_manifest
from omnitaskonomy.gradients.minibatch import analyze_minibatches
from omnitaskonomy.gradients.norms import summarize_norms
from omnitaskonomy.gradients.pca import analyze_reference, load_basis
from omnitaskonomy.taxonomy import load_taxonomy
from omnitaskonomy.umm import read_options


REFERENCE_SELECTION = Path(__file__).resolve().parents[2] / "data/gradients/reference_selection.json.gz"


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, report)
    for name in ("cells", "selected_cells", "capability_means", "summary", "samples", "layers"):
        rows = report.get(name)
        if rows:
            with path.with_name(path.stem + "_" + name + ".csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def _validate_reference(frozen):
    expected = {(task, objective) for task in REFERENCE_TASKS for objective in ("i2i", "i2t")}
    if frozen["folds"] != 5 or {(row["task"], row["objective"]) for row in frozen["rows"]} != expected:
        raise ValueError("Gradient reference requires all six paired tasks and five folds")
    validate_rows(frozen["rows"], paired=True, folds=5)


def _model_options(args):
    options = {}
    if args.adapter:
        options.update(adapter=args.adapter, adapter_options=read_options(args.adapter_options))
    elif args.adapter_options:
        raise ValueError("--adapter-options requires --adapter")
    if args.checkpoint:
        options["checkpoint"] = args.checkpoint
    return options


def run_modules(args):
    from omnitaskonomy.gradients.extract import extract

    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.energy <= 1:
        raise ValueError("PCA retained energy must lie in (0, 1]")
    model_options = _model_options(args)
    if args.config is None:
        from omnitaskonomy.gradients.reference_data import prepare_recipe_reference

        args.config = Path("data/prepared/gradients/recipe/reference_config.json")
        if not args.config.is_file():
            prepare_recipe_reference(REFERENCE_SELECTION, args.config.parent)
    frozen_path = args.output / "frozen.json"
    frozen = freeze(args.config, frozen_path)
    _validate_reference(frozen)
    extract(frozen_path, args.model_path, args.output / "artifact", modules=args.modules, device=args.device,
            **model_options)
    analyze_reference(load_artifact(args.output / "artifact"), args.output / "pca", layers=True, energy=args.energy)


def run_matrix(args):
    from omnitaskonomy.gradients.extract import extract

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or not 1 <= args.batch_size <= 128:
        raise ValueError("Matrix requires at least two draws and batch size in [1, 128]")
    config = json.loads(args.config.read_text())
    groups = {(group["task"], group["objective"]) for group in config["groups"]}
    expected = {(task["id"], task["modality"]) for task in load_taxonomy()["tasks"]}
    if groups != expected:
        raise ValueError(f"Matrix requires all 19 I2I and 25 I2T taxonomy tasks; "
                         f"missing={sorted(expected - groups)}, extra={sorted(groups - expected)}")
    reference_path = (args.reference / "frozen.json").resolve()
    frozen_reference = verify_manifest(reference_path)
    _validate_reference(frozen_reference)
    reference = load_artifact(args.reference / "artifact")
    if reference["metadata"]["state"] != "base":
        raise ValueError("Matrix requires base-checkpoint reference gradients")
    provenance = reference["metadata"]["provenance"]
    if provenance["manifest_sha256"] != sha256(reference_path):
        raise ValueError("Reference gradients and frozen manifest differ")
    model_options = _model_options(args)
    if not args.adapter:
        checkpoint = args.checkpoint or args.model_path / "ema.safetensors"
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
        if provenance["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Matrix model differs from the reference checkpoint")
    elif (provenance.get("adapter", {}).get("factory") != args.adapter or
          provenance["adapter"]["options"] != model_options["adapter_options"]):
        raise ValueError("Matrix adapter differs from the reference adapter")
    bases = load_basis(args.reference / "pca", reference, module=args.module)
    if "adapter" in provenance:
        model_options["reference_identity"] = provenance["adapter"]["checkpoint_sha256"]
    if config.get("folds", 5) != frozen_reference["folds"]:
        raise ValueError("Matrix and reference must use the same folds")
    frozen_path = args.output / "frozen.json"
    frozen = freeze(args.config, frozen_path, reference=reference_path)
    for task, objective in groups:
        if {row["fold"] for row in frozen["rows"] if row["task"] == task and row["objective"] == objective} != set(bases):
            raise ValueError(f"Matrix pool is missing a reference fold: {task}/{objective}")
    extract(frozen_path, args.model_path, args.output / "artifact", modules=[args.module], device=args.device,
            **model_options)
    artifact = load_artifact(args.output / "artifact")
    report = analyze_minibatches(artifact, reference, bases,
                                repeats=args.repeats, batch_size=args.batch_size)
    write_report(args.output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("modules", help="Run paired I2I/I2T gradient analysis by module and layer")
    command.add_argument("--config", type=Path,
                         help="Prepared reference configuration; default restores the frozen pairs from Hugging Face")
    command.add_argument("--model-path", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--modules", nargs="+", default=["all"])
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--energy", type=float, default=.99)
    command = sub.add_parser("matrix", help="Run the full 19-by-25 gradient alignment matrix")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--reference", required=True, type=Path, help="Output directory of the modules command")
    command.add_argument("--model-path", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--repeats", type=int, default=2000)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--module", default="llm.input_ln.weight", help="Adapter module whose reference PCA to reuse")
    command = sub.add_parser("prepare-reference", help="Restore frozen recipe gradient pairs from Hugging Face or raw sources")
    command.add_argument("--manifest", type=Path, default=REFERENCE_SELECTION)
    command.add_argument("--source-root", type=Path, help="Use local raw research episodes instead of the Hugging Face release")
    command.add_argument("--sources", type=Path, default=Path("configs/gradients/reference_sources.json"))
    command.add_argument("--tasks", nargs="+")
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("prepare-targets", help="Import frozen benchmark gradient samples and authentic correct labels")
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--taxonomy", required=True, type=Path)
    command.add_argument("--path-map", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("freeze", help="Freeze portable manifests and five-fold assignments")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("extract", help="Run per-example backward passes through a model adapter")
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--model-path", required=True, type=Path)
    command.add_argument("--state", default="base", choices=["base", "3k", "10k", "30k"])
    command.add_argument("--modules", nargs="+", default=["llm.input_ln.weight"])
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--sketch-seed", type=int, default=2026082602)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("reference", help="Fit shared reference PCA and score held-out paired gradients")
    command.add_argument("--artifact", required=True, type=Path)
    command.add_argument("--concat-only", action="store_true")
    command.add_argument("--energy", type=float, default=.99)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("minibatch", help="Measure cross-task minibatch alignment without refitting PCA")
    command.add_argument("--artifact", required=True, type=Path)
    command.add_argument("--reference-artifact", required=True, type=Path)
    command.add_argument("--reference-analysis", required=True, type=Path)
    command.add_argument("--module", default="llm.input_ln.weight")
    command.add_argument("--repeats", type=int, default=2000)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("import-estimates", help="Import a research matrix_v14/v15 using released taxonomy IDs")
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--taxonomy", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("associate", help="Join gradient estimates to analysis/transfer.py output")
    command.add_argument("--gradients", required=True, type=Path)
    command.add_argument("--transfer", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("norms", help="Compare matched raw norms and losses across checkpoints")
    command.add_argument("--artifacts", required=True, nargs="+", type=Path)
    command.add_argument("--paper", action="store_true", help="Require 500 examples and all four checkpoints")
    command.add_argument("--output", required=True, type=Path)
    for name in ("modules", "matrix", "extract"):
        model_command = sub.choices[name]
        model_command.add_argument("--adapter", help="Installed Python factory: package.module:create_adapter")
        model_command.add_argument("--adapter-options", type=Path, help="Model-specific JSON options")
        model_command.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.command == "modules":
        run_modules(args)
    elif args.command == "matrix":
        run_matrix(args)
    elif args.command == "prepare-reference":
        from omnitaskonomy.gradients.reference_data import prepare_recipe_reference, prepare_reference
        if args.source_root is None:
            prepare_recipe_reference(args.manifest, args.output, tasks=args.tasks)
        else:
            prepare_reference(args.manifest, args.source_root, args.output, sources=args.sources, tasks=args.tasks)
    elif args.command == "prepare-targets":
        from omnitaskonomy.gradients.targets import prepare_targets
        prepare_targets(args.manifest, args.taxonomy, args.output, path_map=args.path_map)
    elif args.command == "freeze":
        freeze(args.config, args.output)
    elif args.command == "extract":
        from omnitaskonomy.gradients.extract import extract
        extract(args.manifest, args.model_path, args.output, state=args.state,
                modules=args.modules, device=args.device, sketch_seed=args.sketch_seed, **_model_options(args))
    elif args.command == "reference":
        analyze_reference(load_artifact(args.artifact), args.output, layers=not args.concat_only, energy=args.energy)
    elif args.command == "minibatch":
        artifact, reference = load_artifact(args.artifact), load_artifact(args.reference_artifact)
        bases = load_basis(args.reference_analysis, reference, module=args.module)
        write_report(args.output, analyze_minibatches(artifact, reference, bases,
                                                     repeats=args.repeats, batch_size=args.batch_size))
    elif args.command == "import-estimates":
        write_report(args.output, import_estimates(args.input, args.taxonomy))
    elif args.command == "associate":
        gradients, transfer = json.loads(args.gradients.read_text()), json.loads(args.transfer.read_text())
        report = associate(gradients, transfer)
        report["input_audit"]["report_sha256"] = {"gradients": sha256(args.gradients), "transfer": sha256(args.transfer)}
        write_report(args.output, report)
    elif args.command == "norms":
        write_report(args.output, summarize_norms([load_artifact(path) for path in args.artifacts], paper=args.paper))
    print(args.output)


if __name__ == "__main__":
    main()
