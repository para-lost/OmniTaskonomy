# Test your own UMM on OmniTaskonomy

Complete the [environment setup](../README.md#quick-start) once, then run these commands from the repository root. See [data availability and preparation](reproduce_bagel.md#omnitaskonomy-data) for the transfer pools and paired recipe data.

## Add an adapter

Keep your model in an importable Python package and expose `my_umm.adapter:create_adapter`. The factory receives `model_path`, `checkpoint`, `device` and an `options` dictionary. See the [BAGEL adapter](../omnitaskonomy/adapters/bagel.py) for a pretrained model and the [tiny CPU example](../omnitaskonomy/examples/tiny_umm.py) for a complete implementation. The tiny model tests the interface.

<details>
<summary>Adapter contract</summary>

- `model`: the PyTorch module containing the parameters used by both objectives.
- `loss(records, objective, context) → Loss(total, count)`: prepare a batch and return a differentiable loss sum and its normalization count. Training and gradient extraction call this same method. `context` supplies the manifest path, RNG seed, conditioning dropout and transform settings. Frozen benchmark records contain `gradient_mcq` with the exact prompt and correct option.
- `parameter_specs()`: yield a `ParameterSpec` for each parameter, with role `generation`, `understanding`, `shared` or `fixed`. Set `module` and `layer` for gradient grouping; use `module=None` to exclude a parameter from analysis. Declare tied parameters once, assign LoRA parameters explicitly, and use `zero_objectives` for structurally absent gradients.
- `generate(messages, **kwargs)`: accept ordered image/text messages and return a text answer for benchmarks. An adapter may also provide image generation via `output="image"`.
- `save_checkpoint(directory)`: save weights that the factory can reload. `checkpoint_files` lists every loaded checkpoint, config and tokenizer asset for provenance.

`Loss`, `LossContext` and `ParameterSpec` are defined in [umm.py](../omnitaskonomy/umm.py). R4 selects eligible `generation` parameters and restores initial eligibility for Mixed. Fixed parameters remain fixed during training. Gradient analysis differentiates the parameters assigned to the selected modules, without updating weights. R4 rejects models with no eligible generation-specific parameters.

</details>

## Train on Jigsaw or Zoom-In

Apply the Toy R4 protocol to your model.

```bash
python scripts/train.py --adapter my_umm.adapter:create_adapter --model-path /path/to/model --suite controlled --task jigsaw --recipe r4 --i2i-manifest data/prepared/jigsaw/train/i2i.jsonl --i2t-manifest data/prepared/jigsaw/train/i2t.jsonl --nproc-per-node 1 --output-dir outputs/custom/jigsaw
```

change `--task`, manifests and `--recipe r1`–`r6` for other runs.

## Train the LLaVA transfer runs

Run the separate transfer configuration with the same adapter.

```bash
python scripts/run_experiment.py --adapter my_umm.adapter:create_adapter --config configs/experiments/transfer.json --data-root data/prepared --model-path /path/to/model --nproc-per-node 1 --output-dir outputs/custom
```

## Evaluate on OmniTaskonomy

Read the adapter and checkpoint settings from a completed run and produce benchmark scores. For the full transfer matrix, repeat for every job and use the [collection and analysis commands](reproduce_bagel.md#transfer-matrix) with your output paths.

```bash
python scripts/evaluate.py --run-file outputs/custom/transfer/i2t_baseline/checkpoints.json --seeds 42 43 44 --output-dir outputs/custom-benchmarks/i2t_baseline
```