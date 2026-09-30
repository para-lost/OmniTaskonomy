# Test your own UMM on OmniTaskonomy

Complete the [environment setup](../README.md#quick-start), then run the commands below from the repository root. The [data preparation guide](reproduce_bagel.md#omnitaskonomy-data) covers the transfer pools and paired recipe data.

## Add an adapter

Keep your model in an importable Python package and expose a factory such as `my_umm.adapter:create_adapter`. It receives `model_path`, `checkpoint`, `device`, and an `options` dictionary. The adapter's `model` should be the PyTorch module containing the parameters used by both objectives. See the [BAGEL adapter](../omnitaskonomy/adapters/bagel.py) for a pretrained model, or the [tiny CPU example](../omnitaskonomy/examples/tiny_umm.py) for a complete implementation that tests the interface.

Implement `loss(records, objective, context)` to prepare a batch and return the differentiable loss sum and its normalization count as `Loss(total, count)`. Training and gradient extraction call the same method. The context supplies the manifest path, random seed, conditioning dropout, and transform settings. Frozen benchmark records include `gradient_mcq`, which contains the exact prompt and correct option.

Describe each parameter with a `ParameterSpec` yielded by `parameter_specs()`. Its role is `generation`, `understanding`, `shared`, or `fixed`, and its `module` and `layer` fields determine how it is grouped for gradient analysis. A parameter with `module=None` is excluded from that analysis. Declare tied parameters once, assign LoRA parameters explicitly, and use `zero_objectives` for gradients that are absent by construction. Gradient analysis differentiates the parameters in the selected modules without updating their weights.

For benchmark evaluation, `generate(messages, **kwargs)` should accept ordered image/text messages and return a text answer. The adapter may also support image generation through `output="image"`.

Implement `save_checkpoint(directory)` so that the factory can reload the saved weights. List every loaded checkpoint, configuration, and tokenizer asset in `checkpoint_files` to record the model's provenance. The definitions of `Loss`, `LossContext`, and `ParameterSpec` are in [umm.py](../omnitaskonomy/umm.py).

## Train on Jigsaw or Zoom-In

The example below trains Jigsaw with R4. Its initial I2I stage updates only trainable `generation` parameters. The Mixed stage restores the set of parameters that were trainable when the adapter was loaded, while `fixed` parameters remain frozen throughout. Models without trainable generation-specific parameters cannot use R4.

```bash
python scripts/train.py \
  --adapter my_umm.adapter:create_adapter \
  --model-path /path/to/model \
  --suite controlled \
  --task jigsaw \
  --recipe r4 \
  --i2i-manifest data/prepared/jigsaw/train/i2i.jsonl \
  --i2t-manifest data/prepared/jigsaw/train/i2t.jsonl \
  --nproc-per-node 1 \
  --output-dir outputs/custom/jigsaw
```

For other runs, change `--task` and the manifests, and set `--recipe` to one of `r1` through `r6`.

## Train the LLaVA transfer runs

Run the transfer configuration with the same adapter:

```bash
python scripts/run_experiment.py \
  --adapter my_umm.adapter:create_adapter \
  --config configs/experiments/transfer.json \
  --data-root data/prepared \
  --model-path /path/to/model \
  --nproc-per-node 1 \
  --output-dir outputs/custom
```

## Evaluate on OmniTaskonomy

Evaluate a completed run using its saved adapter and checkpoint settings:

```bash
python scripts/evaluate.py \
  --run-file outputs/custom/transfer/i2t_baseline/checkpoints.json \
  --seeds 42 43 44 \
  --output-dir outputs/custom-benchmarks/i2t_baseline
```

For the full transfer matrix, repeat evaluation for every job and follow the [collection and analysis commands](reproduce_bagel.md#transfer-matrix) with your output paths.
