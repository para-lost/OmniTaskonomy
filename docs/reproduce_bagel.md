# Reproduce BAGEL as baseline

Complete the [environment setup](../README.md#quick-start) once, then run these commands from the repository root. BAGEL training defaults to four GPUs; inference and gradient extraction load one model per GPU.

## Model

Download the base weights, tokenizer and VAE:

```bash
python scripts/download_model.py --output-dir "$OMNI_MODEL"
```

## OmniTaskonomy data

[Wakals/OmniTaskonomy](https://huggingface.co/datasets/Wakals/OmniTaskonomy) provides one split per leaf task, named `family__task__i2t` or `family__task__i2i` within the corresponding config:

- `i2t`: 9,444 evaluation samples across 25 tasks, with `usage="val"`.
- `i2i`: 350,000 training samples across seven transfer sources: Object editing, Attribute editing, Inpainting, Semantic segmentation, Object pointing, Jigsaw, and Localization. Each has 50,000 rows with `usage="train"`.


See the dataset's [LICENSE.md](https://huggingface.co/datasets/Wakals/OmniTaskonomy/blob/main/LICENSE.md) for source-specific terms.

### Taskonomy data

Restore the twelve Taskonomy transfer pools from official sources. This writes `data/prepared/<task>/train.jsonl`, preserving sample order and repetitions and checking rendered pixel hashes.

```bash
python scripts/prepare_taskonomy.py
```

Data use follows the [Taskonomy](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE) and [Omnidata](https://github.com/EPFL-VILAB/omnidata/blob/main/LICENSE) terms.

### Recipe data

[Wakals/OmniTaskonomy_Recipe_Data](https://huggingface.co/datasets/Wakals/OmniTaskonomy_Recipe_Data) contains six paired I2I/I2T subsets: `jigsaw`, `zoomin`, `video_unshuffle`, `rotate_qa`, `counting`, and `visgym_colorization`. Each has `train` and `val` splits.

Training with `--suite controlled` downloads missing recipe manifests automatically. To prepare all six tasks and both splits in advance:

```bash
python scripts/prepare_recipe.py
```

Use `--tasks jigsaw zoomin --splits train` to download only those pools. This writes `data/prepared/<task>/<split>/{i2i,i2t}.jsonl` and images, preserving prompts and image bytes.

## Train

Train Jigsaw R2 (I2I --> I2T) at 30k, including its shared I2I stage. Checkpoints, sample counts and `checkpoints.json` are saved under `outputs/bagel/EXPERIMENT/JOB/`.

```bash
python scripts/run_experiment.py --config configs/experiments/controlled_scaling.json --data-root data/prepared --model-path "$OMNI_MODEL" --output-dir outputs/bagel --jobs jigsaw_r2_30k
```

`--dry-run` prints the plan without training. `--jobs` selects jobs and includes their dependencies; omit it to run the complete configuration.

<details>
<summary>Recipes</summary>

- R1: I2T only. 
- R2: I2I → I2T. 
- R3: Mixed → I2T.
- R4: Frozen I2I → Mixed.
- R5: Mixed. 
- R6: I2I → Mixed.

</details>
<details>

<summary>Experiment configurations</summary>

- [`controlled_scaling.json`](../configs/experiments/controlled_scaling.json): Jigsaw/Zoom-In, R1–R6, four I2I budgets, data seeds 42/123/456.
- [`controlled_i2t_scaling_three_seed.json`](../configs/experiments/controlled_i2t_scaling_three_seed.json): four I2T pool sizes, 30k visits, three seeds.
- [`transfer.json`](../configs/experiments/transfer.json): 19 I2I sources and the LLaVA baseline, seeds 42/43/44.
- [`instance_paper_15ep.json`](../configs/experiments/instance_paper_15ep.json): 15 epochs with the standard trainable scope.
- [`gradient_checkpoints.json`](../configs/experiments/gradient_checkpoints.json): independent Jigsaw/Zoom-In I2I runs at 3k/10k/30k.

</details>

## Benchmark evaluation

Evaluate completed transfer runs on OmniTaskonomy:

```bash
bash scripts/evaluate_transfer.sh outputs/bagel/transfer outputs/benchmarks
```

Follow the instruction in VLMEvalKit to configure `OPENAI_API_KEY` in `.env`. The default answer judge is `chatgpt-0125` (`gpt-3.5-turbo-0125`), and API failures fall back to exact matching and are recorded in per-question logs and evaluation metadata.

## Transfer matrix

The complete transfer experiment:

```bash
python scripts/collect_transfer_inputs.py --training-root outputs/bagel/transfer --evaluation-root outputs/benchmarks --output outputs/transfer_inputs.json
```

Compute gains and paired tests over 9,444 retained questions:

```bash
python scripts/analyze_transfer.py --config outputs/transfer_inputs.json --seeds 42 43 44 --output-dir outputs/transfer
```

## Gradients by module and layer

Measure paired I2I/I2T gradient cosine similarity on the six recipe tasks.

```bash
python scripts/analyze_gradients.py modules --model-path "$OMNI_MODEL" --output outputs/gradients/modules
```

## Gradients across the transfer matrix

```bash
python scripts/analyze_gradients.py matrix --config data/prepared/gradients/transfer.json --reference outputs/gradients/modules --model-path "$OMNI_MODEL" --output outputs/gradients/transfer
```
