# Reproduce the BAGEL baseline

Complete the [environment setup](../README.md#quick-start) and run the commands below from the repository root. BAGEL training defaults to four GPUs. Evaluation and gradient extraction load one model per GPU.

## Model weights

Download the base weights, tokenizer, and VAE to `OMNI_MODEL`:

```bash
python scripts/download_model.py --output-dir "$OMNI_MODEL"
```

## OmniTaskonomy data

### Paired recipe data

[Wakals/OmniTaskonomy_Recipe_Data](https://huggingface.co/datasets/Wakals/OmniTaskonomy_Recipe_Data) contains six paired I2I/I2T subsets: `jigsaw`, `zoomin`, `video_unshuffle`, `rotate_qa`, `counting`, and `visgym_colorization`. Each has `train` and `val` splits.

Training with `--suite controlled` downloads missing recipe manifests automatically. To prepare all tasks and splits in advance:

```bash
python scripts/prepare_recipe.py
```

The script saves `data/prepared/<task>/<split>/{i2i,i2t}.jsonl` and images with prompts and image bytes intact. Use `--tasks jigsaw zoomin --splits train` to prepare only those training subsets.

### Transfer data

[Wakals/OmniTaskonomy](https://huggingface.co/datasets/Wakals/OmniTaskonomy) contains:

| Dataset config | Contents | Usage |
| --- | --- | --- |
| `i2t` | 9,444 evaluation samples across 25 understanding capabilities. | `val` |
| `i2i` | 350,000 training samples across seven sources, with 50,000 per source. | `train` |

Within each config, splits are named `family__task__i2t` or `family__task__i2i`. The seven I2I sources are Object editing, Attribute editing, Inpainting, Semantic segmentation, Object pointing, Jigsaw, and Localization. Prepare them together with the fixed 50,000-example LLaVA pool:

```bash
python scripts/prepare_transfer.py
```

Use `--llava-json /path/to/llava_instruct_150k.json --coco-root /path/to/coco` to reuse local data. If COCO images are missing, the script downloads the full `train2017.zip` archive and extracts only the selected images.

The remaining twelve I2I sources come from Taskonomy. Restore these pools from the official sources with:

```bash
python scripts/prepare_taskonomy.py
```

Training manifests are saved to `data/prepared/<task>/train.jsonl`. Taskonomy preparation preserves sample order and repetitions, and checks the rendered pixel hashes.

Data use is subject to the [dataset licenses](https://huggingface.co/datasets/Wakals/OmniTaskonomy/blob/main/LICENSE.md) and the [Taskonomy](https://github.com/StanfordVL/taskonomy/blob/master/data/LICENSE) and [Omnidata](https://github.com/EPFL-VILAB/omnidata/blob/main/LICENSE) terms.

## Train

Train Jigsaw with R2 (I2I → I2T) at the 30k I2I budget, including the shared I2I stage:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/controlled_scaling.json \
  --data-root data/prepared \
  --model-path "$OMNI_MODEL" \
  --output-dir outputs/bagel \
  --jobs jigsaw_r2_30k
```

Checkpoints, sample counts, and `checkpoints.json` are saved under `outputs/bagel/<experiment>/<job>/`. Add `--dry-run` to preview the plan. The launcher includes dependencies for the selected jobs, or runs the full configuration if `--jobs` is omitted. The [six recipes](../README.md#finding-1-an-initial-i2i-training-stage-helps-i2t-learning) are described in the README.

<details>
<summary>Experiment configurations</summary>

| Configuration | Experiment |
| --- | --- |
| [`controlled_scaling.json`](../configs/experiments/controlled_scaling.json) | Jigsaw and Zoom-In, R1–R6, four I2I budgets, data seeds 42/123/456. |
| [`controlled_i2t_scaling_three_seed.json`](../configs/experiments/controlled_i2t_scaling_three_seed.json) | Four I2T pool sizes, 30k sample visits, three seeds. |
| [`transfer.json`](../configs/experiments/transfer.json) | 19 I2I sources and the LLaVA baseline, seeds 42/43/44. |
| [`instance_paper_15ep.json`](../configs/experiments/instance_paper_15ep.json) | 15 epochs of I2T training. |
| [`gradient_checkpoints.json`](../configs/experiments/gradient_checkpoints.json) | Independent Jigsaw and Zoom-In I2I runs at 3k/10k/30k. |

</details>

## Benchmark evaluation

Set `OPENAI_API_KEY` in `VLMEvalKit/.env`, then evaluate completed transfer runs on OmniTaskonomy:

```bash
bash scripts/evaluate_transfer.sh outputs/bagel/transfer outputs/benchmarks
```

The default answer judge is `chatgpt-0125` (`gpt-3.5-turbo-0125`). API failures fall back to exact matching and are recorded in the per-question logs and evaluation metadata.

## Transfer matrix

Collect the training records and benchmark scores from the completed transfer experiment:

```bash
python scripts/collect_transfer_inputs.py \
  --training-root outputs/bagel/transfer \
  --evaluation-root outputs/benchmarks \
  --output outputs/transfer_inputs.json
```

Compute gains and paired tests over the 9,444 retained questions:

```bash
python scripts/analyze_transfer.py \
  --config outputs/transfer_inputs.json \
  --seeds 42 43 44 \
  --output-dir outputs/transfer
```

## Gradients by module and layer

Measure gradient cosine similarity between paired I2I and I2T examples on the six recipe tasks:

```bash
python scripts/analyze_gradients.py modules \
  --model-path "$OMNI_MODEL" \
  --output outputs/gradients/modules
```

## Gradients across the transfer matrix

Prepare the matrix inputs from the training manifests above and the released I2T evaluation examples:

```bash
python scripts/prepare_gradient_matrix.py
```

These sample pools are rebuilt from the public release and differ from the historical frozen selection. Run the matrix analysis with the module results as its reference:

```bash
python scripts/analyze_gradients.py matrix \
  --config data/prepared/gradients/transfer.json \
  --reference outputs/gradients/modules \
  --model-path "$OMNI_MODEL" \
  --output outputs/gradients/transfer
```
