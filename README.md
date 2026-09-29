<h1 align="center">OmniTaskonomy: When Does Visual Generation<br>Improve Visual Understanding</h1>

<p align="center">
<a href="https://gejiaxin.org/">Jiaxin Ge</a><sup>1*</sup> &nbsp; <a href="https://wakals.github.io/">Yiming Qin</a><sup>2,6*</sup> &nbsp; <a href="https://horizonwind2004.github.io/">Ji Xie</a><sup>3</sup> &nbsp; <a href="https://astro-eric.github.io/">Haozhe Jiang</a><sup>1</sup> &nbsp; <a href="https://xhan77.github.io/">Xiaochuang Han</a><sup>4</sup><br>
<a href="https://www.junyi42.com/">Junyi Zhang</a><sup>1</sup> &nbsp; <a href="https://github.com/a-dai">Andrew M. Dai</a><sup>5</sup> &nbsp; <a href="https://sites.google.com/site/yinfeiyang">Yinfei Yang</a><sup>5</sup> &nbsp; <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a><sup>1</sup> &nbsp; <a href="https://ranjaykrishna.com/">Ranjay Krishna</a><sup>4</sup> &nbsp; <a href="https://www.sewonmin.com/">Sewon Min</a><sup>1</sup><br>
<a href="https://havenfeng.github.io/">Haiwen Feng</a><sup>1,6†</sup> &nbsp; <a href="https://www.salesforce.com/blog/author/le-xue/">Le Xue</a><sup>5†</sup> &nbsp; <a href="https://bfshi.github.io/">Baifeng Shi</a><sup>1†</sup> &nbsp; <a href="https://people.eecs.berkeley.edu/~trevor/">Trevor Darrell</a><sup>1†</sup> &nbsp; <a href="https://xudongfrankwang.github.io/">XuDong Wang</a><sup>1,2†</sup>
</p>

<p align="center">
<sup>1</sup>University of California, Berkeley &nbsp; <sup>2</sup>Duke University &nbsp; <sup>3</sup>Carnegie Mellon University<br>
<sup>4</sup>University of Washington &nbsp; <sup>5</sup>Elorian &nbsp; <sup>6</sup>Impossible, Inc.
</p>

<p align="center">
*, † Equal contribution.<br>
</p>

<p align="center">
  <a href="https://omni-taskonomy.github.io/"><img src="https://img.shields.io/badge/Project-Page-2563EB?style=for-the-badge" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-Coming_soon-8B8F98?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Arxiv — Coming soon" title="arXiv link pending; the paper PDF is linked below">
  <a href="https://huggingface.co/datasets/Wakals/OmniTaskonomy"><img src="https://img.shields.io/badge/Hugging_Face-OmniTaskonomy-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Huggingface OmniTaskonomy"></a>
</p>

<p align="center">
  <img src="docs/images/teaser.gif" width="100%" alt="I2I reconstructs the Jigsaw image, a dot moves from Visual Generation to Visual Understanding while changing color, and I2T reveals the patch order [3, 2, 1, 0]. All elements then return smoothly to the starting state.">
</p>
<p align="center">
  <a href="#about">About</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#citation">Citation</a>
</p>

## About

OmniTaskonomy studies when visual generation improves visual understanding in unified multimodal models.

### Findings

1. **I2I pretraining provides a useful initialization for I2T learning** when the initial stage updates parameters shared by both objectives.
2. **Generation supervision produces significant gains in specific understanding capabilities**, through transfer between related tasks as well as across tasks.
3. **Gradients align most strongly in early pre-attention normalization layers.** This alignment is positively associated with downstream transfer across understanding capabilities and individual source–target pairs.

<p align="center">
  <img src="docs/images/findings.png" width="100%" alt="Three findings: I2I pretraining improves subsequent I2T learning; transfer depends on the generation and understanding task pair; gradient alignment correlates with downstream transfer.">
</p>

<!-- See the [paper PDF](https://omni-taskonomy.github.io/paper.pdf) and [project page](https://omni-taskonomy.github.io/) for the experiments and results. -->

### Structure of OmniTaskonomy

OmniTaskonomy organizes **19 generation tasks** and **25 understanding capabilities** into Recognition, Reconstruction, and Reorganization.

<p align="center">
  <img src="docs/images/omnitaskonomy.png" width="100%" alt="OmniTaskonomy's three families: Recognition identifies semantic content; Reconstruction recovers geometry and appearance; Reorganization groups, locates, and relates visual elements. Each family contains generation tasks and understanding capabilities.">
</p>

## Quick Start

### Setup

Our experiemtns are conducted on Linux with conda and a CUDA toolkit compatible with PyTorch 2.5.1. Quick setup:

```bash
conda create -n omnitaskonomy python=3.10 -y
conda activate omnitaskonomy
bash scripts/install.sh
export LMUData="$PWD/data/raw/benchmarks"
export OMNI_MODEL="$PWD/checkpoints/BAGEL-7B-MoT"
```

All commands accept `--help`. BAGEL training defaults to four GPUs.

### Reproduce BAGEL as baseline

Follow the [BAGEL guide](docs/reproduce_bagel.md) for model weights, data, recipe training, benchmark evaluation, the transfer matrix, and gradient analysis.

### Test your own UMM

Follow the [custom UMM guide](docs/custom_umm.md) to run the same training, evaluation, and gradient experiments on OmniTaskonomy.

## Citation

```bibtex
% BibTeX pending.
```

Dataset license: [source licenses](https://huggingface.co/datasets/Wakals/OmniTaskonomy/blob/main/LICENSE.md). 

We greatly thank [BAGEL](Bagel/LICENSE), [VLMEvalKit](VLMEvalKit/LICENSE) for their open-source code.

If you have any question or idea to discuss with us, feel free to contact: Jiaxin Ge ([gejiaxin01@gmail.com](mailto:gejiaxin01@gmail.com)) and Yiming Qin ([ymk4474@gmail.com](mailto:ymk4474@gmail.com))!
