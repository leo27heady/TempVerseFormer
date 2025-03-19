# TempVerseFormer: Memory-Efficient Temporal Modeling with Reversible Transformers

[![GitHub Code](https://img.shields.io/github/v/release/leo27heady/TempVerseFormer?label=TempVerseFormer&style=flat-square)](https://github.com/leo27heady/TempVerseFormer)
[![Shape Dataset Toolbox](https://img.shields.io/github/v/release/leo27heady/simple-shape-dataset-toolbox?label=shapekit&style=flat-square)](https://github.com/leo27heady/simple-shape-dataset-toolbox)
[![WandB Log Examples](https://img.shields.io/badge/WandB-Training%20Logs-blue?style=flat-square&logo=wandb)](https://wandb.ai/leo27heady)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Trained%20Models-blue?style=flat-square&logo=huggingface)](https://huggingface.co/LKyluk/TempVerseFormer)

This repository contains the code and resources for the research article: **"Temporal Modeling with Reversible Transformers"**. This work introduces **TempVerseFormer**, a novel deep learning architecture designed for memory-efficient temporal sequence modeling. TempVerseFormer leverages reversible transformer blocks and a time-agnostic backpropagation strategy to decouple memory consumption from temporal depth, enabling efficient training on long prediction horizons.

**Key Features:**

* **Memory Efficiency:** Achieves near-constant memory footprint regardless of prediction time, enabling training on long sequences.
* **Reversible Architecture:** Utilizes reversible transformer blocks inspired by RevViT to eliminate the need for storing intermediate activations.
* **Time-Agnostic Backpropagation:** Implements a novel backpropagation method that reconstructs activations on-demand, further enhancing memory efficiency.
* **Temporal Chaining:** Employs a feedback mechanism inspired by Bytelatent Transformer to iteratively predict future states, allowing for long-range temporal modeling.
* **Synthetic Dataset:** Provides a procedurally generated dataset of rotating 2D shapes for controlled experiments and evaluation of temporal modeling capabilities.
* **Performance Benchmarking:** Includes code and scripts for evaluating TempVerseFormer against the baselines.

Below you'll find instructions on how to set up the environment, train models, evaluate performance, and reproduce the memory efficiency experiments presented in the article.

## Requirements

* **Python:** 3.12
* **GPU:** NVIDIA GPU with CUDA drivers (>= 12.0)

## Setup
*Note: For Windows, use Git Bash terminal.*

1. **Initialize Virtual Environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment:**
   ```bash
   # On Linux/macOS:
   source venv/bin/activate

   # On Windows:
   source venv/Scripts/activate
   ```

3. **Run Setup Script:**
   *(For Windows, use Git Bash terminal)*
   ```bash
   bash setup.sh
   ```
   This script installs all necessary Python packages.

4. **Set WandB API Key (Optional):**
   If you wish to log training runs to Weights & Biases (WandB), create a `.env` file in the repository root and add your API key (can be found [here](https://wandb.ai/authorize)):
   ```
   WANDB_API_KEY="YOUR_WANDB_API_KEY"
   ```
   If you don't have a WandB API key or don't want to use WandB logging, you can skip this step. The code will run without WandB if the key is not provided.

## Train

To train a model, execute the `train.py` script, providing the desired configuration group as an argument. Configuration files are located in the `configs/train` directory.

**Example: Train TempVerseFormer with the `rev-transformer` configuration:**
```bash
python train.py --config_groups rev-transformer
```

Refer to the configuration files in `configs/train` to configure specific training parameters and model settings.

**Training Logs:**

If you have set up WandB, training progress, metrics, and visualizations will be automatically logged to your WandB project.

## Evaluate

The `eval.py` script is used to evaluate the performance of trained models. Evaluation configurations are located in the `configs/eval` directory.

**Example: Evaluate TempVerseFormer with the `rev-transformer` evaluation configuration:**
```bash
python eval.py --config_groups rev-transformer
```

**Available Evaluation Configurations:**

The `configs/eval` directory contains YAML configuration files corresponding to the training configurations, but set up for evaluation. These configurations specify:

* Paths to pre-trained temporal and VAE models. *(You can download pre-trained models from [Hugging Face Hub](https://huggingface.co/LKyluk/TempVerseFormer) and place them in the `trained_models` directory as described in the configuration files).*
* Number of evaluation steps.

**Evaluation Metrics:**

The `eval.py` script calculates and saves the following metrics: FID, MSE and SSIM.
Evaluation results, including metrics and generated images, are saved in the `eval_results` directory.

## Memory Efficiency Test

To reproduce the memory efficiency experiments and measure the memory footprint of different models, use the `memory_test.py` script.

**Run Memory Test:**
```bash
python memory_test.py
```

**Script Arguments:**

The `memory_test.py` script accepts arguments to customize the test:

* `--temp_models`: Specify which temporal models to test (e.g., `--temp_models rev_transformer vanilla_transformer`). Don't provide to test all temporal models.
* `--vae_models`: Specify which VAE models to test (e.g., `--vae_models vae`). Don't provide to test all VAE models.
* `--batches`: Specify batch sizes to test (e.g., `--batches 1 8 64`). Don't provide to replicate the article configuration.
* `--time_steps`: Specify prediction time steps to test (e.g., `--time_steps 1 16 256`). Don't provide to replicate the article configuration.

**Memory Test Results:**

The `memory_test.py` script runs memory tests for various model combinations, batch sizes, and time steps. The results, including GPU memory consumption, estimated training time and models' capacity calculated by `calflops` are saved in JSON format in the `eval_results/memory` directory.

## Pre-trained Models

Pre-trained models for TempVerseFormer and baseline architectures are available on the Hugging Face Hub: [https://huggingface.co/LKyluk/TempVerseFormer](https://huggingface.co/LKyluk/TempVerseFormer).

Download the desired model checkpoints and place them in the `trained_models` directory, following the paths specified in the evaluation configuration files (`configs/eval`).

## Acknowledgements

This work builds upon and adapts code from the following repositories:

* **[RevViT](https://github.com/karttikeya/minREV) (Reversible Vision Transformer)**
* **[DiT-PyTorch](https://github.com/explainingai-code/DiT-PyTorch) (VAE model)**
