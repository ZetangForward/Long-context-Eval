
# Long TexT Evaluation (lte)
An open source framework for evaluating foundation models

## Overview

This project provides a unified framework to test long-text-language models on a large number of different evaluation tasks.

**Features:**

- Over 5 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including information retrieval implemented via [BM25], [Raptors],[opeenai])(https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG), [Snapkv](https://github.com/FasterDecoding/SnapKV), [SelfExtend](https://github.com/datamllab/LongLM), and [streaming_llm](https://github.com/mit-han-lab/streaming-llm) to accelerate model inference, 
- Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Support for local models and benchmarks.
- Evaluation with publicly available prompts ensures reproducibility and comparability between papers.
- Easy support for custom prompts and evaluation metrics.

## Install

To install the `lte` package from the github repository, run:

```bash
git clone https://gitlab.com/nlp_sora/lte.git
cd lte
conda create -n lte python=3.10 -y
conda activate lte
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. A detailed table is available at the end of this document.

# Basic Usage

### User Guide

A user guide detailing the full list of supported arguments is provided [here](./lte/interface.md), and on the terminal by calling `lte -h`. Now you can evaluate these benchmarks: ["RULER","LooGLE","Counting_Stars","L_CiteEval"] . Task descriptions and links to corresponding subfolders are provided [here](./tasks/README.md).

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models),(e.g. GPT-J-6B) on `RULER` and  you can use the following command (this assumes you are using a CUDA-compatible GPU)

RUN

```bash

lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/General/RULER/RULER.yaml \
    --device 0 \
or

python lte/main.py --model_path meta-llama/Llama-3.1-8B-Instruct   --eval  --benchmark_configs tasks/Factuality/L_CiteEval/L_CiteEval.yaml   --device 0  

``` 
> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

#### Multi-GPU Evaluation with Hugging Face `accelerate`

```
```
To evaluate a model hosted on some tasks, you can modify the configuration file in tasks/{self.ability}/{self.benchmark_name}/
```
To evaluate a model hosted on the on `narrativeqa` from `LongBench` you can use the following command (this assumes you are using a CUDA-compatible GPU):

If you don't select --eval, you need to perform the eval

```bash
lte.eval --prediction_time 
```
