# LongTextEval

An open source framework for evaluating foundation models

## Overview

This project provides a unified framework to test long-text-language models on a large number of different evaluation tasks.

**Features:**

- Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- Easy support for custom prompts and evaluation metrics.

## Install

To install the `longtexteval` package from the github repository, run:

```bash
git clone https://gitlab.com/nlp_sora/longtexteval.git
cd longtexteval
conda create -n lte python=3.10 -y
conda activate lte
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. A detailed table is available at the end of this document.

# Basic Usage

### User Guide

A user guide detailing the full list of supported arguments is provided [here](./docs/interface.md), and on the terminal by calling `lte -h`.

A list of supported tasks (or groupings of tasks) can be viewed with `lm-eval --benchmark_names list` Now you can evaluate these benchmarks: ["RULER","LooGLE","Counting_Stars","L_CiteEval"] . Task descriptions and links to corresponding subfolders are provided [here](./tasks/README.md).

### Hugging Face `transformers`

To evaluate a model hosted on the on `LongBench` and `Counting_Stars` you can use the following command (this assumes you are using a CUDA-compatible GPU):

RUN

```bash
lte.run --model_path /opt/data/private/models/Llama-2-7B-32K-Instruct\
    --eval\
    --benchmark_config  tasks/task_configs.yaml\

lte.run --model_path /opt/data/private/models/Llama-2-7B-32K-Instruct \
    --eval \
    --benchmark_config   \
    --device 0,1 \
    --device_split_num 1 \
    --limit 1 
or
python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct/   --eval    --benchmark_config tasks/task_configs.yaml   --device 0,1     --device_split_num 2   --limit 1   --device 0,1 --device_split_num 2 --limit 1
``` 
To evaluate a model hosted on some tasks, you can modify the configuration file in benchmark_config。
and
python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct  --eval --benchmark_names LongBench,Counting_Stars --device 0,1 --device_split_num 2 --limit 1
```

To evaluate a model hosted on the on `narrativeqa` from `LongBench` you can use the following command (this assumes you are using a CUDA-compatible GPU):

RUN

```bash
lte.run --model_path /opt/data/private/models/Llama-2-7B-32K-Instruct \
    --eval \
    --benchmark_names  LongBench:narrativeqa \
    --device 0,1 \
    --device_split_num 1 \
    --limit 1 
or
python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct  --eval --benchmark_names LongBench:narrativeqa --device 0,1 --device_split_num 2 --limit 1
```

If you don't select --eval, you need to perform the eval

```bash
lte.eval --generation_path '' --output_path ''
```
