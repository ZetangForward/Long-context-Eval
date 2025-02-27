# Long TexT Evaluation (lte)

An open source framework for evaluating foundation models

[[中文版](README_ZH.md)] [[English](README.md)]

## Overview

This project provides a unified framework to test long-text-language models on a large number of different evaluation tasks.

**Features:**

- Over 5 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/),including information retrieval methods such as [BM25](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG), [Raptors](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG), [opeenai](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG), as well as accelerated model inference methods like [Snapkv](https://github.com/FasterDecoding/SnapKV), [SelfExtend](https://github.com/datamllab/LongLM) and [streaming_llm](https://github.com/mit-han-lab/streaming-llm).
- Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace&#39;s PEFT library](https://github.com/huggingface/peft).
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
pip install torch==2.5.1
pip install -e .
# Download the appropriate version of flash_attn from the following link
Download the suitable version of flash_attn from https://github.com/Dao-AILab/flash-attention/releases
pip install <path_to_flash_attn_whl_file>
```

> [!Note] Since we have a large number of Python packages, to prevent conflicts, pyproject.toml only contains the most basic functionality by default. If you encounter a missing Python package during runtime, please install it directly. Alternatively, you can try installing all the packages in requirements.txt.

# Basic Usage

### User Guide

A user guide detailing the full list of supported arguments is provided [here](./lte/interface.md), and on the terminal by calling `lte -h`.  Task descriptions and links to corresponding subfolders are provided [here](./tasks/README.md).
If you want to add a task to the framework, please read the doc[here](./tasks/ADD.md).

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models),(e.g. GPT-J-6B) on `RULER` and  you can use the following command (this assumes you are using a CUDA-compatible GPU)

RUN

```bash

lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    # The '--eval' option indicates that the framework should not only generate data but also perform evaluations.
    --eval \
    # The '--benchmark_config' option is used to provide the path(s) to the configuration file(s) for the evaluation tasks. Multiple paths can be specified and separated by commas.
    --benchmark_config tasks/General/RULER/RULER.yaml \
    # The '--device' option specifies the device (e.g., GPU) on which the evaluation will be carried out.
    --device 0 \
    # The '--save_tag' option determines the name of the folder where the evaluation results will be saved.
    --save_tag "tmp";
or

python lte/main.py --model_path meta-llama/Llama-3.1-8B-Instruct   --eval  --benchmark_config tasks/Factuality/L_CiteEval/L_CiteEval.yaml   --device 0  

```

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lte` via `--model_path pretrained=/path/to/model`

To evaluate a model on specific tasks within a benchmark, you can modify the configuration file located at tasks/{ability}/{benchmark_name}/

if you do not select the --eval option during the initial run, you can perform the evaluation separately using
the following command:

```bash
## --folder_name the folder name that saves the result such as Llama-3.1-8B-Instruct_02M_04D_14H_53m 
lte.eval --folder_name  Llama-3.1-8B-Instruct_02M_04D_14H_53m  --model_name your_model_name 
or 
python lte/eval.py --folder_name Llama-3.1-8B-Instruct_02M_04D_14H_53m   --model_name your_model_name --benchmark_config tasks/General/RULER/RULER.yaml

```

Here, --prediction_time represents the duration of your previous run. This function will evaluate all benchmark results generated during that run.

When using it, there are more relevant parameters, such as

- `save_tag`: "Optional parameter. Use this to specify the name of the file to be saved."
- `torch_dtype`: Specifies the PyTorch data type that you want to use.default="torch.bfloat16"
- `limit`: accepts an integer,or just `auto` .it will limit the number of documents to evaluate to the first X documents.default="auto"
- `adapter_path`: the adapter_path parameter specifies the location of an adapter model.
- `template`:  specifies adding a template to the data for the model to generate, for example: '''[INST] Below is a context and an instruction.Context:{YOUR LONG CONTEXT}Instruction:{YOUR QUESTION & INSTRUCTION} [/INST]''
- `max_lenth`: The maximum length. It refers to the maximum length of data.If the length exceeds this limit, the data will be split in the middle.
- `device_split_num`: each task needs the number of gpu.default=1

### Framework and Acceleration Options

You can use --server to select the inference framework to be used,--rag to decide whether to enable information retrieval and choose the retrieval system and  --acceleration to determine whether to enable an acceleration framework and select the specific acceleration method.

#### 1. rag

Our framework now incorporates information retrieval capabilities, offering four different methods for users to choose from: BM25, contriever, llamaindex, and openai. Each of these methods provides unique advantages for retrieving relevant information from large datasets. However, it is important to note that these retrieval methods can be relatively slow in terms of processing speed.

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_config tasks/General/RULER/RULER.yaml \
    --device 0 \
    --rag BM25
```

> [!Note]
> the LlamaIndex method currently supports only single-GPU environments

#### 2. server

We support various model deployment frameworks, such as vllm and transformers,mamba. These frameworks offer different advantages for deploying and running large language models (LLMs) efficiently.
For example, running vLLM is as follows.

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    # The '--eval' option indicates that the framework should not only generate data but also perform evaluations.
    --eval \
    # The '--benchmark_config' option is used to provide the path(s) to the configuration file(s) for the evaluation tasks. Multiple paths can be specified and separated by commas.
    --benchmark_config tasks/General/RULER/RULER.yaml \
    # The '--device' option specifies the device (e.g., GPU) on which the evaluation will be carried out.
    --device 0 \
    # The '--save_tag' option determines the name of the folder where the evaluation results will be saved.
    --save_tag "tmp";
    --server vllm
    --max_model_len 16000
    # The '--save_tag' option determines the maximum input length supported by vLLM. If not specified, the default setting of the model will be used.
or
> [!Note]
> Please note that the selected framework and model must be compatible with each other.

#### 3. acceleration

Our framework supports acceleration frameworks such as  [Snapkv](https://github.com/FasterDecoding/SnapKV), [SelfExtend](https://github.com/datamllab/LongLM), and [streaming_llm](https://github.com/mit-han-lab/streaming-llm) to accelerate model inference,  for accelerating model inference. It is important to note that each of these frameworks has strict requirements for the environment setup and model compatibility. Please configure your environment according to the official documentation(or our README file in ./models_deploy/server.) for each framework and ensure that your models are compatible.

> [!Note]
> After you have set up the environment properly, please uncomment the relevant code in ./models_deploy/__init__.py to use the acceleration frameworks.
```
