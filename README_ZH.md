# 长文本评估（lte）

lte是一个用于评估基础模型的开源框架

[[中文版](README_ZH.md)] [[English](README.md)]

## 概述

该项目提供了一个集成大量不同的评估任务的长文本语言模型评测框架。

**特征:**

- 超过5个LLM标准学术基准，包括数百个子任务和变体。
- 支持通过[transformer](https://github.com/huggingface/transformers/)加载的模型，包括[BM25](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG)、[Raptors](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG)、[opeenai](https://github.com/lixinze777/LC_VS_RAG/blob/main/RAG)信息检索方法，以及[Snapkv](https://github.com/FasterDecoding/SnapKV)、[SelfExtend](https://github.com/datamllab/LongLM)和[streaming_llm](https://github.com/mit-han-lab/streaming-llm)加速模型推理方法。
- 支持使用[vLLM](https://github.com/vllm-project/vllm)快速且内存高效的推理。
- 支持对[HuggingFace&#39;s PEFT library](https://github.com/huggingface/peft)中提供的适配器（例如 LoRA）进行评估。
- 支持本地模型和基准。
- 使用公开的提示进行评估可确保论文之间的可重复性和可比性。
- 支持自定义提示和评估指标。

## 安装

要从github存储库安装 `lte`，请运行：

```bash
git clone https://gitlab.com/nlp_sora/lte.git
cd lte
conda create -n lte python=3.10 -y
conda activate lte
pip install -e .
```

# 基本用法

### 用户指南

[此处](./lte/interface_ZH.md)提供了详细说明所支持参数完整列表的用户指南，也可以在终端上通过调用 `lte -h`来查看。[此处](./tasks/README_ZH.md)提供了任务描述和相应子文件夹的链接。如果您想向框架添加任务，请阅读[此处](./tasks/ADD_ZH.md)的文档。

### Hugging Face `transformers`

要在 `RULER`上评估托管在[HuggingFace Hub](https://huggingface.co/models)上的模型（例如 GPT-J-6B），您可以使用以下命令（假设您使用的是与 CUDA 兼容的 GPU）

运行

```bash

lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    # '--eval' 选项表示该框架不仅应生成数据，还应执行评估。
    --eval \
    # '--benchmark_config' 选项用于提供评估任务的配置文件路径。可以指定多个路径，并用逗号分隔。
    --benchmark_config tasks/General/RULER/RULER.yaml \
    # '--device' 选项用于指定执行评估的设备（例如 GPU）。
    --device 0 \
    # '--save_tag' 选项用于指定保存评估结果的文件夹名称。
    --save_tag "tmp";
or

python lte/main.py --model_path meta-llama/Llama-3.1-8B-Instruct   --eval  --benchmark_config tasks/Factuality/L_CiteEval/L_CiteEval.yaml   --device 0  

```

> [!Note]
> 就像您可以向 `transformers.AutoModel`提供本地路径一样，您也可以向 `lte`通过 `--model_path pretrained=/path/to/model`提供本地路径

要评估基准中的特定任务，您可以修改位于tasks/{ability}/{benchmark_name}/的配置文件

如果在初始运行期间没有选择--eval选项，则可以使用以下命令单独执行评估：

```bash
## --folder_name 选项用于指定保存结果的文件夹名称，例如 Llama-3.1-8B-Instruct_02M_04D_14H_53m 
lte.eval --folder_name  Llama-3.1-8B-Instruct_02M_04D_14H_53m  --model_name your_model_name 
or 
python lte/eval.py --folder_name Llama-3.1-8B-Instruct_02M_04D_14H_53m   --model_name your_model_name

```

其中，--prediction_time 表示您上次运行的时长。此函数将评估该运行期间生成的所有基准测试结果。

使用时，还有更多相关参数，例如

- `save_tag`：可选参数。使用它来指定要保存的文件的名称。
- `torch_dtype`：指定要使用的 PyTorch 数据类型。default="torch.bfloat16"
- `limit`：接受一个整数，或者只是 `auto`。它将限制要评估的文档数量为前 X 个文档。default=“auto”
- `adapter_path`：指定适配器模型的位置。
- `template`：指定在数据中添加模板，供模型生成，例如：'''[INST] Below is a context and an instruction.Context:{YOUR LONG CONTEXT}Instruction:{YOUR QUESTION & INSTRUCTION} [/INST]''
- `max_lenth`：最大长度，指数据的最大长度，如果长度超出此限制，数据将会在中间被分割。
- `device_split_num`：每个任务需要的gpu数量。default=1

### 框架和加速选项

您可以使用 --server 选择要使用的推理框架、--rag 决定是否启用信息检索并选择检索系统以及、--acceleration 确定是否启用加速框架并选择具体的加速方法。

#### 1. rag

我们的框架整合了信息检索功能，为用户提供了三种不同的方法：BM25、contriever和openai。每种方法都具有从大型数据集中检索相关信息的独特优势。但需要注意的是，这些检索方法的处理速度相对较慢。

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_config tasks/General/RULER/RULER.yaml \
    --device 0 \
    --rag BM25
```

> [!Note]
> LlamaIndex 方法当前仅支持单GPU环境

#### 2. server

我们支持各种模型部署框架，例如vllm和transformers、mamba。这些框架具有不同的优势来高效部署和运行大型语言模型(LLM)。

> [!Note]
> 请注意，所选的框架和模型必须相互兼容。

#### 3. acceleration

我们的框架支持加速框架，例如[Snapkv](https://github.com/FasterDecoding/SnapKV)，[SelfExtend](https://github.com/datamllab/LongLM)，和[streaming_llm](https://github.com/mit-han-lab/streaming-llm)来加速模型推理。需要注意的是，这些框架对环境设置和模型兼容性都有严格的要求。请根据每个框架的官方文档（或 ./models_deploy/server 中的 README 文件）配置您的环境，并确保您的模型兼容。

> [!Note]
> 正确设置环境后，请取消注释 ./models_deploy/init.py 中的相关代码以使用加速框架。
