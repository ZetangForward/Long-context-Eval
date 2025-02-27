# 用户指南

本文档详细介绍了 lte 公开的接口，并提供了用户可用的标志信息。

## 命令行接口

此模式支持多个命令行参数，详细信息可通过运行 `-h` 或 `--help` 查看。

- `--model_path` ：选择要评估的模型类型或提供者。必须是与所使用的模型类型/提供者名称对应的字符串。

- `rag`：["",'openai', 'BM25', 'llamaindex', 'contriever'],help="选择用于信息检索的模型类型或提供者。每个选项对应一种不同的信息检索方法。"

- `device_split_num`：每个任务需要的GPU数量。

- `--server`：选择推理框架，必须是一个字符串。 `[transformers,vllm]

- `--acceleration` ：选择加速框架，必须是一个字符串。`[Snapkv,streaming_llm,selfextend]

- `--device` ：用于设置评估模型需要使用的GPU。输入必须是一个字符串，例如： `0,1,2,3`.

- `--limit` ：接受一个整数，或者仅为 `auto`。如果是整数，它将限制每个任务评估的文档数量为前 X 个文档；如果为 `auto`，则代表整个任务。

- `--benchmark` ：指定基准。您应该提供类似 '--benchmark task/configs' 的格式值。

- `--torch_dtype` ：指定您要使用的 PyTorch 数据类型。

- `--adapter_path` ：适配器文件的路径。

- `--eval` ：如果未提供 --eval 选项，则程序将在模型生成后、评估前终止。


- `--selfextend_window_size` ：SelfExtend 的参数
- `--selfextend_group_size` ：SelfExtend 的参数
- `--tova_multi_state_size` ：Tova 的参数

