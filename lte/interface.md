# User Guide

This document details the interface exposed by lte and provides information on the available flags for users.

## Command-line Interface

This mode supports a number of command-line arguments, the details of which can be also be seen via running with `-h` or `--help`:

- `--model_path` : Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used.

- `rag`: ["",'openai', 'BM25', 'llamaindex', 'contriever'],help="Selects the model type or provider for information retrieval.  Each option corresponds to a different approach for retrieving relevant information."

- `device_split_num`:each task needs the number of gpu

- `--server`: Selects which reasoning frame , Must be a string `[transformers,vllm]

- `--acceleration` : Selects which acceleration frame , Must be a string:`[Snapkv,streaming_llm,selfextend]

- `--device` : It is used to set the device on which the model will be placed. The input must be a string, such as `0,1,2,3`.

- `--limit` : Accepts an integer,or just `auto` .it will limit the number of documents to evaluate to the first X documents (if an integer) per task of all tasks with `auto`

- `--benchmark` : Specify the benchmark. You should provide a value in the format like '--benchmark task/configs'.

- `--torch_dtype` : Specifies the PyTorch data type that you want to use. 

- `--adapter_path` : The path to the adapter file

- `--eval` : If the --eval option is not provided, the process will terminate after the model generation and before the evaluation.


- `--selfextend_window_size` : the params of selfextend
- `--selfextend_group_size` : the params of selfextend
- `--tova_multi_state_size` :  the params of tova

