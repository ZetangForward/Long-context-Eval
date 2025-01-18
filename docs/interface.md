# User Guide

This document details the interface exposed by `longtexteval` and provides details on what flags are available to users.

## Command-line Interface

A majority of users run the library by cloning it from Github, installing the package as editable, and running the `python -m longtexteval` script.

Equivalently, running the library can be done via the `longtexteval` entrypoint at the command line.

This mode supports a number of command-line arguments, the details of which can be also be seen via running with `-h` or `--help`:

- `--model_path` : Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used. 

- `device_split_num`:each task needs the number of gpu

- `--server` : Selects which reasoning frame , Must be a string `[transformers,vllm]`

- `--acceleration` : Selects which acceleration frame , Must be a string:`[duo_attn,Snapkv,streaming_llm,ms-poe,selfextend,lm-inifinite(llama,gpt-j,mpt)，activation_beacon,tova],现在别用

- `--batch_size` : Sets the batch size used for evaluation.

- `--device` : It is used to set the device on which the model will be placed. The input must be a string, such as `0,1,2,3`.

- `--limit` : Accepts an integer,or just `auto` .it will limit the number of documents to evaluate to the first X documents (if an integer) per task of all tasks with `auto`

- `--task_name` : you can choose from the following options: 1. Benchmark, which may involve tasks such as L-cite-eval; 2. Task, like L-CiteEval-Data_narrativeqa; 3. Ability, for example General Capability. Multiple tasks can be specified, with each subject separated by a comma."

- `--torch_dtype` : Specifies the PyTorch data type that you want to use. 

- `--save_path` : The final results will be saved in the save_path.

- `--eval` : If the --eval option is not provided, the process will terminate after the model generation and before the evaluation.


- `--model_deploy_details_path` : Sets a path for saving the details regarding model deployment.


- `--selfextend_window_size` : the params of selfextend
- `--selfextend_group_size` : the params of selfextend
- `--tova_multi_state_size` :  the params of tova

for example:
python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct  --eval --tasks_name LongBench  --device 0,1,2,3,4,5,6,7 --device_split_num 4 --limit 10