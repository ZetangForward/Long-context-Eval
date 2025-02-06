
# 运行 
您可以通过运行以下代码来使用我们的框架。例如，如果您想评估 RULER，您可以运行以下代码：

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/General/RULER/RULER.yaml \
```

同样，如果你想评估NIAH，你只需要改变 --benchmark_configs 参数的值，也就是配置文件的路径。你可以运行以下代码：

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/Retrieve/NIAH/niah.yaml \
```

如果您想同时运行两个基准测试，您可以运行以下代码：

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/Retrieve/NIAH/NIAH.yaml,tasks/General/RULER/RULER.yaml \
```

您还可以基于这些现有配置文件创建自定义配置文件。例如：

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs your_config_path \
```
我们按照能力对基准进行了分类，具体的配置文件放在 tasks/Ability/benchmark_name 目录中。
# 能力
## General
1. LooGLE
benchmark_configs: /tasks/General/LooGLE/LooGLE.yaml
2. LongBench
benchmark_configs: /tasks/General/LongBench/LongBench.yaml
3. RULER
benchmark_configs: /tasks/General/RULER/RULER.yaml
## Reasoing
1. Counting_Stars
benchmark_configs: /tasks/Reasoning/Counting_Stars/Counting_Stars.yaml
## Factuality
1. L_CiteEval
benchmark_configs: /tasks/Factuality/L_CiteEval/L_CiteEval.yaml
## Retrieve
1. NIAH
benchmark_configs: /tasks/Retrieve/NIAH/NIAH.yaml

