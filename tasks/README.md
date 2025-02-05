
# RUN 
You can use our framework by running the following code. For example, if you want to evaluate RULER, you can run the code below:

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/General/RULER/RULER.yaml \
```

Similarly, if you want to evaluate NIAH, you just need to change the value of the --benchmark_configs parameter, which is the path to the configuration file. You can run the following code:

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/Retrieve/NIAH/niah.yaml \
```

If you want to run both benchmarks simultaneously, you can run the following code:

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs tasks/Retrieve/NIAH/NIAH.yaml,tasks/General/RULER/RULER.yaml \
```

You can also create customized configuration files based on these existing ones. For example:

```bash
lte.run --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval \
    --benchmark_configs your_config_path \
```
We have categorized the benchmarks according to abilities, and the specific configuration files are placed in the tasks/Ability/benchmark_name directory.
# Ability
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

