# Faithfulness

## Benchmarks

1. L-CiteEval



## Example Code

```python
lte.run --model_path "meta-llama/Llama-3.1-8B-Instruct" --eval --benchmark_configs tasks/faithfulness/L_CiteEval/L_CiteEval.yaml --device 2,5,6,7 --save_tag "tag"
```
