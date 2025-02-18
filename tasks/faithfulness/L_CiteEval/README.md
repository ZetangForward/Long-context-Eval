# Faithfulness

## L-CiteEval

### Data Downloading


### Example Code

L-CiteEval_Data

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/faithfulness/L_CiteEval/L_CiteEval.yaml --device 1,3,4,7 --save_tag "tag"
```

L-CiteEval_Length

L-CiteEval_Hardness
