# Reasoning
[[中文版](README_ZH.md)] [[English](README.md)]
## Counting_Stars

### 数据下载

请从网站 "https://raw.githubusercontent.com/nick7nlp/Counting-Stars/refs/heads/main/test_data/" 下载数据"Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl ...."。请使用以下字典作为文件名的映射：

```python
task_download_name = {
    "counting_stars_en_reasoning.jsonl": "Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
    "counting_stars_en_searching.jsonl": "Counting_Stars_EN_multi-evidence-retrieval-searching_128000_32_32.jsonl",
    "counting_stars_zh_reasoning.jsonl": "Counting_Stars_ZH_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
    "counting_stars_zh_searching.jsonl": "Counting_Stars_ZH_multi-evidence-retrieval-searching_128000_32_32.jsonl"
}
```
然后将下载好的数据保存到文件 "/tasks/Reasoning/Counting_Stars/tmp_Rawdata"

### 示例代码

1. Counting_Stars

## 示例代码

```python
lte.run --model_path "meta-llama/Llama-3.1-8B-Instruct" --eval --benchmark_config ./tasks/Reasoning/Counting_Stars/Counting_Stars.yaml --device 2,5,6,7 --save_tag "tag"
```
或者 

```bash
python lte/main.py--model_path "meta-llama/Llama-3.1-8B-Instruct" --eval --benchmark_config ./tasks/Reasoning/Counting_Stars/Counting_Stars.yaml --device 2,5,6,7 --save_tag "tag"
```

