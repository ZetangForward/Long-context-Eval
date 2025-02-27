# Reasoning
[[中文版](README_ZH.md)] [[English](README.md)]
## Counting_Stars

### Data Downloading

Please download the data "Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl ...." from the website "https://raw.githubusercontent.com/nick7nlp/Counting-Stars/refs/heads/main/test_data/" Use the following dictionary as the mapping for file names:

```python
task_download_name = {
    "counting_stars_en_reasoning.jsonl": "Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
    "counting_stars_en_searching.jsonl": "Counting_Stars_EN_multi-evidence-retrieval-searching_128000_32_32.jsonl",
    "counting_stars_zh_reasoning.jsonl": "Counting_Stars_ZH_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
    "counting_stars_zh_searching.jsonl": "Counting_Stars_ZH_multi-evidence-retrieval-searching_128000_32_32.jsonl"
}
```
Then save the downloaded data in the folder "/tasks/Reasoning/Counting_Stars/tmp_Rawdata".

### Example Code

1. Counting_Stars

## Example Code

```python
lte.run --model_path "meta-llama/Llama-3.1-8B-Instruct" --eval --benchmark_config ./tasks/Reasoning/Counting_Stars/Counting_Stars.yaml --device 2,5,6,7 --save_tag "tag"
```
or 

```bash
python lte/main.py--model_path "meta-llama/Llama-3.1-8B-Instruct" --eval --benchmark_config ./tasks/Reasoning/Counting_Stars/Counting_Stars.yaml --device 2,5,6,7 --save_tag "tag"
```

