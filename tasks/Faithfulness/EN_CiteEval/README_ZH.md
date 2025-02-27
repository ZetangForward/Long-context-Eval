# Faithfulness
[[中文版](README_ZH.md)] [[English](README.md)]
## EN_CiteEval

### metric

"tasksource/deberta-base-long-nli":
请下载 "tasksource/deberta-base-long-nli" 模型到本地，以确认其可以正常进行评估。

### 数据下载
如果无法连接到 Hugging Face，可以尝试以下代码。

```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```

或者

请从网站 "(https://hf-mirror.com/datasets/ZetangForward/EN_CiteEval/tree/main/data)" 下载数据，并直接将文件保存在路径 "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata"。

### 示例代码
EN_CiteEval

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/EN_CiteEval/EN_CiteEval.yaml --device 0,1 --save_tag "Llama-3-8B-NExtLong"
```
或者

```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/EN_CiteEval/EN_CiteEval.yaml --device 0,1 --save_tag "Llama-3-8B-NExtLong"
```
