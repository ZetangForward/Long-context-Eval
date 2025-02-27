# Faithfulness
[[中文版](README_ZH.md)] [[English](README.md)]
## EN_CiteEval

### metric

"tasksource/deberta-base-long-nli":
Download the "tasksource/deberta-base-long-nli" model to the local machine to confirm that it can be evaluated normally.

### Data Downloading
If you can't connect to Hugging Face, you can try the following code.

```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```

or

download the data from the website "(https://hf-mirror.com/datasets/ZetangForward/EN_CiteEval/tree/main/data)" Directly save the  file in the path "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata".

### Example Code
EN_CiteEval

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/EN_CiteEval/EN_CiteEval.yaml --device 0,1 --save_tag "Llama-3-8B-NExtLong"
```
or

```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/EN_CiteEval/EN_CiteEval.yaml --device 0,1 --save_tag "Llama-3-8B-NExtLong"
```
