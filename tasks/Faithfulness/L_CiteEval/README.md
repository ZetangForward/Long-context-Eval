# Faithfulness
[[中文版](README_ZH.md)] [[English](README.md)]
## L-CiteEval

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

download the data from the website "(https://huggingface.co/datasets/L4NLP/LEval/tree/main/LEval/Generation)" Directly save the JSON file in the path "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata".

### Example Code

L-CiteEval_Data

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 1,3,4,7 --save_tag "tag"
```

or

```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 1,3,4,7 --save_tag "tag"
```

L-CiteEval_Length

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Length.yaml --device 1,3,4,7 --save_tag "tag"
```

or

```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Length.yaml --device 1,3,4,7 --save_tag "tag"
```

L-CiteEval_Hardness

```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Hardness.yaml --device 1,3,4,7 --save_tag "tag"
```

or 

```bash 
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/Faithfulness/L_CiteEval/L_CiteEval_Hardness.yaml --device 1,3,4,7 --save_tag "tag"
```