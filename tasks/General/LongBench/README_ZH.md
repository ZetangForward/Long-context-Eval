# General
[[中文版](README_ZH.md)] [[English](README.md)]
## LongBench

### 数据下载
如果无法连接到 Hugging Face，可以尝试以下代码。
```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```
或者 

请从网站 "(https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)" 下载数据。下载安装包后，将其解压到文件夹 "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata"。

### 示例代码

LongBench
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LongBench/LongBench.yaml --device 1,3,4,7 --save_tag "tag"
```
或者
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LongBench/LongBench.yaml --device 1,3,4,7 --save_tag "tag"
```p

