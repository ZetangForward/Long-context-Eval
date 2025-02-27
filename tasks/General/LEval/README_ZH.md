# General
[[中文版](README_ZH.md)] [[English](README.md)]
## LEval

### 数据下载
如果无法连接到 Hugging Face，可以尝试以下代码。
```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```
或者 

请从网站 "(https://huggingface.co/datasets/L4NLP/LEval/tree/main/LEval/Generation)" 下载数据，并直接将 JSON 文件保存在路径 "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata"。
### 示例代码

LEval
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LEval/LEval.yaml --device 1,3,4,7 --save_tag "tag"
```
或者 
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LEval/LEval.yaml --device 1,3,4,7 --save_tag "tag"
```
