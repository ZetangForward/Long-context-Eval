# General
[[中文版](README_ZH.md)] [[English](README.md)]
## LongBench

### Data Downloading
If you can't connect to Hugging Face, you can try the following code.
```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```
or 

download the data from the website "(https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)" After downloading the installation package, extract it to the folder "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata"
### Example Code

LongBench
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LongBench/LongBench.yaml --device 0 --save_tag "tag"
```
or
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config tasks/General/LongBench/LongBench.yaml --device 0 --save_tag "tag"
```p

