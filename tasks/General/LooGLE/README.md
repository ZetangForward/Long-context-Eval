# General

## LooGLE

### Data Downloading
try:
```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
```
or 

download the data from the website "(https://huggingface.co/datasets/bigai-nlco/LooGLE/tree/main/data)" Download the JSON file to the folder "/tasks/Faithfulness/L_CiteEval/tmp_Rawdata".
### Example Code

LooGLE
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/General/LooGLE/LooGLE.yaml --device 1,3,4,7 --save_tag "tag"


