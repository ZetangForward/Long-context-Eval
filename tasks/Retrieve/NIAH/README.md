# Retrieve
[[中文版](README_ZH.md)] [[English](README.md)]
## NIAH

### Example Code
NIAH
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config /tasks/Retrieve/NIAH/NIAH.yaml --device 1,3,4,7 --save_tag "tag"
```
or 
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_config /tasks/Retrieve/NIAH/NIAH.yaml --device 1,3,4,7 --save_tag "tag"
```