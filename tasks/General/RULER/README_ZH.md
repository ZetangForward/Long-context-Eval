# General
[[中文版](README_ZH.md)] [[English](README.md)]
## RULER

### 数据下载

### 示例代码

RULER
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/General/RULER/RULER.yaml --device 1,3,4,7 --save_tag "tag"
```
或者 
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/General/RULER/RULER.yaml --device 1,3,4,7 --save_tag "tag"
```
