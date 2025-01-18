import os
import yaml

# 获取当前目录
ls = []
path = "/opt/data/private/sora/lab/longtexteval/dataset/Factuality/L-CiteEval"
for file in os.listdir(path):
    if file.endswith(".yaml"):
        ls.append(file[:-5])
print(ls)