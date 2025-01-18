import os
import yaml
path = "/opt/data/private/sora/lab/longtexteval/dataset/General/LongBench"
l = {}
for file in os.listdir(path):
    if file.endswith(".yaml"):
        with open(os.path.join(path,file)) as f:
            data = yaml.safe_load(f)
            l[file[:-5]]=data["metric"]
print(l)