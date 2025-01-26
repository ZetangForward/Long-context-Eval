import os
import yaml

# 获取当前目录

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取参照目录（这里是data目录）的绝对路径，这里假设参照目录与当前脚本在同一层级下的不同文件夹中，可根据实际情况调整获取方式
reference_dir = os.path.abspath('/opt/data/private/sora/lab/longtexteval/')
# 计算相对路径
relative_path = os.path.relpath(current_dir, reference_dir)
print(relative_path)
# 遍历当前目录下的所有文件和文件夹
for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith('.yaml'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                print(content["path"])
                print(file)
                content["path"] = os.path.join(relative_path,"data",file[:-5]+".json")
                content["task_name"] = file[:-5]
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(content, f)