
import os
import yaml
import argparse
import copy

# 读取所有 YAML 文件内容并保存到 output_path
def yaml_retrieval(output_path,task_list):

    with open("./dataset/task_config_registry.yaml", 'r', encoding='utf-8') as f:
        task_config_registry =  yaml.safe_load(f)

    yaml_list = [] 

    for task_name in task_list:
        ability,benchmark_name= task_config_registry[task_name].split("\\")
        config = "./dataset/{}/{}/{}.yaml".format(ability,benchmark_name,task_name)
        with open(config, "r", encoding="utf-8") as f:

            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            yaml_list.append(yaml_data)
            # modified_yaml_data = copy.deepcopy(yaml_data)
            # modified_yaml_data["generate"]["params"] = "./dataset/utils/llm-params/L-CiteEval_llm_params4.json"
            # yaml_list.append(modified_yaml_data)

    print("The number of selected datasets is {}".format(len(yaml_list)))
    merged_config = {}
    for data in yaml_list:
        task_name = data["task_name"].split(".")[0]
        merged_config[task_name] = data

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_config, f, default_flow_style=False)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_list", type=str, required=True)
    return parser.parse_args()
if __name__ == "__main__":
    args = set_args()
    output_path =  "./cogfigs/eval_configs.yaml"
    yaml_retrieval(output_path,eval(args.task_list))