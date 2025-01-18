import yaml
with open("/opt/data/private/sora/lab/longtexteval/tasks/task_registry.yaml", 'r', encoding='utf-8') as f:
    task_dict =  yaml.safe_load(f)

for d in task_dict:
    if isinstance(task_dict[d],dict):
        print(task_dict[d])
        for task in task_dict[d]:
            print(task_dict[d][task])
            if task_dict[d][task]==None:
                task_dict[d][task] = task_dict[task]
                print(task_dict[task])
with open("/opt/data/private/sora/lab/longtexteval/tasks/task_registry1.yaml", 'w', encoding='utf-8') as f:
    yaml.dump(task_dict, f)