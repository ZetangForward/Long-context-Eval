import json
import os
import pandas as pd

def results_table(result_path):
    results = {}
    output_path = result_path
    for task_name in os.listdir(output_path):
        task_path = os.path.join(output_path ,task_name)
        if not os.path.exists(os.path.join(task_path,"final_metrics.json")):
            return
        with open(os.path.join(task_path,"final_metrics.json"),"r") as f:
            result = json.load(f)
            results[result["task_name"]] = result["overall_result"]

    # 将字典转换为DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # 重置索引并将其命名为'task_name'
    df = df.reset_index()
    df = df.rename(columns={'index': 'task_name'})
    table_path = os.path.join(output_path,'output_table.xlsx')
    df.to_excel(table_path, index=False)
    print("results_table is saved in {}".format(table_path))
