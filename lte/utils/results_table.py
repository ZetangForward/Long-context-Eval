import json
import os,sys
import pandas as pd
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
def results_table(data_generation_time):
    results = {}
    for ability in os.listdir("tasks"):
        ability_path = os.path.join("tasks", ability)
        if os.path.isdir(ability_path):
            for benchmark in os.listdir(ability_path):
                benchmark_path = os.path.join(ability_path, benchmark, "result",data_generation_time)
                if os.path.exists(benchmark_path):
                    for task_name in os.listdir(benchmark_path):
                        task_path = os.path.join(benchmark_path ,task_name)
                        with open(os.path.join(task_path,"final_metrics.json"),"r") as f:
                            result = json.load(f)
                            results[result["task_name"]] = result["overall_result"]

    # 将字典转换为DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    output_path = f"./tasks/{data_generation_time}"
    os.makedirs(output_path,exist_ok=True)
    # 重置索引并将其命名为'task_name'
    df = df.reset_index()
    df = df.rename(columns={'index': 'task_name'})
    table_path = os.path.join(output_path,'output_table.xlsx')
    print(table_path)
    df.to_excel(table_path, index=False)
    logger.info("results_table is saved in {}".format(table_path))
results_table("01M_31D_16H_15m")