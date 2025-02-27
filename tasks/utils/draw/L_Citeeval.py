import pandas as pd
import os,sys
def draw_excel(in_path):
# 定义文件路径
    tasks = {"narrativeqa":"Single Document QA","natural_questions":"Single Document QA","2wikimultihopqa":"Multi-Doc QA","hotpotqa":"Multi-Doc QA","multi_news":"Summarization","gov_report":"Summarization","qmsum":"Summarization","counting_stars":"Counting Stars","niah":"Needle in a Haystack","locomo":"Dialogue Understanding","dialsim":"Dialogue Understanding"}
    df = pd.read_excel(in_path)
    print(df)
    # 从 'Tasks' 列中提取任务标识，例如从 'L-CiteEval-Data_niah' 中提取 'niah'
    df['Task_Identifier'] = df['Tasks'].str.extract(r'L-CiteEval-(Data|Hardness|Length)_(\w+)')[1]

    # 根据任务标识映射到任务类型
    df['Task_Type'] = df['Task_Identifier'].map(tasks)
    # 筛选出非 'Average' 行的数据
    filtered_df = df[~df['Tasks'].str.contains('Average')]
    # 按任务类型和指标分组，并计算 Score 列的平均值
    grouped = filtered_df.groupby(['Task_Type', 'Metric'])['Score'].mean()
    # 将分组结果转换为 DataFrame 格式，便于查看
    result_df = grouped.reset_index()
    save_path = "/".join(in_path.split("/")[:-1])+"/L_citeeval.xlsx"
    print(result_df)
    result_df.to_excel(save_path)
import json
import matplotlib.pyplot as plt
import numpy as np
def draw_line_graph(in_path,model_name):
    for task_name in os.listdir(in_path):
        if task_name == "L-CiteEval-Hardness_gov_report.jsonl":
            import pdb;pdb.set_trace()
        if "Hardness" in task_name:
            scores = {'easy': {},'medium': {},'hard': {}}
            counts = {'easy': {},'medium': {},'hard': {}}
        elif "Length" in task_name:
            scores = {'8k': {},'16k': {},'32k': {}}
            counts = {'8k': {},'16k': {},'32k': {}}
        else:
            return 
        file_path = os.path.join(in_path, task_name)
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                score = data["score"]
                if not isinstance(score, dict):
                    continue
                if "Hardness" in task_name:
                    key = data["hardness"]
                elif "Length" in task_name:
                    key = data["length"]
                    if key<=8000:key="8k"
                    elif 12000<key<24000:key = "16k"
                    else:key = "32k"
                for score_key, score_value in score.items():
                    if score_key not in scores[key]:
                        scores[key][score_key] = 0
                        counts[key][score_key] = 0
                    scores[key][score_key] += score_value
                    counts[key][score_key] += 1
        
        avg_scores = {key: {
                score_key: scores[key][score_key] / counts[key][score_key]
                for score_key in scores[key]}for key in scores}
        if task_name.endswith(".jsonl"):task_name = task_name[:-6]
        else:task_name = task_name[:-5]
        print(avg_scores)
        levels = list(avg_scores.keys())
        plt.rcParams['figure.dpi'] = 300
        for score_key in avg_scores[key].keys():
            scores = [avg_scores[key][score_key] for key in levels]
            plt.plot(levels, scores, marker='o', label=score_key)
        plt.title(f'{task_name}_{model_name}')
        plt.xlabel('Level')
        plt.ylabel('Average Score')
        plt.legend()
        save_path = f'{task_name}_{score_key}.png'
        outpath = in_path.replace("prediction", "results")
        plt.savefig(outpath+"/"+save_path, dpi=300) 
        plt.clf()
    print(f"L_citeeval line graph save in {outpath}")