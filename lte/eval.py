## python lte/eval.py --folder_name longchat-7b-v1.5-32k_02M_25D_22H_13m --model_name Meta-Llama-3.1-8B-Instruct --benchmark_config tasks/General/LongBench/LongBench.yaml
from transformers import pipeline
import yaml
import os,sys
sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
import os
import re
import sys
import pandas as pd
import json
from tqdm import tqdm
from metrics import get_metric
from collections import defaultdict
from utils.eval_args import handle_cli_args
import numpy as np
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from tasks.utils.benchmark_class import get_benchmark_class

def get_longbench_v2_score(dataset,save_path,model_name):
    output = []
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for data in dataset:
        acc = data["score"]["longbench_v2"]
        if data["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc
        if data['length'] == "short":
            short += 1
            short_acc += acc
        elif data['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc
    x =15
    header = "{:<15}\t{:<15}\t{:<15}\t{:<15}\t{:<15}\t{:<15}\t{:<15}".format("Model", "Overall", "Easy", "Hard", "Short", "Medium", "Long")
    output.append(header)
    row = "{:<15}\t{:<15.1f}\t{:<15.1f}\t{:<15.1f}\t{:<15.1f}\t{:<15.1f}\t{:<15.1f}".format(
    model_name, round(100*(easy_acc+hard_acc)/len(dataset), 1), round(100*easy_acc/easy, 1), round(100*hard_acc/hard, 1), round(100*short_acc/short, 1), round(100*medium_acc/medium, 1), round(100*long_acc/long, 1))
    output.append(row)
    open(f'{save_path}/result.txt', 'w', encoding='utf-8').write('\n'.join(output))
    print("result.txt saved in %s" % save_path)
    return output

        
def make_df_niah(data,PRETRAINED_LEN,save_path):
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]
    pivot_table.to_excel(save_path+'/heatmap.xlsx')
    logger.info(f"heatmap_excel  in {save_path}/niah.xlsx...")
    return pivot_table,pretrained_len


def draw_heatmap_niah(pivot_table,model_name,pretrained_len,save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )
    # More aesthetics
    model_name_ = model_name
    plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)
    logger.info("heatmap saving at %s" % save_path+"/image.png" )
    plt.savefig(save_path+"/image.png", dpi=150)

def print_dict_in_table_format(data, excel_file_path):
    benchmark_name_max_len = max(len(name) for name in data.keys())
    task_name_max_len = max(len(task) for tasks in data.values() for task in tasks.keys())
    metric_max_len = max(len(metric) for tasks in data.values() for metrics in tasks.values() for metric in metrics.keys())
    column_widths = [benchmark_name_max_len + 2, task_name_max_len + 10, metric_max_len + 5, 10, 10]
    header = ["BenchMark", "Tasks", "Metric", "Score", "AVG"]

    logger.info("|{}|{}|{}|{}|{}|".format(
        header[0].center(column_widths[0], ' '),
        header[1].center(column_widths[1], ' '),
        header[2].center(column_widths[2], ' '),
        header[3].center(column_widths[3], ' '),
        header[4].center(column_widths[4], ' ')
    ))
    logger.info("|{}|{}|{}|{}|{}|".format(
        "-" * column_widths[0],
        "-" * column_widths[1],
        "-" * column_widths[2],
        "-" * column_widths[3],
        "-" * column_widths[4]
    ))

    all_scores = []
    rows = []
    for benchmark_name, tasks in data.items():
        benchmark_scores = []
        for task, metrics in tasks.items():
            for metric, value in metrics.items():
                logger.info("|{}|{}|{}|{}|{}|".format(
                    benchmark_name.center(column_widths[0], ' '),
                    task.center(column_widths[1], ' '),
                    metric.center(column_widths[2], ' '),
                    str(value).center(column_widths[3], ' '),
                    "".center(column_widths[4], ' ')
                ))
                rows.append([benchmark_name, task, metric, value, ""])

                try:
                    score = float(value)
                    benchmark_scores.append(score)
                    all_scores.append(score)
                except ValueError:
                    logger.warning(f"无法将 {value} 转换为浮点数，跳过该值。")
            logger.info("|{}|{}|{}|{}|{}|".format(
                "-" * column_widths[0],
                "-" * column_widths[1],
                "-" * column_widths[2],
                "-" * column_widths[3],
                "-" * column_widths[4]
            ))


        if benchmark_scores:
            benchmark_avg = round(np.mean(benchmark_scores), 2)
            logger.info("|{}|{}|{}|{}|{}|".format(
                benchmark_name.center(column_widths[0], ' '),
                "Average".center(column_widths[1], ' '),
                "Overall".center(column_widths[2], ' '),
                str(benchmark_avg).center(column_widths[3], ' '),
                "".center(column_widths[4], ' ')
            ))
            rows.append([benchmark_name, "Average", "Overall", benchmark_avg, ""])
        logger.info("|{}|{}|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2],
            "-" * column_widths[3],
            "-" * column_widths[4]
        ))

    if all_scores:
        total_avg = round(np.mean(all_scores), 2)

        logger.info("|{}|{}|{}|{}|{}|".format(
            "Total".center(column_widths[0], ' '),
            "Average".center(column_widths[1], ' '),
            "Overall".center(column_widths[2], ' '),
            "".center(column_widths[3], ' '),
            str(total_avg).center(column_widths[4], ' ')
        ))
        logger.info("|{}|{}|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2],
            "-" * column_widths[3],
            "-" * column_widths[4]
        ))
        rows.append(["Total", "Average", "Overall", "", total_avg])

    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=header)
    # 保存到 Excel 文件
    df.to_excel(excel_file_path, index=False)
def construct_metrics(metrics_configs):
    clock = 0
    for metrics_name,metrics_config in metrics_configs.items():
        if not metrics_config:
            metrics_configs[metrics_name] = dict()
            metrics_config = {"test":10}
        if metrics_name in ["l_cite_eavl_niah_cite","l_cite_eavl_cite"]:
            if clock==0:
                print("If you get stuck here, please check whether the tasksource/deberta-base-long-nli model has been installed for evaluation.")
                clock +=1
            else:
                print()
            pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli", device="cuda:0")
        else:
            pipe = "0"
        metrics_configs[metrics_name]["evaluation"] = get_metric(metrics_name)(pipe=pipe,**metrics_config)
    return metrics_configs

def eval():

    benchmark_dict = defaultdict(lambda: defaultdict(dict))
    args = handle_cli_args()
    benchmark_config_path = args.benchmark_config.strip()
    with open(benchmark_config_path, "r") as f:
        config = yaml.safe_load(f)
    args.limit = "auto"
    benchmark_list = []
    folder_name = args.folder_name
    for ability in os.listdir("tasks"):
        ability_path = os.path.join("tasks", ability)
        if os.path.isdir(ability_path):
            for benchmark in os.listdir(ability_path):
                benchmark_path = os.path.join(ability_path, benchmark, "prediction",folder_name)
                if os.path.exists(benchmark_path):
                    if benchmark =="RULER":
                        for task_name in  os.listdir(benchmark_path):
                            if task_name.endswith(".json"):
                                length = task_name.split("_")[-1][:-5]
                            elif task_name.endswith(".jsonl"):

                                length = task_name.split("_")[-1][:-6]
                            benchmark_list.append(benchmark+f"_{length}")
                    else:
                        benchmark_list.append(benchmark)
    progress_bar = tqdm(benchmark_list)
    logger.info("*"*40+"  evaluating  "+"*"*40)
    for benchmark_name in progress_bar:
        if benchmark_name=="NIAH":
            data_niah=[]
        progress_bar.set_description(f"eval benchmark:{benchmark_name}")
        match = re.compile(r'_(\d+)').search(benchmark_name)
        if match:
            benchmark = get_benchmark_class(benchmark_name.split("_")[0])(benchmark_name.split("_")[-1],args)
        else:
            benchmark = get_benchmark_class(benchmark_name)(args,config=config)
        task_list = os.listdir(f"tasks/{benchmark.ability}/{benchmark.benchmark_name}/prediction/{folder_name}")
        progress_bar2 = tqdm(task_list)
        for task_name in progress_bar2:
            if task_name.endswith(".json"):
                task_name = task_name[:-5]
            elif task_name.endswith(".jsonl"):
                task_name = task_name[:-6]
            progress_bar2.set_description(f"eval task:{task_name}")
            gathered_metrics = defaultdict(list)
            if "RULER" in benchmark_name:
                length = task_name.split("_")[-1]
                if length!=benchmark.length:
                    continue
                metrics = construct_metrics(benchmark.metric["_".join(task_name.split("_")[:-1])])
                save_task_path = os.path.join("tasks",benchmark.ability,benchmark_name.split("_")[0],"results",folder_name,task_name+".jsonl")
                generation_results_path = os.path.join("tasks",benchmark.ability,benchmark_name.split("_")[0],"prediction",folder_name,task_name+".jsonl")
            else:
                metrics = construct_metrics(benchmark.metric[task_name])
                save_task_path = os.path.join("tasks",benchmark.ability,benchmark_name,"results",folder_name,task_name+".jsonl")
                generation_results_path = os.path.join("tasks",benchmark.ability,benchmark_name,"prediction",folder_name,task_name+".jsonl")
            os.makedirs(save_task_path, exist_ok=True)
            if not os.path.exists(generation_results_path):
                continue
            data = []
            with open(generation_results_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        eval_dict = json.loads(line.strip())
                    except:
                        continue
                    eval_dict["score"] = {}
                    passage,choices,pred,label = eval_dict["passage"],eval_dict["choices"],eval_dict["pred"],eval_dict["label"]
                    if task_name in ["trec", "triviaqa", "samsum", "lsht"]:
                        pred= pred.lstrip('\n').split('\n')[0]
                    for metric_name,metric in metrics.items():
                        if metric_name in ["l_cite_eavl_cite","l_cite_eavl_niah_cite","l_cite_eavl_counting_stars_cite","l_cite_eavl_counting_stars"]:
                            score = metrics[metric_name]["evaluation"](passage,label, pred)
                        else:
                            score = metrics[metric_name]["evaluation"](choices,label, pred)
                        if isinstance(score,dict):
                            for metric_name_sub in score:
                                eval_dict["score"].update(score)
                                gathered_metrics[metric_name_sub].append(score[metric_name_sub])
                        else:
                            eval_dict["score"].update({f"{metric_name}":score})
                            gathered_metrics[metric_name].append(score)
                        
                        data.append(eval_dict)
            
                        if benchmark_name=="NIAH":
                            data_niah.append({"Document Depth": eval_dict["depth_percent"],
                            "Context Length": eval_dict["context_length"],
                            "Score": eval_dict["score"]})
  
            with open(generation_results_path, "w", encoding="utf-8") as f:
                for eval_dict in data:
                    f.write(json.dumps(eval_dict, ensure_ascii=False) + '\n')
            final_metrics = {}
            for metric in gathered_metrics:
                if metric in ["cite_num","niah","cite_num_cite"]:
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),2)
                else:
                    final_metrics[metric] = round(100*np.array(gathered_metrics[metric]).mean(),2)
            if benchmark_name=="LEval":
                if task_name in benchmark.datasets_closed_ended:
                    benchmark_dict[benchmark_name+"_closed"][task_name] = final_metrics
                else:
                    benchmark_dict[benchmark_name+"_open"][task_name] = final_metrics
            else:
                benchmark_dict[benchmark_name][task_name] = final_metrics
            logger.info("<<{}>> Final Metric is: {}".format(task_name, final_metrics))

            dump_data = {
                "task_name": task_name,
                "instance_result": gathered_metrics,
                "overall_result": final_metrics,
            }
            with open(
                os.path.join(save_task_path, "final_metrics.jsonl"), "w", encoding="utf-8"
            ) as fout:
                json.dump(dump_data, fout, indent=4, ensure_ascii=False)
                
            if benchmark.benchmark_name == "NIAH":
                save_path = f"./tasks/Retrieve/NIAH/results/{folder_name}"
                PRETRAINED_LEN=81920
                pivot_table,pretrained_len =make_df_niah(data_niah,PRETRAINED_LEN,save_path)
                draw_heatmap_niah(pivot_table,args.model_name,pretrained_len,save_path)
            if benchmark.benchmark_name == "LongBench_v2":
                save_path = f"./tasks/General/LongBench_v2/results/{folder_name}"
                get_longbench_v2_score(data,save_path,model_name=args.model_name)

        output_path = f"./tasks/{benchmark.ability}/{benchmark.benchmark_name}/results/{folder_name}"
        pred_path = f"./tasks/{benchmark.ability}/{benchmark.benchmark_name}/prediction/{folder_name}"
        os.makedirs(output_path,exist_ok=True)
        print_dict_in_table_format(benchmark_dict,f"./tasks/{benchmark.ability}/{benchmark.benchmark_name}/results/{folder_name}/output_table.xlsx")
        logger.info("results_table is saved in {}".format(output_path+"/output_table.xlsx"))
        
        if benchmark_name=="L_CiteEval":
            from tasks.utils.draw.L_Citeeval import draw_excel,draw_line_graph
            draw_excel(output_path+"/output_table.xlsx")
            print(pred_path)
            draw_line_graph(f"tasks/Faithfulness/L_CiteEval/prediction/{folder_name}",args.model_name)
    
if __name__ =="__main__":
    eval()
