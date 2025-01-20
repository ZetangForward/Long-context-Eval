## python lte/eval.py --generation_path save/genration/01M_20D_12H_05m --output_path save/output/01M_20D_11H_31m

import os
import sys
import json
sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import pdb
from metrics import get_metric
from collections import defaultdict
from utils.eval_args import handle_cli_args
import numpy as np
from utils.results_table import results_table
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from dataset.utils.benchmark_class import get_benchmark_class

def print_dict_in_table_format(data,task_name_max_len,metric_max_len):
    column_widths = [task_name_max_len+10, metric_max_len+5, 10]  # 定义每列的宽度
    header = ["Tasks", "Metric", "Value"]
    # 打印表头
    logger.info("|{}|{}|{}|".format(
        header[0].center(column_widths[0], ' '),
        header[1].center(column_widths[1], ' '),
        header[2].center(column_widths[2], ' ')
    ))
    logger.info("|{}|{}|{}|".format(
        "-" * column_widths[0],
        "-" * column_widths[1],
        "-" * column_widths[2]
    ))
    for task, metrics in data.items():
        for metric, value in metrics.items():
            logger.info("|{}|{}|{}|".format(
                task.center(column_widths[0], ' '),
                metric.center(column_widths[1], ' '),
                str(value).center(column_widths[2], ' ')
            ))
        logger.info("|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2]
        ))
def construct_metrics(metrics_configs):
    for metrics_name,metrics_config in metrics_configs.items():
        if not metrics_config:
            metrics_configs[metrics_name] = dict()
            metrics_config = {"test":10}
        metrics_configs[metrics_name]["evaluation"] = get_metric(metrics_name)(**metrics_config)
    return metrics_configs

def eval():
    args = handle_cli_args()
    benchmark_list = os.listdir(args.generation_path)
    progress_bar = tqdm(benchmark_list)
    logger.info("*"*40+"  evaluating  "+"*"*40)
    for benchmark_name in progress_bar:
        progress_bar.set_description(f"eval benchmark:{benchmark_name}")
        benchmark = get_benchmark_class(benchmark_name)()
        task_list = os.listdir(args.generation_path+"/"+benchmark_name)
        progress_bar2 = tqdm(task_list)
        for task_name in progress_bar2:
            progress_bar2.set_description(f"eval task:{task_name}")
            gathered_metrics = defaultdict(list)
            task_name = task_name[:-5]
            metrics = construct_metrics(benchmark.metric[task_name])
            save_task_path = os.path.join(args.output_path,task_name)
            os.makedirs(save_task_path, exist_ok=True)
            generation_results_path = os.path.join(args.generation_path+"/"+benchmark_name,task_name+".json")
            if not os.path.exists(generation_results_path):
                continue
            with open(generation_results_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        eval_dict = json.loads(line.strip())
                    except:
                        continue
                    choices,pred,answers = eval_dict["choices"],eval_dict["pred"],eval_dict["answers"]
                    if task_name in ["trec", "triviaqa", "samsum", "lsht"]:
                        pred= pred.lstrip('\n').split('\n')[0]
                    for metric_name,metric in metrics.items():
                        score = metrics[metric_name]["evaluation"](choices,answers, pred)
                        if isinstance(score,dict):
                            for metric_name_sub in score:
                                gathered_metrics[metric_name_sub].append(score[metric_name_sub])
                        else:
                            gathered_metrics[metric_name].append(score)
            final_metrics = {}
            for metric in gathered_metrics:
                if metric in ["cite_num"]:
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),4)
                else:
                    final_metrics[metric] = round(100*np.array(gathered_metrics[metric]).mean(),2)
            logger.info("<<{}>> Final Metric is: {}".format(task_name, final_metrics))
            dump_data = {
                "task_name": task_name,
                "instance_result": gathered_metrics,
                "overall_result": final_metrics,
            }
            with open(
                os.path.join(save_task_path, "final_metrics.json"), "w", encoding="utf-8"
            ) as fout:
                json.dump(dump_data, fout, indent=4, ensure_ascii=False)
    save_path = args.output_path
    metric_list = {}
    task_name_max_len = 0;metric_max_len=0
    for task_name in os.listdir(save_path):
        task_name_max_len = max(task_name_max_len,len(task_name))
        task_path = os.path.join(save_path ,task_name)
        if not os.path.exists(os.path.join(task_path,"final_metrics.json")):
            continue
        with open(os.path.join(task_path,"final_metrics.json"),"r") as f:
            result = json.load(f)
            metric_max_len = max(metric_max_len,len(result["task_name"]))
            metric_list[result["task_name"]] = result["overall_result"]
    print_dict_in_table_format(metric_list,task_name_max_len,metric_max_len)
    results_table(args.output_path)

if __name__ =="__main__":
    eval()
