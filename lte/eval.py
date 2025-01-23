## python lte/eval.py --generation_path save/genration/01M_23D_19H_21m --output_path save/output/01M_20D_11H_31m
from transformers import pipeline
import os
import re
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

def print_dict_in_table_format(data, benchmark_name_max_len, task_name_max_len, metric_max_len):
    # 定义每列的宽度，新增 AVG 列
    column_widths = [benchmark_name_max_len + 2, task_name_max_len + 10, metric_max_len + 5, 10, 10]
    header = ["BenchMark", "Tasks", "Metric", "Score", "AVG"]
    # 打印表头
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
                try:
                    # 尝试将值转换为浮点数
                    score = float(value)
                    benchmark_scores.append(score)
                    all_scores.append(score)
                except ValueError:
                    # 处理无法转换为浮点数的情况
                    logger.warning(f"无法将 {value} 转换为浮点数，跳过该值。")
            # logger.info("|{}|{}|{}|{}|{}|".format(
            #         "-" * column_widths[0],
            #         "-" * column_widths[1],
            #         "-" * column_widths[2],
            #         "-" * column_widths[3],
            #         "-" * column_widths[4]
            #     ))

        # 计算当前 BenchMark 的平均分
        if benchmark_scores:
            benchmark_avg = round(np.mean(benchmark_scores), 2)
            logger.info("|{}|{}|{}|{}|{}|".format(
                benchmark_name.center(column_widths[0], ' '),
                "Average".center(column_widths[1], ' '),
                "Overall".center(column_widths[2], ' '),
                str(benchmark_avg).center(column_widths[3], ' '),
                "".center(column_widths[4], ' ')
            ))
        logger.info("|{}|{}|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2],
            "-" * column_widths[3],
            "-" * column_widths[4]
        ))

    # 计算所有分数的总平均值
    if all_scores:
        total_avg = round(np.mean(all_scores), 2)
        # 打印总平均值行
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
def construct_metrics(metrics_configs):
    for metrics_name,metrics_config in metrics_configs.items():
        if not metrics_config:
            metrics_configs[metrics_name] = dict()
            metrics_config = {"test":10}
        if metrics_name in ["l_cite_eavl_niah_cite","l_cite_eavl_cite"]:
            pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli", device='cuda')
        else:
            pipe = "0"
        metrics_configs[metrics_name]["evaluation"] = get_metric(metrics_name)(pipe=pipe,**metrics_config)
    return metrics_configs

def eval():
    benchmark_dict = {}
    benchmark_name_max_len,task_name_max_len,metric_max_len = 0,0,0
    args = handle_cli_args()
    benchmark_list = os.listdir(args.generation_path)
    progress_bar = tqdm(benchmark_list)

    logger.info("*"*40+"  evaluating  "+"*"*40)
    for benchmark_name in progress_bar:

        benchmark_name_max_len = max(benchmark_name_max_len,len(benchmark_name))
        benchmark_dict[benchmark_name] = {}
        progress_bar.set_description(f"eval benchmark:{benchmark_name}")
        match = re.compile(r'_(\d+)').search(benchmark_name)
        if match:
            benchmark = get_benchmark_class(benchmark_name.split("_")[0])(benchmark_name.split("_")[-1])
        else:
            benchmark = get_benchmark_class(benchmark_name)()
        task_list = os.listdir(args.generation_path+"/"+benchmark_name)
        progress_bar2 = tqdm(task_list)
        for task_name in progress_bar2:
            progress_bar2.set_description(f"eval task:{task_name[:-5]}")
            gathered_metrics = defaultdict(list)
            task_name = task_name[:-5]
            task_name_max_len = max(task_name_max_len,len(task_name))
            metrics = construct_metrics(benchmark.metric[task_name])
            save_task_path = os.path.join(args.output_path,benchmark_name,task_name)
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
                metric_max_len = max(len(metric),metric_max_len)
                if metric in ["cite_num"]:
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),2)
                else:
                    final_metrics[metric] = round(100*np.array(gathered_metrics[metric]).mean(),2)
            
            benchmark_dict[benchmark_name][task_name] = final_metrics
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

    print_dict_in_table_format(benchmark_dict,benchmark_name_max_len,task_name_max_len,metric_max_len)
    results_table(args.output_path)

if __name__ =="__main__":
    eval()
