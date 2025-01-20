## python lte/eval.py --generation_path /save/genration/01M_18D_15H_23m --output_path /save/output/01M_18D_15H_23m

import os
import sys
import json
sys.path.append(os.path.join(os.path.abspath(__file__)))
from tqdm import tqdm
from metrics import get_metric
from collections import defaultdict
from utils.eval_args import handle_cli_args
import numpy as np
from utils.results_table import results_table
from loguru import logger
from dataset.utils.benchmark_class import get_benchmark_class


def construct_metrics(metrics_configs):
    for metrics_name,metrics_config in metrics_configs.items():
        if not metrics_config:
            metrics_configs[metrics_name] = dict()
            metrics_config = {"test":10}
        metrics_configs[metrics_name]["evaluation"] = get_metric(metrics_name)(**metrics_config)
    return metrics_configs

def eval():
    benchmark_list = os.listdir(args.generation_path)
    progress_bar = tqdm(benchmark_list)
    args = handle_cli_args()
    logger.info("*"*20+"  evaluating  "+"*"*20)
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
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),2)
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

    results_table(args.output_path)

if __name__ =="__main__":
    eval()
