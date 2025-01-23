# python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct  --eval --benchmark_names LongBench,Counting_Stars --device 0,1 --device_split_num 2 --limit 1
import os
import sys
import yaml
import pdb
sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
import time
import json
import subprocess
from dataset.utils.benchmark_class import get_benchmark_class
import re
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from utils.request import Request
import random
from tasks.instance import Instance
from tqdm import tqdm
from Model_Deploy_URLs import get_model
import numpy as np
from utils.main_args import handle_cli_args
import torch.multiprocessing as mp
import torch
from Model_Deploy_URLs.rag import get_rag_method
class Evaluator():
    def __init__(self,args,all_benchmarks):
        #设置参数
        self.tasks_list = []
        self.args = args
        self.limit = int(args.limit) if args.limit!="auto" else 10000
        self.build_tasks(all_benchmarks) #根据配置文件创建task

    def build_tasks(self,all_benchmarks):
        for benchmark in all_benchmarks:
            for task_name in benchmark.task_names:
                #后处理函数返回【【output】【postprocessed_output】】
                if hasattr(benchmark, 'length'):
                    path = benchmark.data_path+task_name+f"_{benchmark.length}"+".json"
                else:
                    path = benchmark.data_path+task_name+".json"
                with open(path, "r", encoding="utf-8") as file:
                    for index, line in enumerate(file):
                        if index>=self.limit:
                            break
                        raw_input = Instance(json.loads(line.strip()))
                        prompt_input = benchmark.transform(raw_input.data,task_name)
                        self.tasks_list.append([task_name,benchmark,Request(
                            instances=prompt_input,
                            params=benchmark.llm_params[task_name],
                            raw_example=raw_input,
                        )])


    def run(self,device_split_num):
        devices_list=self.args.device.split(",")
        processes = []
        chunk_num = len(devices_list)//device_split_num
        for i in range(0, len(devices_list), device_split_num):   
            raw_data = self.tasks_list[i//device_split_num::chunk_num]
            devices = ",".join(devices_list[i:i + device_split_num])
            logger.info("devices:{}".format(devices))
            p = mp.Process(target=self.get_pred, args=(i//device_split_num,raw_data,devices))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        

    def get_pred(self,i,raw_data,devices):
        #model depoly     
        try:
            model = get_model(self.args.acceleration)(self.args,devices)
            model_path = self.args.model_path
        except:
            model = get_model(self.args.server)(self.args,devices)
            model_path = self.args.model_path
        model.deploy()
        for task_name,benchmark,request in tqdm(raw_data,desc="The {}th chunk".format(i)):
            request.instances["input"] = benchmark.modify(request.instances["input"],model,model_path)
            #调出模型的生成函数
            with torch.no_grad(): 
                result = model.generate(request.params, request.instances["input"])
            
            #后处理
            raw_outputs, processed_outputs = result[::],result[::]
            if  hasattr(benchmark, 'postprocess'):
                processed_outputs = benchmark.postprocess(raw_outputs)

            # logger.info("模型输入:\n{}".format(request.instances["input"][-50:]))
            # logger.info("模型输出结果:\n{}".format(raw_outputs[0]))
            # logger.info("处理后的结果:\n{}".format(processed_outputs[0]))
            # logger.info("答案:\n{}".format(request.instances["processed_output"]))
            
            request.raw_example.raw_outputs = raw_outputs
            request.raw_example.processed_outputs = processed_outputs
            request.raw_example.ground_truth = request.instances["processed_output"]
            request.raw_example.prompt_inputs = request.instances["input"]
        
            # os.makedirs("测试", exist_ok=True)
            # with open(os.path.join("测试",task_name+".json"), "a", encoding="utf-8") as f:
            #     json.dump({"task":task_name,"模型输入":request.instances["input"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth}, f, ensure_ascii=False)
            #     f.write('\n')
            # with open('output.doc', 'a') as f:
            #     logger.info("评测文本输入:\n{}\n".format(request.raw_example.data["passage"]),file=f)
            #     logger.info("模型预测:\n{}\n".format(request.raw_example.processed_outputs),file=f)
            #     logger.info("正确结果:\n{}\n".format(request.raw_example.ground_truth),file=f)

            os.makedirs(os.path.join(self.args.generation_path,benchmark.benchmark_name), exist_ok=True)
            
            with open(os.path.join(self.args.generation_path,benchmark.benchmark_name,task_name+".json"), "a", encoding="utf-8") as f:
                if True:
                    json.dump({"choices":request.raw_example.data["passage"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth,"model_input":request.instances["input"]}, f, ensure_ascii=False)
                else:
                    json.dump({"choices":request.raw_example.data["choices"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth}, f, ensure_ascii=False)
                f.write('\n')

    #完善transfrom后的数据,加description和num_fewshot

def format_tasks(all_tasks):
    formatted_tasks = ""
    l = 0
    for key, value in all_tasks.items():
        formatted_tasks+=f"{len(value)} tasks: from"
        formatted_tasks += f"{key}:{value} "
        l += len(value)
        formatted_tasks += f"\n"
    formatted_tasks += f"Totally {l} tasks"
    return formatted_tasks
def main():
    current_time = time.localtime()
    formatted_time = time.strftime("%mM_%dD_%HH_%Mm", current_time)
    os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
    args = handle_cli_args()
    args.generation_path = args.generation_path+"/"+formatted_time
    args.save_path = args.save_path+"/"+formatted_time

    seed = 0;random.seed(seed);np.random.seed(seed)
    start_time = time.time()
    # 读取 asd.json 文件
    if len(args.device.split(","))%args.device_split_num!=0:
        raise ValueError("The number of GPUs cannot be divided evenly by the number of blocks.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #task data download
    all_tasks = {}
    task_len = 0
    all_benchmarks= []
    logger.info(f"Loading the config information for benchmarks: {args.benchmark_names}")
    for benchmark_name in args.benchmark_names.split(","):
        benchmark_name = benchmark_name.strip()
        with open(f"tasks/configs/{benchmark_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        if "length" in config:
            for l in config["length"]:
                benchmark = get_benchmark_class(benchmark_name)(l)
                benchmark.task_names = config["tasks"]
                all_benchmarks.append(benchmark)
                task_len += len(benchmark.task_names)
                all_tasks[benchmark.benchmark_name]=benchmark.task_names
        else:
            benchmark = get_benchmark_class(benchmark_name)()
            benchmark.task_names = config["tasks"]
            all_benchmarks.append(benchmark)
            task_len += len(benchmark.task_names)
            all_tasks[benchmark.benchmark_name]=benchmark.task_names

    formatted_output = format_tasks(all_tasks)
    logger.info(f"The tasks you selected are:\n{formatted_output}")
    logger.info("benchmark  downoading . .. . .. .. .. . .. .. .. . .. .. .. . .. .. .. . .. .. .. . .. .")
    progress_bar = tqdm(all_benchmarks)
    tasks_path_list = []
    for benchmark in progress_bar:   
        tasks_list = []
        progress_bar.set_description(f"Downloading {benchmark.benchmark_name}")
        benchmark.download_and_transform_data(args=args)
        if args.rag!="":
            data_path = benchmark.data_path
            for task_name in benchmark.task_names:
                task_path = data_path+"/"+task_name+".json"
                tasks_path_list.append(task_path)
                task_name += f"_{args.rag}"
                tasks_list.append(task_name)
            benchmark.task_names = tasks_list
    if args.rag!="":
        rag = get_rag_method(args.rag)(args.model_path,tasks_path_list)
        logger.info("performing information retrieval")
        rag.traverse_task()            

    #开始评测
    logger.info(f"The model starts generating data.")
    evaluator = Evaluator(args,all_benchmarks)
    evaluator.run(args.device_split_num)
    logger.info(f"All data is stored in {args.generation_path}")

    #eval
    if args.eval:
        command = ["python","./lte/eval.py",
                    "--generation_path",args.generation_path ,
                    "--output_path",args.save_path,
                    ]
        subprocess.run(command)


    #查看运行时间
    all_time = time.time()-start_time
    
    # 计算小时、分钟和秒
    logger.info("The total running time was ：{:02d}:{:02d}:{:02d}".format(int(all_time // 3600), int((all_time % 3600) // 60), int(all_time % 60)))

if __name__ == "__main__":

    main()