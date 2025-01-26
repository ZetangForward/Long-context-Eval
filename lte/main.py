# python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct/ --rag BM25   --eval    --benchmark RULER:tasks/General/RULER/RULER.yaml  --device 0,1 --device_split_num 2 --limit 1
import os
import sys
import yaml
import pdb
sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
import time
import json
import subprocess
from tasks.utils.benchmark_class import get_benchmark_class
import re
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from utils.request import Request
import random
from evaluation.instance import Instance
from tqdm import tqdm
from models_deploy import get_model
import numpy as np
from utils.main_args import handle_cli_args
import torch.multiprocessing as mp
import torch
# from models_deploy.rag import get_rag_method
class Evaluator():
    def __init__(self,args,all_benchmarks):
        #Set parameters
        self.tasks_list = []
        self.args = args
        self.limit = int(args.limit) if args.limit!="auto" else 10000
        self.build_tasks(all_benchmarks) #Create a task based on the configuration file.

    def build_tasks(self,all_benchmarks):
        for benchmark in all_benchmarks:
            for task_name in benchmark.task_names:
                if self.args.rag!="":
                    task_name+=f"_{self.args.rag}"
                if hasattr(benchmark, 'length'):
                    path = benchmark.data_path+"/"+task_name+f"_{benchmark.length}"+".json"
                else:
                    path = benchmark.data_path+"/"+task_name+".json"
                with open(path, "r", encoding="utf-8") as file:
                    for index, line in enumerate(file):
                        if index>=self.limit:
                            break
                        raw_input = Instance(json.loads(line.strip()))
                        prompt_input = benchmark.transform(raw_input.data,task_name)
                        self.tasks_list.append([task_name,benchmark,Request(
                            instances=prompt_input,
                            params=benchmark.llm_params[task_name if self.args.rag=="" else "_".join(task_name.split("_")[:-1])],
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
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            p = mp.Process(target=self.get_pred, args=(i//device_split_num,raw_data,devices))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    def get_pred(self,i,raw_data,devices):
        #model depoly     
        try:
            model = get_model(self.args.acceleration)(self.args, devices)
            model_path = self.args.model_path
        except:
            model = get_model(self.args.server)(self.args,devices)
            model_path = self.args.model_path
        model.deploy()
        for task_name,benchmark,request in tqdm(raw_data,desc="The {}th chunk".format(i)):
            request.instances["input"] = benchmark.modify(request.instances["input"],model,model_path)
            #Call the model's generation function.
            with torch.no_grad(): 
                result = model.generate(request.params, request.instances["input"])

            #Post-processing
            raw_outputs, processed_outputs = result[::],result[::]
            if  hasattr(benchmark, 'postprocess'):
                processed_outputs = benchmark.postprocess(raw_outputs)

            request.raw_example.raw_outputs = raw_outputs
            request.raw_example.processed_outputs = processed_outputs
            request.raw_example.ground_truth = request.instances["processed_output"]
            request.raw_example.prompt_inputs = request.instances["input"]

            if hasattr(benchmark, 'length'):
                path = os.path.join(self.args.generation_path,benchmark.benchmark_name+f"_{benchmark.length}",task_name+".json")
                os.makedirs(os.path.join(self.args.generation_path,benchmark.benchmark_name+f"_{benchmark.length}"), exist_ok=True)
            else:

                path = os.path.join(self.args.generation_path,benchmark.benchmark_name,task_name+".json")
                os.makedirs(os.path.join(self.args.generation_path,benchmark.benchmark_name), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                if True:
                    json.dump({"choices":request.raw_example.data["passage"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth,"model_input":request.instances["input"]}, f, ensure_ascii=False)
                else:
                    json.dump({"choices":request.raw_example.data["choices"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth}, f, ensure_ascii=False)
                f.write('\n')

def format_tasks(all_tasks):
    formatted_tasks = ""
    l = 0
    for key, value in all_tasks.items():
        formatted_tasks+=f"{len(value)} tasks: from "
        formatted_tasks += f"{key}:{value} "
        l += len(value)
        formatted_tasks += f"\n"
    formatted_tasks += f"Totally {l} tasks"
    return formatted_tasks


def main():
    mp.set_start_method('spawn')
    ## init
    # mp.set_start_method('spawn')
    current_time = time.localtime()
    formatted_time = time.strftime("%mM_%dD_%HH_%Mm", current_time)
    args = handle_cli_args()
    if args.device ==" ":
        gpu_count = torch.cuda.device_count()
        args.device = ','.join(map(str, range(gpu_count)))

    args.generation_path = args.generation_path+"/"+formatted_time
    args.save_path = args.save_path+"/"+formatted_time
    seed = 0;random.seed(seed);np.random.seed(seed)
    start_time = time.time()
    if len(args.device.split(","))%args.device_split_num!=0:
        raise ValueError("The number of GPUs cannot be divided evenly by the number of blocks.")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    #task data download
    all_tasks = {}
    task_len = 0
    all_benchmarks= []
    logger.info(f"Loading the config information")
    progress_bar = tqdm(args.benchmark_configs.split(","))
    for benchmark_config_path in progress_bar:
        benchmark_config_path = benchmark_config_path.strip()
        with open(benchmark_config_path, "r") as f:
            config = yaml.safe_load(f)
            benchmark_name = config["benchmark_name"].strip()
            progress_bar.set_description(f"Loading {benchmark_name} config from {benchmark_config_path}")
            if "length" in config and "num_samples" in config:
                for l in config["length"]:            
                    benchmark = get_benchmark_class(benchmark_name)(l,config["num_samples"])
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
    logger.info(f"The tasks you've selected are as follows:\n{formatted_output}")
    logger.info("Benchmark data is currently being downloaded and transformed...")
    progress_bar = tqdm(all_benchmarks)
    tasks_path_list = [] 

    for benchmark in progress_bar:   
        progress_bar.set_description(f"Downloading {benchmark.benchmark_name} data")
        # benchmark.download_and_transform_data(args=args)
        if args.rag!="":
            data_path = benchmark.data_path
            for task_name in benchmark.task_names:
                task_path = data_path+"/"+task_name+".json"
                tasks_path_list.append(task_path)
    if args.rag!="":
        rag = get_rag_method(args.rag)(args.model_path,tasks_path_list)
        logger.info("performing information retrieval")
        rag.traverse_task()            

    #start to generate
    logger.info("The model has initiated the data generation process.")
    evaluator = Evaluator(args, all_benchmarks)
    evaluator.run(args.device_split_num)
    logger.info(f"All generated data has been successfully stored in {args.generation_path}.")

    #eval
    if args.eval:
        command = ["python","./lte/eval.py",
                    "--generation_path",args.generation_path ,
                    "--output_path",args.save_path,
                    ]
        subprocess.run(command)


    #execution_time
    execution_time = time.time()-start_time
    logger.info("The total running time was : {:02d}:{:02d}:{:02d}".format(int(execution_time // 3600), int((execution_time % 3600) // 60), int(execution_time % 60)))

if __name__ == "__main__":
    main()