# python lte/main.py --model_path /data/hf_models/Meta-Llama-3.1-8B-Instruct --eval  --benchmark tasks/General/LongBench_v2/LongBench_v2.yaml 
import os,sys
sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
import sys
import yaml
import time
import json
import subprocess
import shutil
from tasks.utils.benchmark_class import get_benchmark_class
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from lte.utils.request import Request
import random
from evaluation.instance import Instance
from tqdm import tqdm
from models_deploy import get_model
import numpy as np
from lte.utils.main_args import handle_cli_args
import torch.multiprocessing as mp
import torch
from models_deploy.rag import get_rag_method


class Evaluator():
    def __init__(self,args,all_benchmarks):
        #Set parameters
        self.tasks_list = []
        self.args = args
        self.limit = int(args.limit) if args.limit!="auto" else 10000
        self.build_tasks(all_benchmarks) #Create a task based on the configuration file.
        self.model_name = self.args.model_name

    def build_tasks(self,all_benchmarks):
        for benchmark in all_benchmarks:
            for task_name in benchmark.task_names:
                if self.args.rag!="":
                    if hasattr(benchmark, 'length'):
                        path = benchmark.data_path+"/"+f"{task_name}_{self.args.rag}"+f"_{benchmark.length}"+".jsonl"
                    else:
                        path = benchmark.data_path+"/"+f"{task_name}_{self.args.rag}"+".jsonl"
                else:
                    if hasattr(benchmark, 'length'):
                        path = benchmark.data_path+"/"+task_name+f"_{benchmark.length}"+".jsonl"
                    else:
                        path = benchmark.data_path+"/"+task_name+".jsonl"
                with open(path, "r", encoding="utf-8") as file:
                    for index, line in enumerate(file):
                        if index>=self.limit:
                            break
                        raw_input = Instance(json.loads(line.strip()))
                        prompt_input = benchmark.transform(raw_input.data, task_name)
                        self.tasks_list.append([task_name,benchmark,Request(
                            prompt_input=prompt_input,
                            params=benchmark.llm_params[task_name],
                            raw_example=raw_input,
                        )])

    def run(self,device_split_num):
        random.shuffle(self.tasks_list)
        print(len(self.tasks_list))
        devices_list=self.args.device.split(",")
        processes = []
        chunk_num = len(devices_list)//device_split_num
        for i in range(0, len(devices_list), device_split_num):   
            raw_data = self.tasks_list[i//device_split_num::chunk_num]
            devices = ",".join(devices_list[i: i + device_split_num])
            logger.info("devices:{}".format(devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            p = mp.Process(target=self.get_pred, args=(i//device_split_num, raw_data, devices))
            p.start()
            processes.append(p)
            time.sleep(5)
        for p in processes:
            p.join()


    def get_pred(self, i, raw_data, devices):
        #model depoly     
        try:
            model = get_model(self.args.acceleration)(self.args, devices)
            model_path = self.args.model_path
        except:
            model = get_model(self.args.server)(self.args,devices)
            model_path = self.args.model_path
        model.deploy()
        failed = 0
        for task_name, benchmark, request in tqdm(raw_data, desc="The {}th chunk".format(i)):
            request.prompt_input = benchmark.modify(
                request.prompt_input, 
                model, model_path, args=self.args
            )
            try:
                with torch.no_grad(): 
                    result = model.generate(request.params, request.prompt_input)
            except Exception as general_err:
                failed += 1
                print(f"An unexpected error occurred: {general_err}. Total failed runs: {failed} **********")

            if "sci_fi" in task_name:
                text_inputs = request.raw_example.data["question"].replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "Please directly answer without any additional output or explanation. \nAnswer:"
                result += f" [fact: {model.generate(request.params, text_inputs)}]"
            if "longbench_v2" in task_name:
                if benchmark.config["cot"]:
                    response = result.strip()
                    request.raw_example.data['response_cot'] = response
                    data = request.raw_example.data
                    prompt = benchmark.template_0shot_cot_ans.replace('$DOC$', data["passage"].strip()).replace('$Q$', data['question'].strip()).replace('$C_A$', data['choices'][0].strip()).replace('$C_B$',data['choices'][1].strip()).replace('$C_C$', data['choices'][2].strip()).replace('$C_D$', data['choices'][3].strip()).replace('$COT$', response)
                    result = model.generate({"temperature":0.1, "max_new_tokens":128}, prompt).strip()
                    

            if  hasattr(benchmark, 'postprocess'):
                if benchmark.benchmark_name=="LEval":
                    request.raw_example.data["label"],result = benchmark.postprocess(task_name,request.raw_example.data["label"],result)
                else:
                    result = benchmark.postprocess(task_name,result)
 
            if hasattr(benchmark, 'length'):
                path = os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{self.args.file_name}",task_name+f"_{benchmark.length}"+".jsonl")
                os.makedirs(os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{self.args.file_name}"), exist_ok=True)
            else:
                path = os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{self.args.file_name}",task_name+".jsonl")
                os.makedirs(os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{self.args.file_name}"), exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                data = request.raw_example.data
                data["passage"]= data["passage"]
                data["choices"] = data["choices"]
                data["pred"] = result
                data["model_input"] = request.prompt_input
                json.dump(data, f, ensure_ascii=False)
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
    current_time = time.localtime()
    formatted_time = time.strftime("%mM_%dD_%HH_%Mm", current_time)

    args = handle_cli_args()
    args.model_name = args.model_path.split("/")[-1]
    args.current_time = formatted_time
    if not args.save_tag:
        args.file_name = f"{args.model_name}_{args.current_time}"
    else:
        args.model_name = args.save_tag
        args.file_name = args.save_tag
    if len(args.device.strip()) == 0:
        gpu_count = torch.cuda.device_count()
        args.device = ','.join(map(str, range(gpu_count)))

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    if len(args.device.split(","))%args.device_split_num!=0:
        raise ValueError("The number of GPUs cannot be divided evenly by the number of blocks.")


    #task data download
    all_tasks = {}
    task_len = 0
    all_benchmarks= []
    logger.info(f"Loading the config information")
    progress_bar = tqdm(args.benchmark_config.split(","))
    for benchmark_config_path in progress_bar:
        benchmark_config_path = benchmark_config_path.strip()
        with open(benchmark_config_path, "r") as f:
            config = yaml.safe_load(f)
            benchmark_name = config["benchmark_name"].strip()
            progress_bar.set_description(f"Loading {benchmark_name} config from {benchmark_config_path}")
            if "length" in config and "num_samples" in config:
                for l in config["length"]:            
                    benchmark = get_benchmark_class(benchmark_name)(l,args)
                    benchmark.task_names = config["tasks"]
                    all_benchmarks.append(benchmark)
                    task_len += len(benchmark.task_names)
                    all_tasks[benchmark.benchmark_name]=benchmark.task_names
            else:
                benchmark = get_benchmark_class(benchmark_name)(args,config=config)
                benchmark.task_names = config["tasks"]
                all_benchmarks.append(benchmark)
                task_len += len(benchmark.task_names)
                all_tasks[benchmark.benchmark_name]=benchmark.task_names
            if hasattr(benchmark, 'length'):
                path = os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{args.file_name}")
            else:
                path = os.path.join("tasks",benchmark.ability,benchmark.benchmark_name,"prediction",f"{args.file_name}")
            if os.path.exists(path):
                try:shutil.rmtree(path)
                except:pass



    formatted_output = format_tasks(all_tasks)
    logger.info(f"The tasks you've selected are as follows:\n{formatted_output}")
    logger.info("Benchmark data is currently being downloaded and transformed...")
    progress_bar = tqdm(all_benchmarks)
    tasks_path_list = [] 
    for benchmark in progress_bar:   
        progress_bar.set_description(f"Downloading {benchmark.benchmark_name} data")
        benchmark.download_and_transform_data(args=args)
        if args.rag!="":
            data_path = benchmark.data_path
            for task_name in benchmark.task_names:
                task_path = data_path+"/"+task_name+".jsonl"
                tasks_path_list.append(task_path)
    if args.rag!="":
        with open("models_deploy/rag/rag_configs.yaml","r") as f:
            config = yaml.safe_load(f)
        if args.rag in ["raptor","llamaindex"]:
            rag = get_rag_method(args.rag)(args.model_path,tasks_path_list,config ,current_time=args.current_time,device = args.device)
            logger.info("performing information retrieval and inference")
            rag.traverse_task()   
            return 
        else:
            rag = get_rag_method(args.rag)(args.model_path,tasks_path_list,config )
            logger.info("performing information retrieval")
            rag.traverse_task()    
    

    #start to generate
    logger.info("The model has initiated the data generation process.")
    evaluator = Evaluator(args, all_benchmarks)
    evaluator.run(args.device_split_num)
    logger.info(f"All generated data has been successfully stored in tasks/'ability'/'benchmark_name'/prediction/{args.file_name}")

    #eval
    if args.eval:
        command = ["python","./lte/eval.py",
                    "--folder_name",f"{args.file_name}",
                    "--benchmark_config",f"{args.benchmark_config}",
                    "--model_name",f"{args.model_name}"]
        subprocess.run(command)

    #execution_time
    execution_time = time.time()-start_time
    logger.info("The total running time was : {:02d}:{:02d}:{:02d}".format(int(execution_time // 3600), int((execution_time % 3600) // 60), int(execution_time % 60)))

if __name__ == "__main__":
    main()