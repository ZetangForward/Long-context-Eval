#python lte/main.py --model_path /opt/data/private/models/Llama-3.1-8B-Instruct  --eval --tasks_name L-CiteEval-Data_narrativeqa,L-CiteEval-Data_natural_questions,L-CiteEval-Data_hotpotqa,L-CiteEval-Data_2wikimultihopqa,L-CiteEval-Data_gov_report,L-CiteEval-Data_multi_news,L-CiteEval-Data_qmsum,L-CiteEval-Data_locomo,L-CiteEval-Data_dialsim,L-CiteEval-Data_counting_stars,L-CiteEval-Data_niah  --device 0,1,2,3,4,5,6,7 --device_split_num 2
import os,sys
import pdb
sys.path.append(os.getcwd())
import argparse
import subprocess
from utils.request import Request
import yaml
import json
import time
import random
from tasks.instance import Instance
from collections import ChainMap
import random


from tqdm import tqdm
from Model_Deploy_URLs import get_model
from utils.utils import import_function_from_path
from tasks.postprocess import get_postprocess
import numpy as np
from utils.generation_args import handle_cli_args
import torch.multiprocessing as mp
import torch
import logging
logger = logging.getLogger('my_logger')
 
class Evaluator():
    def __init__(self,args,cfg):
        #设置参数
        self.tasks_list = []
        self.args = args
        self.limit = int(args.limit) if args.limit!="auto" else 10000
        self.build_tasks(cfg) #根据配置文件创建task

    def build_tasks(self,cfg):
        #建立对应字典用于遍历
        for name in cfg:
            task_json = cfg[name]

            #以输入key为主，合并task的配置和main set的配置参数
            merge_dict = dict(ChainMap(vars(self.args), task_json))
            args = argparse.Namespace(**merge_dict)
            self._transform_func = import_function_from_path(args.transform, "transform")
            
            #读取模型参数保存在self.sample_config
            self.sample_config = args.generate  # sampling的配置
            _params = args.generate["params"] if args.generate["params"] else args.params
            with open(_params, "r", encoding="utf-8") as f:
                self.model_args = json.load(f)
            self.sample_config["args"] = self.model_args

            #后处理函数返回【【output】【postprocessed_output】】
            with open(task_json["path"], "r", encoding="utf-8") as file:
                for index, line in enumerate(file):
                    if index>=self.limit:
                        break
                    raw_input = Instance(json.loads(line.strip()))
                    prompt_input = self.construct_input(args,name,raw_input.data)
                    self.tasks_list.append([name,args.postprocess,args.postprocess_task,Request(
                        request_type=self.sample_config["method"],
                        instances=prompt_input,
                        params=self.model_args,
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
            time.sleep(120)
            processes.append(p)
        for p in processes:
            p.join()

    def get_pred(self,i,raw_data,devices):
        #model depoly   
        k=0       
        try:
            model = get_model(self.args.acceleration)(self.args,devices)
            model_path = self.args.model_path
        except:
            model = get_model(self.args.server)(self.args,devices)
            model_path = self.args.model_path
        model.deploy()
        for task_name,postprocess,postprocess_task,request in tqdm(raw_data,desc="The {}th chunk".format(i)):
            with open("test1.txt","w") as f:
                print(request.instances["input"],file=f)
            modify = import_function_from_path("./dataset/utils/modify/general.py", "modify")
            request.instances["input"] = modify(request.instances["input"],model,model_path)
            with open("test2.txt","w") as f:
                print(request.instances["input"],file=f)
            model_postprocess = get_postprocess(postprocess)()
            task_postprocess = (get_postprocess(postprocess_task)() if postprocess_task else "")

            
            #调出模型的生成函数
            with torch.no_grad(): 
                result = model.generate(request.params, request.instances["input"])
            
            #后处理
            raw_outputs, processed_outputs = model_postprocess(result)
            if task_postprocess:
                raw_outputs, processed_outputs = task_postprocess(raw_outputs, processed_outputs)
            with open("test3.txt","w") as f:
                print(processed_outputs[0],file=f)

            # with open('output.doc', 'a') as f:
            #     print("*********{}********\n".format(k),file=f)
            #     k+=1
            #     print("模型输入:\n{}\n".format(request.instances["input"]),file=f)
            # print("模型输入:\n{}".format(request.instances["input"][-50:]))
            # print("模型输出结果:\n{}".format(raw_outputs[0]))
            # print("处理后的结果:\n{}".format(processed_outputs[0]))
            # print("答案:\n{}".format(request.instances["processed_output"]))
            
            request.raw_example.raw_outputs = raw_outputs[0]
            request.raw_example.processed_outputs = processed_outputs[0]
            request.raw_example.ground_truth = request.instances["processed_output"]
            request.raw_example.prompt_inputs = request.instances["input"]
           
            # os.makedirs("测试", exist_ok=True)
            # with open(os.path.join("测试",task_name+".json"), "a", encoding="utf-8") as f:
            #     json.dump({"task":task_name,"模型输入":request.instances["input"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth}, f, ensure_ascii=False)
            #     f.write('\n')
            # with open('output.doc', 'a') as f:
            #     print("评测文本输入:\n{}\n".format(request.raw_example.data["passage"]),file=f)
            #     print("模型预测:\n{}\n".format(request.raw_example.processed_outputs),file=f)
            #     print("正确结果:\n{}\n".format(request.raw_example.ground_truth),file=f)

            os.makedirs(self.args.generation_path, exist_ok=True)
            with open(os.path.join(self.args.generation_path,task_name+".json"), "a", encoding="utf-8") as f:
                if True:
                    json.dump({"choices":request.raw_example.data["passage"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth,"model_input":request.instances["input"]}, f, ensure_ascii=False)
                else:
                    json.dump({"choices":request.raw_example.data["choices"],"pred": request.raw_example.processed_outputs, "answers": request.raw_example.ground_truth}, f, ensure_ascii=False)
                f.write('\n')


    #完善transfrom后的数据,加description和num_fewshot
    def construct_input(self,args,name,doc):
        description = args.description + "\n\n" if args.description else ""
        example = self._transform_func(doc,name)
        return {
        "input": description  + example["input"],
        "output": example["output"],
        "processed_output": example["processed_output"],
    }

def main():
    #初始化
    seed = 0;random.seed(seed);np.random.seed(seed)
    start_time = time.time()
    args = handle_cli_args()

    #读取评测参数
    with open(r"./cogfigs/eval_configs.yaml", 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    #开始评测
    
    evaluator = Evaluator(args,cfg)
    evaluator.run(args.device_split_num)

    #eval
    if args.eval:
        command = ["python","./lte/eval.py",
                    "--generation_path",args.generation_path,
                    "--output_path",args.output_path,
                    ]
        subprocess.run(command)


    #查看运行时间
    all_time = time.time()-start_time
    
    # 计算小时、分钟和秒
    logger.info("The total time is ：{:02d}:{:02d}:{:02d}".format(int(all_time // 3600), int((all_time % 3600) // 60), int(all_time % 60)))
if __name__ == "__main__":

    main()