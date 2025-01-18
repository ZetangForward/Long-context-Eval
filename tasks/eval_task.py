import json
from utils.utils import import_function_from_path
from tasks.postprocess import get_postprocess
from tasks.instance import Instance
from tqdm import tqdm
from utils.request import Request
import os

import sys
import os
import json
import torch
from Model_Deploy_URLs import get_model
import torch.multiprocessing as mp
sys.path.append(os.path.join(os.path.abspath(__file__)))

class EvalTask:
    def __init__(self,args):
        #初始化
        self.task_name = args.task_name
        self.args = args
        
        #将task中的每条数据分装到Instance类中，保存在self.dataset中
        self.dataset = []
        # os.chdir(current_path)

        with open(args.path, "r", encoding="utf-8") as file:
            for line in file:
                self.dataset.append(Instance(json.loads(line.strip())))
        self._transform_func = import_function_from_path(args.transform, "transform")

        
        #读取模型参数保存在self.sample_config
        self.sample_config = args.generate  # sampling的配置
        _params = args.generate["params"] if args.generate["params"] else args.params
        with open(_params, "r", encoding="utf-8") as f:
            self.model_args = json.load(f)
        self.sample_config["args"] = self.model_args

        # 以task里的bath_size为主
        self.batch_size = (
            self.model_args["batch_size"]
            if self.model_args.get("batch_size")
            else args.batch_size
        )

        #后处理函数返回【【output】【postprocessed_output】】

        self.model_postprocess = get_postprocess(args.postprocess)()
        self.task_postprocess = (get_postprocess(args.postprocess_task)() if args.postprocess_task else "")
        
        #每次评估任务数目
        self.limit = int(args.limit) if args.limit!="auto" else len(self.dataset)

    def run(self,devices_list,device_split_num,model_list):
        processes = []
        for i in range(0, len(devices_list), device_split_num):
            raw_input = self.dataset[: self.limit][i::device_split_num]
            devices = ",".join(devices_list[i:i + device_split_num])
            print("devices:{}".format(devices))
            p = mp.Process(target=self.get_pred, args=(model_list,i,raw_input))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    def get_pred(self,model_list,i,raw_input):
        # try:
        #     model = get_model(self.args.acceleration)(self.args,devices)
        # except:
        #     model = get_model(self.args.server)(self.args,devices)
        # model.deploy()
        #生成完整输入
        
        prompt_input = [self.construct_input(item.data)for item in raw_input]
        print(len(prompt_input))
        #遍历完整数据发送请求
        batch_size = self.batch_size
        for j in tqdm(range(0, len(prompt_input), batch_size)):
            request = Request(
                request_type=self.sample_config["method"],
                instances=prompt_input[j : j + batch_size],
                params=self.model_args,
                raw_example=raw_input,
            )
            #调出模型的生成函数
            result = model_list[i].generate(request.params,[req["input"]+ "" for req in request.instances],self.task_name)
            torch.cuda.empty_cache()
            #后处理
            raw_outputs, processed_outputs = self.model_postprocess(result, request)

            if self.task_postprocess:
                raw_outputs, processed_outputs = self.task_postprocess(
                    raw_outputs, processed_outputs
                        )
            for index in range(len(request.raw_example)):
                print("模型输入:\n{}".format(request.instances[i]["input"][-50:]))
                print("模型输出结果:\n{}".format(raw_outputs[i]))
                print("处理后的结果:\n{}".format(processed_outputs[i]))
                print("答案:\n{}".format(request.instances[i]["processed_output"]))

                request.raw_example[index].raw_outputs = raw_outputs[index]
                request.raw_example[index].processed_outputs = processed_outputs[index]
                request.raw_example[index].ground_truth = request.instances[index]["processed_output"]
                request.raw_example[index].prompt_inputs = request.instances[index]["input"]
                os.makedirs(self.args.generation_path, exist_ok=True)
                with open(os.path.join(self.args.generation_path,self.task_name+".json"), "a", encoding="utf-8") as f:
                    json.dump({"pred": processed_outputs[index], "answers": request.instances[index]["processed_output"]}, f, ensure_ascii=False)
                    f.write('\n')


    #完善transfrom后的数据,加description和num_fewshot
    def construct_input(self,doc):
        description = self.args.description + "\n\n" if self.args.description else ""

        example = self._transform_func(doc,self.task_name)
        return {
        "input": description  + example["input"],
        "output": example["output"],
        "processed_output": example["processed_output"],
    }



