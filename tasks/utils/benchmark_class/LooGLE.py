from transformers import AutoConfig
import sys,os
import subprocess
import json
import pdb
from tqdm import tqdm
from datasets import load_dataset



llm_params = {
    "max_tokens": 500,
    "temperature":1.0,
    "num_beams":1,
    "do_sample":False,
    "repetition_penalty":2.0
}
metric_list = {"bleu1":None,"bleu4":None,"rouge":None,"meteor_score":None,"bert_score":None}
#{"gpt":{"type":"gpt_loogle_qa","model":"gpt-3.5-turbo"}}
from tasks.utils.benchmark_class.base_class import Base

class LooGLE(Base):
    def __init__(self,limit):
        super().__init__()
        self.benchmark_name = "LooGLE"
        self.task_names = ["longdep_qa","longdep_summarization"]
        self.ability = "General"
        self.hf = "bigainlco/LooGLE"
        self.download_all =False
        self.llm_params = {"longdep_qa":llm_params,"longdep_summarization":llm_params}
        self.metric = {"longdep_qa":metric_list,"longdep_summarization":metric_list}
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.limit = int(limit) if limit!="auto" else 10000
    
    def make_data(self,dataset,ability,task_name):
        output_path = "./tasks/{}/{}/data/{}.json".format(ability,self.benchmark_name,task_name)
        os.makedirs("./tasks/{}/{}/data".format(ability,self.benchmark_name), exist_ok=True)
        if isinstance(dataset,str):
            data_path = dataset
            if task_name.endswith("summarization"):
                self.convert(data_path,output_path)
            else:
                self.convert_qa(data_path,output_path)
        else:
            with open(output_path, "w", encoding="utf-8") as f2:
                if task_name.endswith("summarization"):
                    for index, raw_data in enumerate(dataset):
                        if index>=self.limit:
                            break
                        new_data = self.transform_data(raw_data)
                        f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                else:
                    for index, raw_data in enumerate(dataset):

                        if index>=self.limit:
                            break
                        for qa in eval(raw_data["qa_pairs"]):
                            new_data = self.transform_data_qa(raw_data)
                            f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            try:
                data = load_dataset(self.hf,task_name,cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), split="test",trust_remote_code=True)
                self.make_data(data,self.ability,task_name)
            except:
                if not self.download_data:
                    self.download_data=True
                    command = ["huggingface-cli","download",
                                    "--repo-type", "dataset" ,
                                    "--resume-download",self.hf,
                                    "--local-dir","./tasks/{}/{}/tmp_Rawdata/{}".format(self.ability,self.benchmark_name,self.benchmark_name)]
                    subprocess.run(command)
                    command = ["unzip","-n","./tasks/{}/{}/tmp_Rawdata/{}/data.zip".format(self.ability,self.benchmark_name,self.benchmark_name),
                                "-d","./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name)]
                    subprocess.run(command)
                path = "./tasks/{}/{}/tmp_Rawdata/{}.json".format(self.ability,self.benchmark_name,task_name)
                self.make_data(path,self.ability,task_name)

    def transform(self,data,task_name,**kwargs):
        prompt_list = {
        "longdep_qa":"Please answer the question based on the long texts below. If it is a multiple choice question, only the options are output\n + {input} + \nQuestion:  {question} + \nAnswer: ",
        "longdep_summarization":"Please generate a summary of the below paper. \n{input}\n Summarization:"
    }
        if "longdep_qa" in task_name:
            prompt = prompt_list["longdep_qa"]
        else:
            prompt = prompt_list["longdep_summarization"]
        return {
            "input": prompt.format(input=data["passage"],question=data["question"]),
            "output": data["answer"],
            "processed_output": data["answer"],
        }
    def transform_data(self,raw_data):
        return {
            "passage": raw_data["input"],
            "question": raw_data["qa_pairs"],
            "choices": {},
            "answer": raw_data["output"],
        }

    def transform_data_qa(self,raw_data, qa):

        return {
            "passage": raw_data["input"],
            "question": qa["Q"],
            "choices": "",
            "answer": qa["A"],
        }

    def convert(self,input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f1:
            with open(output_path, "w", encoding="utf-8") as f2:
                for index, line in enumerate(f1):
                    if index>=self.limit:
                        break
                    raw_data = json.loads(line.strip())
                    new_data = self.transform_data(raw_data)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def convert_qa(self,input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f1:
            with open(output_path, "w", encoding="utf-8") as f2:
                for index, line in enumerate(f1):
                    if index>=self.limit:
                        break
                    raw_data = json.loads(line.strip())
                    for qa in eval(raw_data["qa_pairs"]):
                        new_data = self.transform_data_qa(raw_data, qa)
                        f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")




