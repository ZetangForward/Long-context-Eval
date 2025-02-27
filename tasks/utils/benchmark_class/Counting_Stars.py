from tqdm import tqdm
import json
import os
from tasks.utils.benchmark_class.base_class import Base
import subprocess

llm_param = {
    "max_tokens": 128,
    "temperature": 0,
    "top_p": 1,
    "stop":"\n",
    "do_sample":False
}
metric1 = {"reasoning_acc":None}
metric2 = {"searching_acc":None}
task_download_name = {
"counting_stars_en_reasoning": "Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl" ,
"counting_stars_en_searching": "Counting_Stars_EN_multi-evidence-retrieval-searching_128000_32_32.jsonl" ,
"counting_stars_zh_reasoning": "Counting_Stars_ZH_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
"counting_stars_zh_searching": "Counting_Stars_ZH_multi-evidence-retrieval-searching_128000_32_32.jsonl"}

class Counting_Stars(Base):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.benchmark_name = "Counting_Stars"
        self.ability = "Reasoning"
        self.hf = "https://raw.githubusercontent.com/nick7nlp/Counting-Stars/refs/heads/main/test_data/"
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.tasks_meta_data = {"counting_stars_en_reasoning": {'llm_params': llm_param,'metric':metric1},"counting_stars_en_searching":{'llm_params': llm_param,'metric':metric2},"counting_stars_zh_reasoning":{'llm_params': llm_param,'metric':metric1},"counting_stars_zh_searching":{'llm_params': llm_param,'metric':metric2}}
        self.task_names = list(self.tasks_meta_data.keys())
        self.llm_params = {task_name:self.tasks_meta_data[task_name]["llm_params"] for task_name in self.task_names} 
        self.metric = {task_name:self.tasks_meta_data[task_name]["metric"] for task_name in self.task_names} 
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
  
    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            download_path = "./tasks/{}/{}/tmp_Rawdata/{}.jsonl".format(self.ability,self.benchmark_name,task_name)
            os.makedirs("./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name),exist_ok=True)
            command = ["wget","-c",self.hf+task_download_name[task_name],"-O",download_path]
            try:
                subprocess.run(command)
                self.make_data(download_path,self.ability,task_name)
            except:
                print()
                self.make_data(download_path,self.ability,task_name)
            finally:
                raise ImportError(f"cannot load {task_name}, check your network or You can refer to the corresponding README to manually download the data.")
                


    def transform_data(self,raw_data):
        return {
            "passage": "",
            "question": raw_data["question"],
            "choices": "",
            "label": raw_data["reference_counting_results"],
        }


