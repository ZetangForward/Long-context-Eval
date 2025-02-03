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
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.benchmark_name = "Counting_Stars"
        self.task_names = ["counting_stars_en_reasoning","counting_stars_en_searching","counting_stars_zh_reasoning","counting_stars_zh_searching"]
        self.ability = "Reasoning"
        self.hf = "https://raw.githubusercontent.com/nick7nlp/Counting-Stars/refs/heads/main/test_data/"
        self.llm_params = {"counting_stars_en_reasoning":llm_param,"counting_stars_en_searching":llm_param,"counting_stars_zh_reasoning":llm_param,"counting_stars_zh_searching":llm_param}
        self.metric = {"counting_stars_en_reasoning":metric1,"counting_stars_en_searching":metric2,"counting_stars_zh_reasoning":metric1,"counting_stars_zh_searching":metric2}
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
  


    def make_data(self,input_path,ability,task_name):
        output_path = "./tasks/{}/{}/data/{}.json".format(ability,self.benchmark_name,task_name)
        os.makedirs("./tasks/{}/{}/data".format(ability,self.benchmark_name), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f2:
            with open(input_path, "r", encoding="utf-8") as f3:
                for index, line in enumerate(f3):
                    if index>=self.limit:
                            break
                    raw_data = json.loads(line)
                    new_data = self.transform_data(raw_data)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            download_path = "./tasks/{}/{}/tmp_Rawdata/{}.json".format(self.ability,self.benchmark_name,task_name)
            os.makedirs("./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name),exist_ok=True)
            command = ["wget","-c",self.hf+task_download_name[task_name],"-O",download_path]
            subprocess.run(command)
            self.make_data(download_path,self.ability,task_name)


    def transform_data(self,raw_data):
        return {
            "passage": "",
            "question": raw_data["question"],
            "choices": "",
            "answer": raw_data["reference_counting_results"],
        }


