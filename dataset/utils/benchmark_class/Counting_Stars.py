from transformers import AutoConfig
from tqdm import tqdm
import json
import os
from dataset.utils.benchmark_class.base_class import Base
from datasets import load_dataset
import subprocess

llm_param = {
    "max_tokens": 128,
    "temperature": 0,
    "top_p": 1,
    "stop_token_ids":"\n",
    "do_sample":False
}
metric1 = {"counting_stars_reasoning":None}
metric2 = {"counting_stars_searching":None}
task_download_name = {
"counting_stars_en_reasoning": "Counting_Stars_EN_multi-evidence-retrieval-reasoning_128000_32_32.jsonl" ,
"counting_stars_en_searching": "Counting_Stars_EN_multi-evidence-retrieval-searching_128000_32_32.jsonl" ,
"counting_stars_zh_reasoning": "Counting_Stars_ZH_multi-evidence-retrieval-reasoning_128000_32_32.jsonl",
"counting_stars_zh_searching": "Counting_Stars_ZH_multi-evidence-retrieval-searching_128000_32_32.jsonl"}

class Counting_Stars(Base):
    def __init__(self):
        super().__init__()
        self.benchmark_name = "Counting_Stars"
        self.task_names = ["counting_stars_en_reasoning","counting_stars_en_searching","counting_stars_zh_reasoning","counting_stars_zh_searching"]
        self.ability = "Reasoning"
        self.hf = "https://raw.githubusercontent.com/nick7nlp/Counting-Stars/refs/heads/main/test_data/"
        self.llm_params = {"counting_stars_en_reasoning":llm_param,"counting_stars_en_searching":llm_param,"counting_stars_zh_reasoning":llm_param,"counting_stars_zh_searching":llm_param}
        self.data_path = "dataset/Reasoning/data/"
        self.metric = {"counting_stars_en_reasoning":metric1,"counting_stars_en_searching":metric2,"counting_stars_zh_reasoning":metric1,"counting_stars_zh_searching":metric2}



    def make_data(self,input_path,ability,task_name):
        output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
        os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f2:
            with open(input_path, "r", encoding="utf-8") as f3:
                for line in f3:
                    raw_data = json.loads(line)
                    new_data = self.transform_data(raw_data)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def download_and_transform_data(self,**kwargs):
        for task_name in tqdm(self.task_names):
            download_path = "./dataset/{}/tmp_Rawdata/{}.json".format(self.ability,task_name)
            os.makedirs("./dataset/{}/tmp_Rawdata".format(self.ability),exist_ok=True)
            command = ["wget",self.hf+task_download_name[task_name],"-O",download_path]
            subprocess.run(command)
            self.make_data(download_path,self.ability,task_name)


    def transform_data(self,raw_data):
        return {
            "passage": "",
            "question": raw_data["question"],
            "choices": "",
            "answer": raw_data["reference_counting_results"],
        }


