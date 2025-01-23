# cwe,fwe,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,niah_single_1,niah_single_2,niah_single_3,qa_1,qa_2,vt
#[4096,8192,16384,32768,65536,131072]
from transformers import AutoConfig
import sys,os
import pdb
import subprocess
import json
import copy
import re
from tqdm import tqdm
from datasets import load_dataset
def transform_data(raw_data):
    return {
        "passage": "",
        "question": raw_data["input"],
        "choices": "",
        "answer": raw_data["outputs"],
    }


def make_data(dataset,ability,task_name):

    output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
    os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f2:
        for raw_data in dataset:
            new_data = transform_data(raw_data)
            f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
max_tokens_list = [0,30,32,50,120,128]
llm_params1 = {
    "do_sample":True,
    "repetition_penalty":1,
    "top_p":1.0,
    "top_k":32,
    "temperature": 1.0,
    "max_tokens": 30
}
llm_params2,llm_params3,llm_params4,llm_params5 = copy.deepcopy(llm_params1),copy.deepcopy(llm_params1),copy.deepcopy(llm_params1),copy.deepcopy(llm_params1)
for i,llm_params in enumerate([llm_params2,llm_params3,llm_params4,llm_params5]):
    llm_params["max_tokens"] = max_tokens_list[i+2]

metric_list1 = {"string_match_all": None}
metric_list2 = {"string_match_part": None}
from dataset.utils.benchmark_class.base_class import Base

class RULER(Base):
    def __init__(self):
        super().__init__()
    def __init__(self,length=0):
        self.length = length
        self.benchmark_name = f"RULER_{self.length}"
        self.task_names =  ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]
        self.ability = "General"
        self.hf = None
        self.download_all =False
        self.llm_params = {"cwe":llm_params4, "fwe":llm_params3, "niah_multikey_1":llm_params5, "niah_multikey_2":llm_params5, "niah_multikey_3":llm_params5, "niah_multiquery":llm_params5, "niah_multivalue":llm_params5, "niah_single_1":llm_params5, "niah_single_2":llm_params5, "niah_single_3":llm_params5, "qa_1":llm_params2, "qa_2":llm_params2, "vt":llm_params1}
        self.metric = {'cwe': metric_list1, 'fwe': metric_list1, 'niah_multikey_1': metric_list1, 'niah_multikey_2': metric_list1, 'niah_multikey_3': metric_list1, 'niah_multiquery': metric_list1, 'niah_multivalue': metric_list1, 'niah_single_1': metric_list1, 'niah_single_2': metric_list1, 'niah_single_3': metric_list1, 'qa_1': metric_list2, 'qa_2': metric_list2, 'vt': metric_list1}
        self.data_path = "dataset/General/data/"

    def download_and_transform_data(self,args,**kwargs):
        command = ["python","./dataset/utils/data_synthetic/RULER/data/synthetic/json/download_paulgraham_essay.py"]
        subprocess.run(command)
        command = ["bash","./dataset/utils/data_synthetic/RULER/data/synthetic/json/download_qa_dataset.sh"]
        subprocess.run(command)
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task: {task_name}_{self.length}")
            command = ["python","./dataset/utils/data_synthetic/RULER/data/prepare.py",
                        "--save_dir","./dataset/General/data",
                        "--task",task_name,
                        "--tokenizer_path",args.model_path,
                        "--max_seq_length",str(self.length)]
            subprocess.run(command)

            # path = "./dataset/{}/tmp_Rawdata/{}.json".format(self.ability,task_name)
            # make_data(path,self.ability,task_name)
        
    def postprocess(self,predict_str,**kwargs):
        predict_str = predict_str.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        predict_str = np_pattern.sub('\n', predict_str).strip()
        return predict_str

    def transform(self,data,task_name,**kwargs):

        return {
            "input": data["question"],
            "output": data["answer"],
            "processed_output": data["answer"],
        }


    def modify(self,prompt,model,model_path,**kwargs):
        if hasattr(model.tokenizer, 'apply_chat_template'):
            tokenized_prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True
            )
        else:
            tokenized_prompt = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        config = AutoConfig.from_pretrained(model_path)
        max_length=config.max_position_embeddings-500
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = model.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+model.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        return prompt


