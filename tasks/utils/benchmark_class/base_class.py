from transformers import AutoConfig

from datasets import config
from lte.utils.main_args import handle_cli_args
from datasets import load_dataset_builder
import os
import pdb
import subprocess
from datasets import load_dataset
from tqdm import tqdm
import json
class Base:
    def __init__(self,**kwargs):
        self.benchmark_name = None
        self.task_names = None
        self.ability = None
        self.hf_repo = None
        self.data_path = None
        self.download_all = False
        self.download_data =False
        self.llm_params = {}
        self.metric = {}
        self.limit=1000000
    
    def download_and_transform_data(self,**kwargs):
        """Placeholder for the data-making logic."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def transform(self, data, task_name,**kwargs):
        return  data["question"]
    def modify(self, prompt, model, model_path,args,**kwargs):
        """Adjust input prompt to fit within the model's token limit."""
        if args.template:
            prompt = args.template.format(user_input=prompt)
            tokenized_prompt = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        elif hasattr(model.tokenizer, 'apply_chat_template') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template:
            tokenized_prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True
            )
        else:
            tokenized_prompt = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        max_length = args.max_lenth
        
        if len(tokenized_prompt) > max_length:
            half = max_length // 2
            prompt = (
                model.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                model.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )
        else:
            prompt = model.tokenizer.decode(tokenized_prompt, skip_special_tokens=True) 
        return prompt
    def check_cache_exists(self, hf, task_name, cache_dir):
        builder = load_dataset_builder(hf, name=task_name, cache_dir=cache_dir)
        cache_path = builder._cache_dir
        if os.path.exists(cache_path):
            for root, dirs, files in os.walk(cache_path):
                if files:
                    return True
        return False
    def convert_from_path(self,input_path, output_path,**kwargs):
        with open(input_path, "r", encoding="utf-8") as f1:
            with open(output_path, "w", encoding="utf-8") as f2:
                for index, line in enumerate(f1):
                    if index>=self.limit:
                        break
                    raw_data = json.loads(line.strip())
                    new_data = self.transform_data(raw_data)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    def convert_from_datasets(self,dataset, output_path):
        with open(output_path, "w", encoding="utf-8") as f2:
            for index, raw_data in enumerate(dataset):
                if index>=self.limit:
                    break
                new_data = self.transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    def make_data(self,dataset,ability,task_name):
        output_path = "./tasks/{}/{}/data/{}.jsonl".format(ability,self.benchmark_name,task_name)
        os.makedirs("./tasks/{}/{}/data".format(ability,self.benchmark_name), exist_ok=True)
        if isinstance(dataset,str):
            data_path = dataset
            self.convert_from_path(data_path,output_path)
        else:
            self.convert_from_datasets(dataset,output_path)
    def download():
        print("Connection refused>")
        return 
    def download_and_transform_data(self,**kwargs):
        check = False
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            default_cache_dir = config.HF_DATASETS_CACHE
            if not check:
                try:
                    if self.check_cache_exists(self.hf, task_name, default_cache_dir):
                        data = load_dataset(self.hf,task_name,split="test",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                    else:
                        data = load_dataset(self.hf,task_name,cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), split="test",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                except Exception as e:
                    print(f"\n{type(e).__name__}: {e}")
                    print(f"cannot load {task_name}, check your network. or You can refer to the corresponding README to manually download the data")
                try:
                    data = "./tasks/{}/{}/tmp_Rawdata/{}.jsonl".format(self.ability,self.benchmark_name,task_name)
                    self.make_data(data,self.ability,task_name)
                    check = True
                except:
                    raise
            else:
                data = "./tasks/{}/{}/tmp_Rawdata/{}.jsonl".format(self.ability,self.benchmark_name,task_name)
                self.make_data(data,self.ability,task_name)
       
            
   