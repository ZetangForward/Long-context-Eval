# cwe,fwe,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,niah_single_1,niah_single_2,niah_single_3,qa_1,qa_2,vt
#[4096,8192,16384,32768,65536,131072]
import sys,os
import subprocess
import json
import yaml
import re
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")

llm_params = {
    "do_sample":False,
    "temperature": 0.0,
    "max_tokens": 50,
}
metric_list = {"niah": None}
from tasks.utils.benchmark_class.base_class import Base

class NIAH(Base):
    def __init__(self,args,config,**kwargs):
        super().__init__()
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.benchmark_name="NIAH"
        self.task_names =  ["niah"]
        self.ability = "Retrieve"
        self.hf = None
        self.download_all =False
        self.llm_params = {"niah":llm_params}
        self.metric = {"niah":metric_list}
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.config = config
        
    def download_and_transform_data(self,args,**kwargs):
        logger.info("downloading paulgraham_essay")
        command = ["python","./tasks/utils/data_synthetic/NIAH/download_paulgraham_essay.py"]
        subprocess.run(command)
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"preparing task {task_name}")
            command = ["python","./tasks/utils/data_synthetic/NIAH/synthetic_data.py",
                        "--model_path",self.args.model_path]
            subprocess.run(command)
        
    def transform(self,data,task_name,**kwargs):
        question = data['question']
        context = data["passage"]
        prompt = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {question}\nAnswer:"
        return prompt


    def modify(self, prompt, model, model_path,**kwargs):
        """Adjust input prompt to fit within the model's token limit."""
        return prompt

