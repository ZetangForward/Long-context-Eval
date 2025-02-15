# cwe,fwe,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,niah_single_1,niah_single_2,niah_single_3,qa_1,qa_2,vt
#[4096,8192,16384,32768,65536,131072]
import sys,os
import subprocess
import json
import copy
import re
import yaml
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")

max_tokens_list = [0,30,32,50,120,128]
llm_params1 = {
    "do_sample":False,
    "repetition_penalty":1,
    "top_p":1.0,
    "top_k":32,
    "temperature": 0,
    "max_tokens": 30,
}
llm_params2,llm_params3,llm_params4,llm_params5 = copy.deepcopy(llm_params1),copy.deepcopy(llm_params1),copy.deepcopy(llm_params1),copy.deepcopy(llm_params1)
for i,llm_params in enumerate([llm_params2,llm_params3,llm_params4,llm_params5]):
    llm_params["max_tokens"] = max_tokens_list[i+2]

metric_list1 = {"string_match_all": None}
metric_list2 = {"string_match_part": None}
from tasks.utils.benchmark_class.base_class import Base

class RULER(Base):
    def __init__(self,length,args,**kwargs):
        super().__init__()
        with open("./tasks/General/RULER/RULER.yaml") as f:
            config = yaml.safe_load(f)
        self.args = args
        self.limit = int(args.limit) if args.limit!="auto" else 100000
        self.length = length
        self.benchmark_name="RULER"
        self.num_samples = config["num_samples"]
        self.task_names =  ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]
        self.ability = "General"
        self.hf = None
        self.download_all =False
        self.llm_params = {"cwe":llm_params4, "fwe":llm_params3, "niah_multikey_1":llm_params5, "niah_multikey_2":llm_params5, "niah_multikey_3":llm_params5, "niah_multiquery":llm_params5, "niah_multivalue":llm_params5, "niah_single_1":llm_params5, "niah_single_2":llm_params5, "niah_single_3":llm_params5, "qa_1":llm_params2, "qa_2":llm_params2, "vt":llm_params1}
        self.metric = {'cwe': metric_list1, 'fwe': metric_list1, 'niah_multikey_1': metric_list1, 'niah_multikey_2': metric_list1, 'niah_multikey_3': metric_list1, 'niah_multiquery': metric_list1, 'niah_multivalue': metric_list1, 'niah_single_1': metric_list1, 'niah_single_2': metric_list1, 'niah_single_3': metric_list1, 'qa_1': metric_list2, 'qa_2': metric_list2, 'vt': metric_list1}
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"

    def download_and_transform_data(self,args,**kwargs):
        logger.info("downloading paulgraham_essay")
        command = ["python","./tasks/utils/data_synthetic/RULER/data/synthetic/json/download_paulgraham_essay.py"]
        subprocess.run(command)
        logger.info("downloading qa_dataset")
        command = ["bash","./tasks/utils/data_synthetic/RULER/data/synthetic/json/download_qa_dataset.sh"]
        subprocess.run(command)
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"preparing task {task_name}_{self.length}")
            command = ["python","./tasks/utils/data_synthetic/RULER/data/prepare.py",
                        "--save_dir","./tasks/General/RULER/data",
                        "--task",task_name,
                        "--tokenizer_path",args.model_path,
                        "--max_seq_length",str(self.length),
                        "--num_samples",str(self.num_samples)]
            subprocess.run(command)

            # path = "./dataset/{}/tmp_Rawdata/{}.json".format(self.ability,task_name)
            # self.make_data(path,self.ability,task_name)
        
    def postprocess(self,predict_str,**kwargs):
        predict_str = predict_str.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        predict_str = np_pattern.sub('\n', predict_str).strip()
        return predict_str

    def transform(self,data,task_name,**kwargs):

        return data["question"]
   
    def make_data(self,dataset,ability,task_name):
        output_path = "./tasks/{}/{}/data/{}.json".format(ability,self.benchmark_name,task_name)
        os.makedirs("./tasks/{}/{}/data".format(ability,self.benchmark_name), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f2:
            for index, raw_data in enumerate(dataset):
                if index>=self.limit:
                    break
                new_data = self.transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    def transform_data(raw_data):
        return {
            "passage": "",
            "question": raw_data["input"],
            "choices": "",
            "answer": raw_data["outputs"],
        }



