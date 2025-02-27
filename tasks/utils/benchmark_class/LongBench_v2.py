from tqdm import tqdm
from datasets import config
from tasks.utils.benchmark_class.base_class import Base
from datasets import load_dataset
import re

llm_param = {"temperature":0.1, "max_new_tokens":128}
llm_param1 = {"temperature":0.1, "max_new_tokens":1024}

class LongBench_v2(Base):
    def __init__(self,args,config,**kwargs):
        super().__init__()
        self.benchmark_name = "LongBench_v2"
        self.ability = "General"
        self.hf = "THUDM/LongBench-v2"
        self.download_all =False
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.args = args
        self.config = config
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.task_names = "longbench_v2"
        if self.config["cot"]:
            self.llm_params = {"longbench_v2":llm_param1}
        else:
            self.llm_params = {"longbench_v2":llm_param} 
        self.metric = {"longbench_v2":{"longbench_v2":None}} 
        self.template_rag = open('tasks/utils/demo_prompt/LongBench_v2/0shot_rag.txt', encoding='utf-8').read()
        self.template_no_context = open('tasks/utils/demo_prompt/LongBench_v2/0shot_no_context.txt', encoding='utf-8').read()
        self.template_0shot = open('tasks/utils/demo_prompt/LongBench_v2/0shot.txt', encoding='utf-8').read()
        self.template_0shot_cot = open('tasks/utils/demo_prompt/LongBench_v2/0shot_cot.txt', encoding='utf-8').read()
        self.template_0shot_cot_ans = open('tasks/utils/demo_prompt/LongBench_v2/0shot_cot_ans.txt', encoding='utf-8').read()



    def download_and_transform_data(self,**kwargs):
        check = False
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            default_cache_dir = config.HF_DATASETS_CACHE
            if not check:
                try:
                    data = load_dataset(self.hf,cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), split="train",trust_remote_code=True,download_mode="reuse_cache_if_exists")
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


    def transform(self,data,task_name,**kwargs):
        if self.config["no_context"]:
            template = self.template_no_context
        elif self.config["cot"]:
            template = self.template_0shot_cot
        else:
            template = self.template_0shot
        prompt = template.replace('$DOC$', data["passage"].strip()).replace('$Q$', data['question'].strip()).replace('$C_A$', data['choices'][0].strip()).replace('$C_B$',data['choices'][1].strip()).replace('$C_C$', data['choices'][2].strip()).replace('$C_D$', data['choices'][3].strip())
        return prompt
   
    def transform_data(self,raw_data):
        return {
            "domain": raw_data["domain"], 
            "sub_domain": raw_data["sub_domain"], 
            "difficulty": raw_data["difficulty"],
            "length": raw_data["length"],
            "passage": raw_data["context"],
            "question": raw_data["question"],
            "choices": [raw_data["choice_A"],raw_data["choice_B"],  raw_data["choice_C"], raw_data["choice_D"]],
            "label": raw_data["answer"],
        }

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
    def extract_answer(self,response):
        response = response.replace('*', '')
        match = re.search(r'The correct answer is \(([A-D])\)', response)
        if match:
            return match.group(1)
        else:
            match = re.search(r'The correct answer is ([A-D])', response)
            if match:
                return match.group(1)
            else:
                return None
    def postprocess(self,task_name,results):
        results =results.strip()
        return self.extract_answer(results)