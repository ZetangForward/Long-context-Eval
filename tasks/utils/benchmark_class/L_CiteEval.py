from transformers import AutoConfig
from tqdm import tqdm
import json
import os
from tasks.utils.benchmark_class.base_class import Base
from datasets import load_dataset, config
import subprocess

llm_param1 = {"max_tokens": 128,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param2 = {"max_tokens": 200,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param3 = {"max_tokens": 800,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param4 = {"max_tokens": 1000,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}

metric1 = {"l_cite_eavl_cite":None}
metric2 = {"l_cite_eavl_counting_stars_cite":None}
metric3 = {"l_cite_eavl_niah_cite":None}

class L_CiteEval(Base):
    def __init__(self, args, *configs, **kwargs):
        super().__init__()
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.benchmark_name = "L_CiteEval"
        self.task_names = ['L-CiteEval-Length_narrativeqa', 'L-CiteEval-Length_locomo', 'L-CiteEval-Length_hotpotqa', 'L-CiteEval-Length_gov_report', 'L-CiteEval-Length_counting_stars', 'L-CiteEval-Hardness_narrativeqa', 'L-CiteEval-Hardness_locomo', 'L-CiteEval-Hardness_hotpotqa', 'L-CiteEval-Hardness_gov_report', 'L-CiteEval-Hardness_counting_stars', 'L-CiteEval-Data_qmsum', 'L-CiteEval-Data_niah', 'L-CiteEval-Data_natural_questions', 'L-CiteEval-Data_narrativeqa', 'L-CiteEval-Data_multi_news', 'L-CiteEval-Data_locomo', 'L-CiteEval-Data_hotpotqa', 'L-CiteEval-Data_gov_report', 'L-CiteEval-Data_dialsim', 'L-CiteEval-Data_counting_stars', 'L-CiteEval-Data_2wikimultihopqa']
        
        self.ability = "Factuality"
        self.hf = "Jonaszky123/L-CiteEval"
        self.download_all =False
        
        self.llm_params = {'L-CiteEval-Length_narrativeqa':llm_param2, 'L-CiteEval-Length_locomo':llm_param2, 'L-CiteEval-Length_hotpotqa':llm_param2, 'L-CiteEval-Length_gov_report':llm_param3, 'L-CiteEval-Length_counting_stars':llm_param1, 'L-CiteEval-Hardness_narrativeqa':llm_param2, 'L-CiteEval-Hardness_locomo':llm_param2, 'L-CiteEval-Hardness_hotpotqa':llm_param2, 'L-CiteEval-Hardness_gov_report':llm_param3, 'L-CiteEval-Hardness_counting_stars':llm_param1, 'L-CiteEval-Data_qmsum':llm_param3, 'L-CiteEval-Data_niah':llm_param1, 'L-CiteEval-Data_natural_questions':llm_param2, 'L-CiteEval-Data_narrativeqa':llm_param2, 'L-CiteEval-Data_multi_news':llm_param3, 'L-CiteEval-Data_locomo':llm_param2, 'L-CiteEval-Data_hotpotqa':llm_param2, 'L-CiteEval-Data_gov_report':llm_param3, 'L-CiteEval-Data_dialsim':llm_param2, 'L-CiteEval-Data_counting_stars':llm_param1, 'L-CiteEval-Data_2wikimultihopqa':llm_param2}
        
        self.metric = {'L-CiteEval-Length_narrativeqa':metric1, 'L-CiteEval-Length_locomo':metric1, 'L-CiteEval-Length_hotpotqa':metric1, 'L-CiteEval-Length_gov_report':metric1, 'L-CiteEval-Length_counting_stars':metric2, 'L-CiteEval-Hardness_narrativeqa':metric1, 'L-CiteEval-Hardness_locomo':metric1, 'L-CiteEval-Hardness_hotpotqa':metric1, 'L-CiteEval-Hardness_gov_report':metric1, 'L-CiteEval-Hardness_counting_stars':metric2, 'L-CiteEval-Data_qmsum':metric1, 'L-CiteEval-Data_niah':metric3, 'L-CiteEval-Data_natural_questions':metric1, 'L-CiteEval-Data_narrativeqa':metric1, 'L-CiteEval-Data_multi_news':metric1, 'L-CiteEval-Data_locomo':metric1, 'L-CiteEval-Data_hotpotqa':metric1, 'L-CiteEval-Data_gov_report':metric1, 'L-CiteEval-Data_dialsim':metric1, 'L-CiteEval-Data_counting_stars':metric2, 'L-CiteEval-Data_2wikimultihopqa':metric1}
        
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
    
    
    def make_data(self,dataset,ability,task_name):
        output_path = "./tasks/{}/{}/data/{}.json".format(ability,self.benchmark_name,task_name)
        os.makedirs("./tasks/{}/{}/data".format(ability,self.benchmark_name), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f2:
             for index, raw_data in enumerate(dataset):
                if index>=self.limit:
                    break
                new_data = self.transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Loading task {task_name}")
            try:
                data = load_dataset(
                    self.hf, task_name, split="test",
                    trust_remote_code=True,
                    download_mode="reuse_cache_if_exists"
                )
            except:
                data = load_dataset(
                    self.hf, task_name, cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), 
                    split="test", trust_remote_code=True, 
                    download_mode="reuse_cache_if_exists"
                )
            finally:
                raise ImportError(f"cannot load {task_name}, check your network")
            
            self.make_data(data,self.ability,task_name)
    
    def transform(self,data,task_name,**kwargs):

        prompt_list = {
            "2wikimultihopqa":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer: ",
            "counting_stars":"{D}\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting \u2605. Please help the little penguin collect the correct number of \u2605 and cite the corresponding passage ID where the counting is mentioned, for example: {{'little_penguin': [x, x, x,...], 'passage_id': [y, y, y,...]}}. The summation is not required. The numbers in [x, x, x,...] represent the correctly counted number of \u2605 by the little penguin and the number in [y, y, y,...] represent the passage IDs where these counts are recorded. Only output the results in JSON format without any explanation.\nAnswer:",
            "dialsim":"{D}\n\nYou are <<<chatbox>>>, a long-term conversation agent capable of interacting with multiple users. Write an accurate, engaging, and concise answer to the given question using only the provided conversations (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one conversation and at most three. When citing several conversations, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple conversations support the sentence, only cite a minimum sufficient subset of the conversations.\n\nQuestion: {Q}\nAnswer:",
            "gov_report":"{D}\n\nWrite a concise and engaging summary of the provided passages. Use a neutral and informative tone. Every sentence in the summary must include a citation at the end, referencing at least one passage and at most three. When citing several passages in a single sentence, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support a sentence, only cite the minimum sufficient subset of the passages necessary to substantiate the information.\nSummary:",
            "hotpotqa":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
            "locomo":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided conversations (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one conversation and at most three. When citing several conversations, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple conversations support the sentence, only cite a minimum sufficient subset of the conversations.\n\nQuestion: {Q}\nAnswer:",
            "multi_news":"{D}\n\nWrite a concise and engaging summary of the provided passages. Use a neutral and informative tone. Every sentence in the summary must include a citation at the end, referencing at least one passage and at most three. When citing several passages in a single sentence, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support a sentence, only cite the minimum sufficient subset of the passages necessary to substantiate the information.\nSummary:",
            "narrativeqa":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
            "natural_questions":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
            "niah":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
            "qmsum":"{D}\n\nWrite a concise and engaging summary of the provided passages. Use a neutral and informative tone. Every sentence in the summary must include a citation at the end, referencing at least one passage and at most three. When citing several passages in a single sentence, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support a sentence, only cite the minimum sufficient subset of the passages necessary to substantiate the information.\n\nQuery: {Q}\nSummary:",
        }
    
        for prompt in prompt_list:
            if prompt in task_name:
                with open("./tasks/utils/demo_prompt/L-CiteEval/{}".format(prompt+"_default.json"), 'r') as f:
                    demo_prompt = json.load(f)
                model_input= self.get_instruction_template(prompt, demo_prompt, data)
                return model_input
      
    def make_doc_prompt(self,doc, doc_id, doc_prompt):
        if type(doc) == str:
            text = doc
        elif type(doc) == dict:
            if 'title' in doc:
                title = doc['title']
                text = doc['text'].strip('\n')
                if text[:len(title)+1] == title + '\n':
                    text = text[len(title)+1:]
            else:
                text = doc['text'].strip('\n')

        return doc_prompt.replace("{P}", text).replace("{ID}", str(doc_id+1))


    def make_demo(self,item, prompt, ndoc=None, doc_prompt=None, instruction=None, test=False):

        if "{Q}" in prompt:
            prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
        else:
            prompt = prompt.replace("{INST}", instruction)
        if "{D}" in prompt:
            doc_list = item["docs"]
            text = "".join([self.make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)
            
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
        return prompt

    def make_demo2(self,data, prompt,ndoc=None, doc_prompt=None, instruction=None, test=False):

        if "{Q}" in prompt:
            prompt = prompt.replace("{INST}", instruction).replace("{Q}", data["question"])
        else:
            prompt = prompt.replace("{INST}", instruction)
        if "{D}" in prompt:
            doc_list = data["passage"]

            text = "".join([self.make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])

            prompt = prompt.replace("{D}", text)
        prompt = prompt.replace("{A}", "").rstrip() 
        return prompt

    def get_instruction_template(self,task, prompt, sample):

        head_prompt = ""
        if task in ["dialsim"]:      
            head_prompt += self.make_demo(
                prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"].replace("<<<chatbox>>>", prompt['demo_role'])
            )
        else:
            head_prompt += self.make_demo(
                prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"]
            )
        head_prompt += prompt["demo_sep"]

        if task in ["dialsim"]:  
            head_prompt += self.make_demo2(
                sample, prompt["demo_prompt"] ,doc_prompt=prompt["doc_prompt"],
                instruction=prompt["instruction"].replace("<<<chatbox>>>", "Sheldon"), test=True
            )
        else:
            head_prompt += self.make_demo2(
                sample, prompt["demo_prompt"],doc_prompt=prompt["doc_prompt"],
                instruction=prompt["instruction"], test=True
            )
        return head_prompt
    def transform_data(self,raw_data):
        return {
            "passage": raw_data["docs"],
            "question": raw_data["question"],
            "choices":"",
            "answer": raw_data["answer"],
        }
