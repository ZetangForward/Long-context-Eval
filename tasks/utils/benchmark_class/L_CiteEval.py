

import json
from tasks.utils.benchmark_class.base_class import Base
from datasets import load_dataset, config
from tqdm import tqdm

llm_param1 = {"max_tokens": 128,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param2 = {"max_tokens": 200,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param3 = {"max_tokens": 800,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}

metric1 = {"l_cite_eavl_cite":None}
metric2 = {"l_cite_eavl_counting_stars_cite":None}
metric3 = {"l_cite_eavl_niah_cite":None}
tasks_meta = {'L-CiteEval-Length_narrativeqa': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Length_locomo': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Length_hotpotqa': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Length_gov_report': {'llm_params': llm_param3, 'metric':{"l_cite_eavl_cite":None,"l_cite_eavl_rouge_score":None}}, 'L-CiteEval-Length_counting_stars': {'llm_params': llm_param1, 'metric':{"l_cite_eavl_counting_stars_cite":None,"l_cite_eavl_counting_stars":None}}, 'L-CiteEval-Hardness_narrativeqa': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Hardness_locomo': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Hardness_hotpotqa': {'llm_params': llm_param2, 'metric':{"l_cite_eavl_qa_score":None,"l_cite_eavl_cite":None}}, 'L-CiteEval-Hardness_gov_report': {'llm_params': llm_param3, 'metric':{"l_cite_eavl_cite":None,"l_cite_eavl_rouge_score":None}}, 'L-CiteEval-Hardness_counting_stars': {'llm_params': llm_param1, 'metric':{"l_cite_eavl_counting_stars_cite":None,"l_cite_eavl_counting_stars":None}}, 'L-CiteEval-Data_qmsum': {'llm_params': llm_param3, 'metric':metric1}, 'L-CiteEval-Data_niah': {'llm_params': llm_param1, 'metric':metric3}, 'L-CiteEval-Data_natural_questions': {'llm_params': llm_param2, 'metric':metric1}, 'L-CiteEval-Data_narrativeqa': {'llm_params': llm_param2, 'metric':metric1}, 'L-CiteEval-Data_multi_news': {'llm_params': llm_param3, 'metric':metric1}, 'L-CiteEval-Data_locomo': {'llm_params': llm_param2, 'metric':metric1}, 'L-CiteEval-Data_hotpotqa': {'llm_params': llm_param2, 'metric':metric1}, 'L-CiteEval-Data_gov_report': {'llm_params': llm_param3, 'metric':metric1}, 'L-CiteEval-Data_dialsim': {'llm_params': llm_param2, 'metric':metric1}, 'L-CiteEval-Data_counting_stars': {'llm_params': llm_param1, 'metric':metric2}, 'L-CiteEval-Data_2wikimultihopqa': {'llm_params': llm_param2, 'metric':metric1}}


class L_CiteEval(Base):
    def __init__(self, args, *configs, **kwargs):
        super().__init__()
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.benchmark_name = "L_CiteEval"
        self.ability = "Faithfulness"
        self.hf = "Jonaszky123/L-CiteEval"
        self.download_all =False
        self.tasks_meta_data = tasks_meta
        self.task_names = list(self.tasks_meta_data.keys())
        self.llm_params = {task_name:self.tasks_meta_data[task_name]["llm_params"] for task_name in self.task_names} 
        self.metric = {task_name:self.tasks_meta_data[task_name]["metric"] for task_name in self.task_names} 
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"

    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading and transforming task {task_name}")
            default_cache_dir = config.HF_DATASETS_CACHE
            try:
                if self.check_cache_exists(self.hf, task_name, default_cache_dir):
                    data = load_dataset(self.hf,task_name,split="test",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                else:
                    data = load_dataset(self.hf,task_name,cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), split="test",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                self.make_data(data,self.ability,task_name)
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                path = "./tasks/{}/{}/tmp_Rawdata/{}.jsonl".format(self.ability,self.benchmark_name,task_name)
                try:
                    self.make_data(path, self.ability, task_name)
                except Exception as inner_e:
                    print(f"cannot load {task_name}, check your network. or You can refer to the corresponding README to manually download the data")
                    raise
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
                with open("./tasks/utils/demo_prompt/L-CiteEval/{}".format(prompt+"_default.jsonl"), 'r') as f:
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
        raw_data["passage"] = raw_data["docs"]
        raw_data["choices"] = ""
        raw_data["label"] = raw_data["answer"]
        raw_data.pop("answer")
        raw_data.pop("docs")
        return raw_data