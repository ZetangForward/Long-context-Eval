from transformers import AutoConfig
from tqdm import tqdm
from datasets import config
import json
import os
from tasks.utils.benchmark_class.base_class import Base
from datasets import load_dataset
import subprocess
from lte.utils.main_args import handle_cli_args


from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train')
class LongBench(Base):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.benchmark_name = "LongBench"
        self.ability = "General"
        self.hf = "THUDM/LongBench-v2"
        self.download_all =False
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.task_names = list(self.tasks_meta_data.keys())
        self.llm_params = {task_name:self.tasks_meta_data[task_name]["llm_params"] for task_name in self.task_names} 
        self.metric = {task_name:self.tasks_meta_data[task_name]["metric"] for task_name in self.task_names} 
        template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
        template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
        template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
        template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
        template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()



    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task {task_name}")
            default_cache_dir = config.HF_DATASETS_CACHE
            try:
                if self.check_cache_exists(self.hf, default_cache_dir):
                    data = load_dataset(self.hf,split="train",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                else:
                    data = load_dataset(self.hf,cache_dir="./tasks/{}/{}/tmp_Rawdata".format(self.ability,self.benchmark_name), split="train",trust_remote_code=True,download_mode="reuse_cache_if_exists")
                self.make_data(data,self.ability,task_name)
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                path = "./tasks/{}/{}/tmp_Rawdata/{}.jsonl".format(self.ability,self.benchmark_name,task_name)
                try:
                    self.make_data(path, self.ability, task_name)
                except Exception as inner_e:
                        print(f"cannot load {task_name}, check your network. or You can refer to the corresponding README to manually download the data")
    

    def transform(self,data,task_name,**kwargs):
        if task_name[-2:] == "_e":
            task_name = task_name[:-2]
        prompt_list = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
    }
        return prompt_list[task_name].format(context=data["passage"],input=data["question"],choices = data["choices"])
   
    
    def transform_data(self,raw_data):
        return {
            "passage": raw_data["context"],
            "question": raw_data["input"],
            "choices": raw_data["all_classes"],
            "label": raw_data["answers"],
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
