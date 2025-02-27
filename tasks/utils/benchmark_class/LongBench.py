from transformers import AutoConfig
from tqdm import tqdm
import json
import os
from tasks.utils.benchmark_class.base_class import Base
from datasets import load_dataset
import subprocess
from lte.utils.main_args import handle_cli_args

llm_params1 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 32}
llm_params2 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 64}
llm_params3 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 128}
llm_params4 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 512}
llm_params5 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 64,"stop":"\n"}
tasks_meta_data = {'narrativeqa': {'llm_params': llm_params3, 'metric': {'qa_f1_score': None}}, 'qasper': {'llm_params': llm_params3, 'metric': {'qa_f1_score': None}}, 'multifieldqa_en': {'llm_params': llm_params2, 'metric': {'qa_f1_score': None}}, 'multifieldqa_zh': {'llm_params': llm_params2, 'metric': {'qa_f1_zh_score': None}}, 'hotpotqa': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, '2wikimqa': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, 'musique': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, 'dureader': {'llm_params': llm_params3, 'metric': {'rouge_zh_score': None}}, 'gov_report': {'llm_params': llm_params4, 'metric': {'rouge_score': None}}, 'qmsum': {'llm_params': llm_params4, 'metric': {'rouge_score': None}}, 'multi_news': {'llm_params': llm_params4, 'metric': {'rouge_score': None}}, 'vcsum': {'llm_params': llm_params4, 'metric': {'rouge_zh_score': None}}, 'trec': {'llm_params': llm_params5, 'metric': {'classification_score': None}}, 'triviaqa': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, 'samsum': {'llm_params': llm_params3, 'metric': {'rouge_score': None}}, 'lsht': {'llm_params': llm_params2, 'metric': {'classification_score': None}}, 'passage_count': {'llm_params': llm_params1, 'metric': {'count_score': None}}, 'passage_retrieval_en': {'llm_params': llm_params1, 'metric': {'retrieval_score': None}}, 'passage_retrieval_zh': {'llm_params': llm_params1, 'metric': {'retrieval_zh_score': None}}, 'lcc': {'llm_params': llm_params2, 'metric': {'code_sim_score': None}}, 'repobench-p': {'llm_params': llm_params2, 'metric': {'code_sim_score': None}}, 'qasper_e': {'llm_params': llm_params3, 'metric': {'qa_f1_score': None}}, 'multifieldqa_en_e': {'llm_params': llm_params2, 'metric': {'qa_f1_score': None}}, 'hotpotqa_e': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, '2wikimqa_e': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, 'gov_report_e': {'llm_params': llm_params4, 'metric': {'rouge_score': None}}, 'multi_news_e': {'llm_params': llm_params4, 'metric': {'rouge_score': None}}, 'trec_e': {'llm_params': llm_params5, 'metric': {'classification_score': None}}, 'triviaqa_e': {'llm_params': llm_params1, 'metric': {'qa_f1_score': None}}, 'samsum_e': {'llm_params': llm_params3, 'metric': {'rouge_score': None}}, 'passage_count_e': {'llm_params': llm_params1, 'metric': {'count_score': None}}, 'passage_retrieval_en_e': {'llm_params': llm_params1, 'metric': {'retrieval_score': None}}, 'lcc_e': {'llm_params': llm_params2, 'metric': {'code_sim_score': None}}, 'repobench-p_e': {'llm_params': llm_params2, 'metric': {'code_sim_score': None}}}

class LongBench(Base):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.benchmark_name = "LongBench"
        self.ability = "General"
        self.hf = "THUDM/LongBench"
        self.download_all =False
        self.data_path = f"tasks/{self.ability}/{self.benchmark_name}/data"
        self.args = args
        self.limit = int(self.args.limit) if args.limit!="auto" else 10000
        self.tasks_meta_data = tasks_meta_data
        self.task_names = list(self.tasks_meta_data.keys())
        self.llm_params = {task_name:self.tasks_meta_data[task_name]["llm_params"] for task_name in self.task_names} 
        self.metric = {task_name:self.tasks_meta_data[task_name]["metric"] for task_name in self.task_names} 


    

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
