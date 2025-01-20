from transformers import AutoConfig
from tqdm import tqdm
import json
import os
from dataset.utils.benchmark_class.base_class import Base
from datasets import load_dataset
import subprocess

task_list = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p","qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", "gov_report_e", "multi_news_e", "trec_e", "triviaqa_e", "samsum_e", "passage_count_e", "passage_retrieval_en_e", "lcc_e", "repobench-p_e"]
llm_params1 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 32}
llm_params2 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 64}
llm_params3 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 128}
llm_params4 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 512}
llm_params5 = {"num_beams": 1, "do_sample": False, "temperature": 1.0, "max_tokens": 64,"stop_token_ids":"\n"}
llm_param = {"narrativeqa":llm_params3, "qasper":llm_params3, "multifieldqa_en":llm_params2, "multifieldqa_zh":llm_params2, "hotpotqa":llm_params1, "2wikimqa":llm_params1, "musique":llm_params1, "dureader":llm_params3, "gov_report":llm_params4, "qmsum":llm_params4, "multi_news":llm_params4, "vcsum":llm_params4, "trec":llm_params5, "triviaqa":llm_params1, "samsum":llm_params3, "lsht":llm_params2, "passage_count":llm_params1, "passage_retrieval_en":llm_params1, "passage_retrieval_zh":llm_params1, "lcc":llm_params2, "repobench-p":llm_params2,"qasper_e":llm_params3, "multifieldqa_en_e":llm_params2, "hotpotqa_e":llm_params1, "2wikimqa_e":llm_params1, "gov_report_e":llm_params4, "multi_news_e":llm_params4, "trec_e":llm_params5, "triviaqa_e":llm_params1, "samsum_e":llm_params3, "passage_count_e":llm_params1, "passage_retrieval_en_e":llm_params1, "lcc_e":llm_params2, "repobench-p_e":llm_params2}
task_metric = {'vcsum': {'rouge_zh_score': None}, 'triviaqa_e': {'qa_f1_score': None}, 'triviaqa': {'qa_f1_score': None}, 'trec_e': {'classification_score': None}, 'trec': {'classification_score': None}, 'samsum_e': {'rouge_score': None}, 'samsum': {'rouge_score': None}, 'repobench-p_e': {'code_sim_score': None}, 'repobench-p': {'code_sim_score': None}, 'qmsum': {'rouge_score': None}, 'qasper_e': {'qa_f1_score': None}, 'qasper': {'qa_f1_score': None}, 'passage_retrieval_zh': {'retrieval_zh_score': None}, 'passage_retrieval_en_e': {'retrieval_score': None}, 'passage_retrieval_en': {'retrieval_score': None}, 'passage_count_e': {'count_score': None}, 'passage_count': {'count_score': None}, 'narrativeqa': {'qa_f1_score': None}, 'musique': {'qa_f1_score': None}, 'multifieldqa_zh': {'qa_f1_zh_score': None}, 'multifieldqa_en_e': {'qa_f1_score': None}, 'multifieldqa_en': {'qa_f1_score': None}, 'multi_news_e': {'rouge_score': None}, 'multi_news': {'rouge_score': None}, 'lsht': {'classification_score': None}, 'lcc_e': {'code_sim_score': None}, 'lcc': {'code_sim_score': None}, 'hotpotqa_e': {'qa_f1_score': None}, 'hotpotqa': {'qa_f1_score': None}, 'gov_report_e': {'rouge_score': None}, 'gov_report': {'rouge_score': None}, 'dureader': {'rouge_zh_score': None}, '2wikimqa_e': {'qa_f1_score': None}, '2wikimqa': {'qa_f1_score': None}}

class LongBench(Base):
    def __init__(self):
        super().__init__()
        self.benchmark_name = "LongBench"
        self.task_names = task_list
        self.ability = "General"
        self.hf = "THUDM/LongBench"
        self.download_all =False
        self.llm_params = llm_param
        self.metric = task_metric
        self.data_path = "dataset/General/data/"


    def make_data(self,dataset,ability,task_name):
        output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
        os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f2:
            for raw_data in dataset:
                new_data = self.transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    def download_and_transform_data(self,**kwargs):
        progress_bar = tqdm(self.task_names)
        for task_name in progress_bar:
            progress_bar.set_description(f"Downloading task: {task_name}")
            data = load_dataset(self.hf,task_name,cache_dir="./dataset/{}/tmp_Rawdata".format(self.ability), split="test",trust_remote_code=True)
            self.make_data(data,self.ability,task_name)
                


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
        return {

            "input": prompt_list[task_name].format(context=data["passage"],input=data["question"],choices = data["choices"]),
            "output": data["answer"],
            "processed_output": data["answer"],
        }
    def transform_data(self,raw_data):
        return {
            "passage": raw_data["context"],
            "question": raw_data["input"],
            "choices": raw_data["all_classes"],
            "answer": raw_data["answers"],
        }

