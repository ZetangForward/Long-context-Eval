import random
import json
import re


def transform(data,task_name):
    prompt_list = {
    "longdep_qa":"Please answer the question based on the long texts below. If it is a multiple choice question, only the options are output\n + {input} + \nQuestion:  {question} + \nAnswer: ",
    "longdep_summarization":"Please generate a summary of the below paper. \n{input}\n Summarization:"
}
    if "longdep_qa" in task_name:
        prompt = prompt_list["longdep_qa"]
    else:
        prompt = prompt_list["longdep_summarization"]
    return {
        "input": prompt.format(input=data["passage"],question=data["question"]),
        "output": data["answer"],
        "processed_output": data["answer"],
    }

