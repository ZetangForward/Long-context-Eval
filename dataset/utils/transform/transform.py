from transformers import AutoConfig


def transform(data,task_name,**kwargs):

    return {
        "input": data["question"],
        "output": data["answer"],
        "processed_output": data["answer"],
    }
