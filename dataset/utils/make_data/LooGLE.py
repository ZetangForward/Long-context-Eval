import json
import os


def transform_data(raw_data):
    return {
        "passage": raw_data["input"],
        "question": raw_data["qa_pairs"],
        "choices": {},
        "answer": raw_data["output"],
    }

def transform_data_qa(raw_data, qa):

    return {
        "passage": raw_data["input"],
        "question": qa["Q"],
        "choices": "",
        "answer": qa["A"],
    }

def convert(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f1:
        with open(output_path, "w", encoding="utf-8") as f2:
            for line in f1:
                raw_data = json.loads(line.strip())
                new_data = transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

def convert_qa(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f1:
        with open(output_path, "w", encoding="utf-8") as f2:
            for line in f1:
                raw_data = json.loads(line.strip())
                for qa in eval(raw_data["qa_pairs"]):
                    new_data = transform_data_qa(raw_data, qa)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")

def make_data(dataset,ability,task_name):
    output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
    os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
    if isinstance(dataset,str):
        data_path = dataset
        if task_name.endswith("summarization"):
            convert(data_path,output_path)
        else:
            convert_qa(data_path,output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f2:
            if task_name.endswith("summarization"):
                for raw_data in dataset:
                    new_data = transform_data(raw_data)
                    f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            else:
                for raw_data in dataset:
                    for qa in eval(raw_data["qa_pairs"]):
                        new_data = transform_data_qa(raw_data)
                        f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            # 更新记录文件，写入当前已写入的行数


        

