import json
import os


def transform_data(raw_data):
    return {
        "passage": "",
        "question": raw_data["input"],
        "choices": "",
        "answer": raw_data["outputs"],
    }


def make_data(dataset,ability,task_name):
    output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
    os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f2:
        for raw_data in dataset:
            new_data = transform_data(raw_data)
            f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        # 更新记录文件，写入当前已写入的行数