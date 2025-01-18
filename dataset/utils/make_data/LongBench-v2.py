import json
import os


def transform_data(raw_data):
    return {
        "passage": raw_data["context"],
        "question": raw_data["question"],
        "choices": [raw_data["choice_A"],raw_data["choice_B"],raw_data["choice_C"],raw_data["choice_D"]],
        "answer": raw_data["answer"],
        "_id": raw_data["_id"], 
        "domain": raw_data["domain"], 
        "sub_domain": raw_data["sub_domain"], 
        "difficulty": raw_data["difficulty"], 
        "length": raw_data["length"], 
    }
def make_data(dataset,ability,task_name):
    output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
    os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f2:
        for raw_data in dataset:
            new_data = transform_data(raw_data)
            f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        # 更新记录文件，写入当前已写入的行数
 