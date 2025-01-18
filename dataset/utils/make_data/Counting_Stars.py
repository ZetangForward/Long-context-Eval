import json
import os


def transform_data(raw_data):
    return {
        "passage": "",
        "question": raw_data["question"],
        "choices": "",
        "answer": raw_data["reference_counting_results"],
    }

def make_data(input_path,ability,task_name):
    output_path = "./dataset/{}/data/{}.json".format(ability,task_name)
    os.makedirs("./dataset/{}/data".format(ability), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f2:
        with open(input_path, "r", encoding="utf-8") as f3:
            for line in f3:
                raw_data = json.loads(line)
                new_data = transform_data(raw_data)
                f2.write(json.dumps(new_data, ensure_ascii=False) + "\n")



