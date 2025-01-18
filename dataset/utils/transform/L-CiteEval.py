import random
import json
def make_doc_prompt(doc, doc_id, doc_prompt):
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


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, test=False):

    if "{Q}" in prompt:
        prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    else:
        prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        doc_list = item["docs"]
        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])
        prompt = prompt.replace("{D}", text)
        
    answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
    prompt = prompt.replace("{A}", "").rstrip() + answer
    return prompt

def make_demo2(data, prompt,ndoc=None, doc_prompt=None, instruction=None, test=False):

    if "{Q}" in prompt:
        prompt = prompt.replace("{INST}", instruction).replace("{Q}", data["question"])
    else:
        prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        doc_list = data["passage"]

        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])

        prompt = prompt.replace("{D}", text)
    prompt = prompt.replace("{A}", "").rstrip() 
    return prompt

def get_instruction_template(task, prompt, sample):

    head_prompt = ""
    if task in ["dialsim"]:      
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"].replace("<<<chatbox>>>", prompt['demo_role'])
        )
    else:
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"]
        )
    head_prompt += prompt["demo_sep"]

    if task in ["dialsim"]:  
        head_prompt += make_demo2(
            sample, prompt["demo_prompt"] ,doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"].replace("<<<chatbox>>>", "Sheldon"), test=True
        )
    else:
        head_prompt += make_demo2(
            sample, prompt["demo_prompt"],doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"], test=True
        )


    return head_prompt

def transform(data,task_name,**kwargs):

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
            with open("./dataset/utils/demo_prompt/L-CiteEval/{}".format(prompt+"_default.json"), 'r') as f:
                demo_prompt = json.load(f)
            model_input= get_instruction_template(prompt, demo_prompt, data)
            return{
                "input": model_input,
                "output": data["answer"],
                "processed_output": data["answer"],
            }
    