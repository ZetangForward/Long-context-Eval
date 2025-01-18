import openai
import os

from openai import OpenAI
client = OpenAI(
    api_key="sk-5bSQ136a3mE5YicmP4ikgPMHUH7ZdEagVGXpNHVHi4ay194K",
    base_url="https://api.bianxie.ai/v1"
)


class GPT_Loogle_Su:
    def __init__(self,model):
        self.model = model
    def __call__(self, doc, ground_truth, results) :

        prompt_format = "There is a groundtruth summary of a arxiv paper and a auto-generated summary .Please Compare generated summary with the goundtruth and evaluate the generated summary from the perspectives of information completeness, consistency, fluency, and grammar by giving a score within the range of 0 to 100 and Please only output score(0-100) \nGroundtruth = {} \nGenerated = {} \nScore = "
        prompt = prompt_format.format(ground_truth, results)
        prompt = [{"role": "system", "content": prompt}]
        rr = client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.0,
                top_p=1,
                max_tokens=10,
                frequency_penalty=0,
                presence_penalty=0,
            )
        rsp = rr.choices[0].message.content
        try:
            return int(rsp)
        except:
            return 0
class GPT_Loogle_Qa:
    def __init__(self,model):
        self.model = model
    def __call__(self, doc, ground_truth, results) :
        results = results[0]
        p = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."
        prompt = [{"role": "system", "content": p,},
        {
            "role": "user",
            "content": "Question: "
            + doc["question"]
            + "\n"
            + "groudtruth = "
            + ground_truth
            + "\n"
            + "predict_answer = "
            + results,
        }]
        rr = client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.0,
                top_p=1,
                max_tokens=10,
                frequency_penalty=0,
                presence_penalty=0,
            )
        rsp = rr.choices[0].message.content
        if rsp=="True":
            return 0
        else:
            return 1

