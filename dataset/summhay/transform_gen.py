import random
import json
import re



def transform(data,task_name):
    relev_docs_str = ""
    for doc in data["passage"]:
        if "conv" in task_name:
            relev_docs_str += f"Conversation {doc[1]+1}:\n{doc[0]}\n\n"
        else:
            relev_docs_str += f"Article {doc[1]+1}:\n{doc[0]}\n\n"

    if "conv" in task_name:
        prompt_summarization = open("/nvme1/xii/3090_summhay/datasets/summhay/prompts/summarization_conv.txt").read()
        prompt_summarization_populated = prompt_summarization.replace("[N_conversations]", str(len(data["passage"]))).replace("[SCENARIO]", data["choices"]["topic"]).replace("[PARTICIPANTS]", ", ".join(data["choices"]["participants"])).replace("[CONVERSATIONS]", relev_docs_str).replace("[QUERY]", data["question"]).replace("[N_insights]", str(len(data["answer"])))
    else:
        prompt_summarization = open("/nvme1/xii/3090_summhay/datasets/summhay/prompts/summarization_news.txt").read()        
        prompt_summarization_populated = prompt_summarization.replace("[N_articles]", str(len(data["passage"]))).replace("[ARTICLES]", relev_docs_str).replace("[TOPIC]", data["choices"]['topic']).replace("[N_insights]", str(data["choices"]["num_insights_discussed"])).replace("[SUBTOPIC]", data['choices']['subtopic'])

    return {
        "input": prompt_summarization_populated,
        "output": data["answer"],
        "processed_output": data["answer"],
    }