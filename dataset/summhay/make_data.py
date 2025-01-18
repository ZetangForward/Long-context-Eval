import os
import json, tqdm
import tiktoken
from collections import defaultdict

def get_insights_discussed(documents, subtopic):
    subtopic_insights = set([insight['insight_id'] for insight in subtopic['insights']])
    counter = defaultdict(int)
    for doc in documents[:]:
        cur_insights_discussed = doc['insights_included']
        for insight in cur_insights_discussed:
            if insight in subtopic_insights:
                counter[insight] += 1
    return [x for x, y in counter.items()]

def get_docs_token_limit(documents, subtopic, retriever, max_retrieval_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    sorted_docs = sorted(documents, key=lambda x: subtopic["retriever"][retriever][x["document_id"]], reverse=True)
    docs_final = []
    total_tokens = 0
    for doc in sorted_docs:
        doc_str = doc['document_text']
        toks = encoding.encode(doc_str)
        if total_tokens + len(toks) >= max_retrieval_tokens:
            diff = max_retrieval_tokens - total_tokens
            toks_to_add = encoding.decode(toks[:diff])
            docs_final.append((toks_to_add, doc['idx']))
            break
        total_tokens += len(toks)
        docs_final.append((doc_str, doc['idx'])) ## assume that idx is now part of the schema
    return docs_final

def convert(input_path, output_path):
    with open(output_path, "w", encoding="utf-8") as f1:
        topic = json.load(open(input_path, "r" ))
        for subtopic in tqdm.tqdm(topic["subtopics"]):
            retrievers = subtopic["retriever"].keys()
            for retriever in retrievers:
                relev_docs = get_docs_token_limit(topic["documents"], subtopic, retriever, 800)

                if "conv" in input_path:
    
                    result = {
                                "passage": relev_docs,
                                "question": subtopic["query"],
                                "choices": {"topic": topic["topic"], "participants": topic["topic_metadata"]["participants"], "subtopic": subtopic["subtopic"], "retriever": retriever},
                                "answer": subtopic["insights"],
                            }
                else:
                    insights_discussed = get_insights_discussed(topic['documents'], subtopic)
                    num_insights_discussed = len(insights_discussed)


                    result = {
                                "passage": relev_docs,
                                "question": subtopic["query"],
                                "choices": {"topic": topic["topic"], "subtopic": subtopic["subtopic"], "retriever": retriever, "num_insights_discussed": num_insights_discussed},
                                "answer": subtopic["insights"],
                            }
                
                f1.write(json.dumps(result, ensure_ascii=False) + "\n")
    
def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(current_path)
    file_path = os.path.join(current_path,"../../RawData/{}".format(file_name))
    for data_name in os.listdir(file_path):
        output_path = os.path.join(current_path,"data/{}".format(data_name))
        os.makedirs(os.path.join(current_path,"data/"), exist_ok=True)
        convert(os.path.join(file_path,data_name), output_path)


                    

if __name__ == "__main__":
    main()
