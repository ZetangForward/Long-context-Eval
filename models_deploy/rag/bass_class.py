import sys
import json
import re
from loguru import logger
from tqdm import tqdm
import pdb
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
class Base:
    def __init__(self,model_name,path_list,config ,**kwargs):
        self.model_name = model_name
        self.tasks_path = path_list
        self.config  = config 
        self.chunk_size = config["chunk_size"]
        self.num_chunks = config["num_chunks"]
        self.chunk_overlap = config["chunk_overlap"]
    def traverse_task(self):
        progress_bar = tqdm(self.tasks_path)
        for task_path in progress_bar:
            progress_bar.set_description(f"retrieving {task_path.split('/')[-1][:-5]}")
            with open(task_path,"r", encoding="utf-8") as f:
                with open(task_path[:-5]+f"_{self.retrieval_methods}.json", "w") as f2:
                    for line in f:
                        data = json.loads(line.strip())
                        data["passage"] = self.retrieve(data["passage"],data["question"])
                        json.dump(data, f2)  
                        f2.write('\n')
        logger.info(f"rag_file save in {task_path[:-5]}_{self.retrieval_methods}.json")
    def set_treeindex(self,context):
        pass
    def retrieve(self):
        pass
    def chunk_text(self,context):
        sentences = re.split(r'(?<=[.!?]) +', context)
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    