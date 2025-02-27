import json
import os,sys
import torch
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, TreeIndex
from llama_index.core.node_parser import SimpleNodeParser
from transformers import AutoTokenizer
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.llms.huggingface.base import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models_deploy.rag.bass_class import Base
from models_deploy.rag.bass_class import Base
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
class llamaindex(Base):
    def __init__(self,model_name,path_list,chunk_size,num_chunks,current_time,device):
        super().__init__(model_name,path_list,chunk_size,num_chunks)
        self.retrieval_methods = "llamaindex"
        self.path_list = path_list
        self.model_name = model_name
        self.current_time = current_time
        self.device = device
        Settings.llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            context_window=3900,
            max_new_tokens=256,
            generate_kwargs={"do_sample":False, "top_k": 1, "top_p": 1},
            device_map=f"cuda:{self.device}",
        )
        print(torch.cuda.is_available()) 
        # set tokenizer to match LLM
        Settings.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # set the embed model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        
    def traverse_task(self):
        progress_bar = tqdm(self.path_list)
        for task_path in progress_bar:
            task_name = task_path.split("/")[-1][:-5]
            progress_bar.set_description(f"retrieving {task_path.split('/')[-1][:-5]}")
            with open(task_path,"r", encoding="utf-8") as f:
                pred_path = os.path.join("/".join(task_path.split("/")[:-2]),"prediction",self.current_time,task_name+f"_{self.retrieval_methods}.json")
                os.makedirs(os.path.join("/".join(task_path.split("/")[:-2]),"prediction",self.current_time), exist_ok=True)
                with open(pred_path, "w") as f2:
                    for line in f:
                        data = json.loads(line.strip())
                        data["pred"] = self.retrieve(data["passage"],data["question"])
                        json.dump(data, f2)  
                        f2.write('\n')
            logger.info(f"results save in {task_path[:-5]}_{self.retrieval_methods}.json")

    def retrieve(self,context,question):
        document = StringIterableReader().load_data(texts=[context])
        parser = SimpleNodeParser.from_defaults(self.chunk_size, self.chunk_overlap)
        nodes = parser.get_nodes_from_documents(document)
        index = TreeIndex(nodes)
        query_engine = index.as_query_engine()
        prediction = query_engine.query("Answer the question briefly with no explanation. "+question).response
        return prediction

    