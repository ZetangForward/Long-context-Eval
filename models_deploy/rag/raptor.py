
import os,sys
import json
from tqdm import tqdm
from models_deploy.rag.raptor_utils.RetrievalAugmentation import RetrievalAugmentation
from models_deploy.rag.bass_class import Base
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
class raptor(Base):
    def __init__(self,model_name,path_list,chunk_size,num_chunks):
        super().__init__(model_name,path_list,chunk_size,num_chunks)
        self.retrieval_methods = "raptor"
        self.path_list = path_list
        self.start_point = 86
        self.model_name = model_name
    def traverse_task(self):
        tree_count = self.start_point
        progress_bar = tqdm(self.path_list)
        for task_path in progress_bar:
            task_name = task_path.split("/")[-1][:-5]
            progress_bar.set_description(f"retrieving {task_path.split('/')[-1][:-5]}")
            with open(task_path,"r", encoding="utf-8") as f:
                with open(task_path[:-5]+f"_{self.retrieval_methods}.json", "w") as f2:
                    i = 0
                    for line in f:
                        if i<self.start_point:
                            i=i+1
                        else:
                            continue
                        tree_name = task_name+"_"+str(tree_count)
                        tree_count += 1
                        data = json.loads(line.strip())
                        data["passage"] = self.answer_questions(tree_name,data["passage"],data["question"])
                        json.dump(data, f2)  
                        f2.write('\n')
        logger.info(f"rag_file save in {task_path[:-5]}_{self.retrieval_methods}.json")

    def answer_questions(self,tree,context,question):
        # Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
        RA = RetrievalAugmentation()
        RA.qa_model =self.model_name
        SAVE_PATH = "models_deploy/rag/raptor/trees_sample_set/"+tree
        if os.path.isfile(SAVE_PATH):
            RA = RetrievalAugmentation(tree=SAVE_PATH)
            RA.qa_model =self.model_name
        else:
            RA.add_documents(context)
            RA.save(SAVE_PATH)
        answer = RA.answer_question(question=question)
        print(answer)
        return answer