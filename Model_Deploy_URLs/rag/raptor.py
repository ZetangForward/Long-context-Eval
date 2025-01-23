
import os,sys
sys.path.append('/opt/data/private/sora/lab/lte')
import json
from tqdm import tqdm
from Model_Deploy_URLs.utils.RetrievalAugmentation import RetrievalAugmentation
from Model_Deploy_URLs.rag.bass_class import Base
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
class raptor(Base):
    def __init__(self,model_name,context,question):
        super().__init__(model_name,context,question)
        self.start_point = 86
        self.retrieval_methods = "raptor"
    def traverse_task(self):
        progress_bar = tqdm(self.tasks_path)
        for task_path in progress_bar:
            progress_bar.set_description(f"retrieving {task_path.split('/')[-1][:-5]}:")
            with open(task_path,"r", encoding="utf-8") as f:
                with open(task_path[:-5]+f"_{self.retrieval_methods}.json", "w") as f2:
                    for line in f:
                        data = json.loads(line.strip())
                        data["passage"] = self.retrieve(data["passage"],data["question"])
                        json.dump(data, f2,indent=4)  
        logger.info(f"rag_file save in {task_path[:-5]}_{self.retrieval_methods}.json")
    def retrieve(self,context,question):
        tree_count = self.args.start_point
        for item in tqdm(jsonlist[args.start_point:]):
            tree_count += 1
            tree_name = file_name+"_"+str(tree_count)
            questions = item["questions"]
            context = item["context"]
            predictions = answer_questions(tree_name, context, questions)
            item["prediction"] = predictions
            with open(args.output_path+'/'+args.model_name+'/'+file, 'a') as outfile:
                json.dump(item, outfile)
                outfile.write('\n')
    def answer_questions(self,tree, context, questions):

        # Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
        RA = RetrievalAugmentation()
        SAVE_PATH = "raptor/trees_sample_set/"+tree
        outputs = []

        if os.path.isfile(SAVE_PATH):
            RA = RetrievalAugmentation(tree=SAVE_PATH)
        else:
            RA.add_documents(context)
            RA.save(SAVE_PATH)

        #load questions:

        for question in questions:
            answer = RA.answer_question(question=question)
            outputs.append(answer)

        return outputs