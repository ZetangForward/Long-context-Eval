
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
from tqdm import tqdm
import pdb
import torch
from raptor_utils.RetrievalAugmentation import RetrievalAugmentation
from bass_class import Base
from sentence_transformers import SentenceTransformer
from raptor_utils.SummarizationModels import BaseSummarizationModel
from raptor_utils.QAModels import BaseQAModel
from raptor_utils.EmbeddingModels import BaseEmbeddingModel
from raptor_utils.RetrievalAugmentation import RetrievalAugmentationConfig
from transformers import AutoTokenizer, pipeline
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")

class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="google/gemma-2b-it",device="cpu"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device(f'cuda:{device}')
        )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)
    def create_embedding(self, text):
        return self.model.encode(text)
class GEMMAQAModel(BaseQAModel):
    def __init__(self, model_name= "google/gemma-2b-it",device="cpu"):
        # Initialize the tokenizer and the pipeline for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device(f'cuda:{device}')
        )
    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        messages=[
              {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the answer using the pipeline
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated answer
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer

class raptor(Base):
    def __init__(self,model_name,path_list,chunk_size,num_chunks,current_time,device):
        super().__init__(model_name,path_list,chunk_size,num_chunks)
        self.retrieval_methods = "raptor"
        self.path_list = path_list
        self.model_name = model_name
        self.current_time = current_time
        self.device = device
    def traverse_task(self):
        progress_bar = tqdm(self.path_list)
        for task_path in progress_bar:
            task_name = task_path.split("/")[-1][:-5]
            progress_bar.set_description(f"retrieving {task_path.split('/')[-1][:-5]}")
            with open(task_path,"r", encoding="utf-8") as f:
                pred_path = os.path.join("/".join(task_path.split("/")[:-2]),"prediction",self.current_time,task_name+f"_{self.retrieval_methods}.json")
                os.makedirs(os.path.join("/".join(task_path.split("/")[:-2]),"prediction",self.current_time), exist_ok=True)
                with open(pred_path, "w") as f2:
                    tree_count =0
                    for line in f:
                        tree_name = task_name+"_"+str(tree_count)
                        data = json.loads(line.strip())
                        data["pred"] = self.answer_questions(tree_name,data["passage"],data["question"])
                        json.dump(data, f2)  
                        f2.write('\n')
            logger.info(f"results save in {task_path[:-5]}_{self.retrieval_methods}.json")

    def answer_questions(self,tree,context,question):

        RAC = RetrievalAugmentationConfig(summarization_model=GEMMASummarizationModel(self.model_name,self.device), qa_model=GEMMAQAModel(self.model_name,self.device), embedding_model=SBertEmbeddingModel())
    
        # Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
        RA = RetrievalAugmentation(config=RAC)
        SAVE_PATH = "models_deploy/rag/raptor/trees_sample_set/"+tree
        if os.path.isfile(SAVE_PATH):
            RA = RetrievalAugmentation(tree=SAVE_PATH)
        else:
            RA.add_documents(context)
            RA.save(SAVE_PATH)
        answer = RA.answer_question(question=question)
        return answer
    
   