

import openai
from openai import OpenAI
import numpy as np
from Model_Deploy_URLs.rag.bass_class import Base
class openai(Base):
    def __init__(self,model_name,context,question):
        super().__init__(model_name,context,question)
        self.retrieval_methods = "openai"
        self.model_name = "gpt-4o-2024-05-13"
        self.openai.api_key = "your key here"
        self.client = OpenAI(api_key = openai.api_key)
    def set_treeindex(self):
        pass
    def retrieve(self):
        chunks = self.retrieve_relevant_chunks_for_question(self.context, self.question)
        combined_chunks = " ".join(chunks)
        q = "From the context: " + combined_chunks 
        return q
    def get_embedding(self,texts, model="text-embedding-3-small"):
        response = self.client.embeddings.create(input=texts, model=model)
        openai_embeddings = []
        for i in range(len(response.data)):
            openai_embeddings.append(response.data[i].embedding)
        return openai_embeddings

    def get_cosine_similarity(self,vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    def retrieve_relevant_chunks_for_question(self,context, questions, chunk_size, num_chunks):
        context_embeddings = self.get_embedding(self.chunk_text(context, chunk_size=chunk_size))
        question_embeddings = self.get_embedding(questions) 

        relevant_chunks = []
        for question_embedding in question_embeddings:
            similarities = [self.get_cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings]
            top_indices = np.argsort(similarities)[-num_chunks:]
            top_chunks = [context[idx] for idx in top_indices]
            relevant_chunks.extend(top_chunks)
        
        return relevant_chunks
