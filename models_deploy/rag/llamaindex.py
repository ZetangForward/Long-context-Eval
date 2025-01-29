import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from models_deploy.rag.bass_class import Base
class llamaindex(Base):
    def __init__(self,model_name,path_list,chunk_size,num_chunks):
        super().__init__(model_name,path_list,chunk_size,num_chunks)
    def retrieve(self,context,question):
        documents = SimpleDirectoryReader("data").load_data()
        # bge-base embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        # ollama
        Settings.llm = Ollama(model="llama3", request_timeout=360.0)
        index = VectorStoreIndex.from_documents(
            documents,
        )
        return prediction


from models_deploy.rag.bass_class import Base
class llamaindex(Base):
    def __init__(self,model_name,path_list,chunk_size,num_chunks):
        super().__init__(model_name,path_list,chunk_size,num_chunks)
    def retrieve(self,context,question):
        document = StringIterableReader().load_data(texts=[context])
        parser = SimpleNodeParser.from_defaults(self.chunk_size, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(document)
        self.index = TreeIndex(nodes)
        query_engine = index.as_query_engine(model_name=model_name)
        query_engine = self.index.as_query_engine(model_name=self.model_name)
        prediction = query_engine.query("Answer the question briefly with no explanation. "+question).response
        return prediction

    