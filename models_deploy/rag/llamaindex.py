import os

# from llama_index.readers.string_iterable import StringIterableReader
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, TreeIndex

from Model_Deploy_URLs.rag.bass_class import Base
class llamaindex(Base):
    def __init__(self,model_name,context,question):
        super().__init__(model_name,context,question)
    def set_treeindex(self,context):
        document = StringIterableReader().load_data(texts=[context])
        parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(document)
        self.index = TreeIndex(nodes)
    def retrieve(self,context,question):
        query_engine = self.index.as_query_engine(model_name=self.model_name)
        prediction = query_engine.query("Answer the question briefly with no explanation. "+question).response
        return prediction

    