import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Model_Deploy_URLs.rag.llamaindex import llamaindex
# from Model_Deploy_URLs.rag.raptor import raptor
from models_deploy.rag.contriever import contriever
from models_deploy.rag.BM25 import BM25
from models_deploy.rag.openai import openai 
RAG_REGISTRY = {
    # "llamaindex":llamaindex,
    "BM25":BM25,
    "contriever":contriever,
    # "raptor":raptor,
    "openai":openai
}

def get_rag_method(rag):
    return RAG_REGISTRY[rag]
