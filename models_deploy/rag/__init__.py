import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_deploy.rag.llamaindex.llamaindex import llamaindex
from models_deploy.rag.contriever.contriever import contriever
from models_deploy.rag.BM25.BM25 import BM25
from models_deploy.rag.openai.openai import openai 
RAG_REGISTRY = {
    "llamaindex":llamaindex,
    "BM25":BM25,
    "contriever":contriever,
    "openai":openai
}

def get_rag_method(rag):
    return RAG_REGISTRY[rag]
