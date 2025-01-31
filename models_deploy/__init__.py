# _note: metrics 分类 + 归并
import sys,os
from .models.transformer.transformer import Transformer
from .models.vllm.vllm import Vllm
# from .server.streaming_llm.streaming_llm import Streaming_LLM
# from .server.Snapkv.SnapKV import SnapKV
from .server.duo_attention.duo_attn import 
MODEL_REGISTRY = {
    "transformers":Transformer,
    "vllm":Vllm,
    # "streaming_llm":Streaming_LLM,
    # "SnapKV":SnapKV
}


def get_model(metric_name):
    return MODEL_REGISTRY[metric_name]
