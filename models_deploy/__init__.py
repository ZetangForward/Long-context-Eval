# _note: metrics 分类 + 归并

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .models.transformer.transformer import Transformer

# from .server.streaming_llm.streaming_llm import Streaming_LLM
# from .server.duo_attention.duo_attn_ import duo_attn
# from .server.Snapkv.SnapKV import SnapKV
# from .server.SelfExtend.selfextend_ import SelfExtend
from .models.vllm.vllm import Vllm

MODEL_REGISTRY = {
    "transformers":Transformer,
    "vllm":Vllm,
    # "streaming_llm":Streaming_LLM,
    # "SnapKV":SnapKV,
    # "duo_attn":duo_attn,
    # "SelfExtend":SelfExtend
}


def get_model(metric_name):
    return MODEL_REGISTRY[metric_name]
