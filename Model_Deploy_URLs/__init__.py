# _note: metrics 分类 + 归并
import sys,os
from .models.transformer_rul.transformer import Transformer
from .models.vllm_url.vllm import Vllm
MODEL_REGISTRY = {
    "transformers":Transformer,
    "vllm":Vllm
}


def get_model(metric_name):
    return MODEL_REGISTRY[metric_name]
