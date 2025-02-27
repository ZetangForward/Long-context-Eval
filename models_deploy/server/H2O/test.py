

from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import argparse
import os
import time 
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}
ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  default="/opt/data/private/models/vicuna-7b-v1.5",type=str,help="Model name on hugginface")
parser.add_argument("--device",  default="0", help="GPUid to be deployed")
parser.add_argument("--torch_dtype",default="torch.bfloat16")
parser.add_argument("--heavy_ratio", type=float, default=0.1)
parser.add_argument("--recent_ratio", type=float, default=0.1)
args = parser.parse_args()


print("Initializing model and tokenizer...",flush=True)

model_path = args.model_path
config = AutoConfig.from_pretrained(model_path)
config.heavy_ratio = args.heavy_ratio
config.recent_ratio = args.recent_ratio
# Check and add pad token if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
model.half()
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer initialized.",flush=True )
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost),flush=True )



prompt = "你好"
inputs = tokenizer(prompt, truncation=False,padding=True, return_tensors="pt").to(model.device)  # Prepare the input tensor

generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_text)