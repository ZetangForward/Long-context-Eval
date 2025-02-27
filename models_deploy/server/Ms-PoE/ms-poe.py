from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 

import os
from utils.modify_arch.llama import MsPoELlamaForCausalLM

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",  default="/opt/data/private/models/vicuna-7b-v1.5",type=str,help="Model name on hugginface")
parser.add_argument("--device",  default="0", help="GPUid to be deployed")
parser.add_argument("--torch_dtype",default="torch.bfloat16")
parser.add_argument("--compress_ratio_min", type=float, default=1.2)
parser.add_argument("--compress_ratio_max", type=float, default=1.8)
parser.add_argument("--head_type", type=str, default="")
parser.add_argument("--cache_dir", type=str, default="./demo/cache")
parser.add_argument("--apply_layers", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...",flush=True)



config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

print('Using Ms-PoE Positional Embedding')
config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
config.compress_ratio_min = args.compress_ratio_min
config.compress_ratio_max = args.compress_ratio_max
config.head_type = args.head_type
print('Compress Ratio: from {} to {}'.format(config.compress_ratio_min, config.compress_ratio_max))
model = MsPoELlamaForCausalLM.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
# Check and add pad token if necessary

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer initialized.",flush=True )
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost),flush=True )



prompt = "你好"
inputs = tokenizer(prompt, truncation=False,padding=True, return_tensors="pt").to(model.device)  # Prepare the input tensor
generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)

# Decoding the generated ids to text
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
