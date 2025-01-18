import SelfExtend 
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import argparse
import os
import time 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  default="/opt/data/private/models/vicuna-7b-v1.5",type=str,help="Model name on hugginface")
parser.add_argument("--device",  default="0", help="GPUid to be deployed")
parser.add_argument("--torch_dtype",default="torch.bfloat16")
parser.add_argument("--compress_ratio_min", type=float, default=1.2)
parser.add_argument("--compress_ratio_max", type=float, default=1.8)
parser.add_argument("--head_type", type=str, default="")
parser.add_argument("--cache_dir", type=str, default="./demo/cache")
parser.add_argument("--apply_layers", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31")
args = parser.parse_args()

window_size = 1024
group_size = 32
use_flash = True

print("Initializing model and tokenizer...",flush=True)

model_path = args.model_path
if 'Mistral' in model_path:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_path)
        config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=eval(args.torch_dtype), attn_implementation="flash_attention_2")
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=eval(args.torch_dtype), attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn")

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
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
