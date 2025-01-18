
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import argparse
import os
import time 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  default="/opt/data/private/models/Llama-2-7b-hf",type=str,help="Model name on hugginface")
parser.add_argument("--device",  default="0", help="GPUid to be deployed")
parser.add_argument("--torch_dtype",default="torch.bfloat16")
args = parser.parse_args()


print("Initializing model and tokenizer...",flush=True)

model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",torch_dtype=eval(args.torch_dtype))
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

prompt = "你好"
inputs = tokenizer(prompt, truncation=False,padding=True, return_tensors="pt").to(model.device)  # Prepare the input tensor

generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_text)

# Load the model and tokenizer
print("Initializing model and tokenizer...",flush=True)
if 'llama' in model_path.lower():
    from models.llama import convert_llama_model
    model = convert_llama_model(model, 4096, 10)
elif "mpt" in model_path.lower():
    from models.mpt_7b import convert_mpt_model
    model = convert_mpt_model(model, 4096, 10)
elif "gpt-j" in model_path.lower():
    from models.gpt_j import convert_gpt_j_model
    model = convert_gpt_j_model(model, 4096, 10)
else:
    raise ValueError("model illegal")
print("Model and tokenizer initialized.",flush=True )
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost),flush=True )


generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
