import SelfExtend 
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import argparse
import os
import time 
from transformers import LlamaConfig, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  default="/opt/data/private/models/Llama-2-7b-hf",type=str,help="Model name on hugginface")
parser.add_argument("--device",  default="0", help="GPUid to be deployed")
parser.add_argument("--torch_dtype",default="torch.bfloat16")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained( args.model_path,  trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
     args.model_path,
     device_map="auto",
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)

prompt = "你好"
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
outputs = model.generate(**inputs, do_sample=False, top_p=1, temperature=1, max_new_tokens=20)[:, inputs["input_ids"].shape[1]:]
print(f"Input Length: {inputs['input_ids'].shape[1]}")

print(f"Prediction:   {tokenizer.decode(outputs[0], skip_special_tokens=True)}")