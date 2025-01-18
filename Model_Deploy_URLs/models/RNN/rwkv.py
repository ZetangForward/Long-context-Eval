from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="/opt/data/private/models/rwkv-6-world/RWKV-x060-World-7B-v3-20241112-ctx4096.pth",help="Model name on hugginface")
parser.add_argument("--port", type=int, default=5003, help="the port")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...")
model = RWKV(args.model_name, strategy='cpu fp32')
pipeline = PIPELINE(model, "/opt/data/private/models/rwkv-6-world/20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

#fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# download models: https://huggingface.co/BlinkDL

print("Model and tokenizer initialized.")
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost))

params_dict = {  "temperature" : 1.0, "top_p":  0.7, "top_k" : 100, # top_k = 0 then ignore
                    "alpha_frequency":0.25,
                    "alpha_presence" : 0.25,
                    "alpha_decay" : 0.996, # gradually decay the penalty
                    "token_ban" : [0], # ban the generation of some tokens
                    "token_stop" : [], # stop generation whenever you see any token here
                    "chunk_len" : 256
    }
@app.route('/infer', methods=['POST'])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompt = datas["instances"]
    print(type(prompt))
    for key, value in params.items():
        if value in params_dict:
            params_dict[key]=value

    if "stop_token_ids" ==key:
        params_dict["token_stop"].append(value)

    args = PIPELINE_ARGS(**params_dict)
    generated_text = pipeline.generate(prompt, token_count=200, args=args)
    

    return jsonify(generated_text)

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', args.port), app)
    http_server.serve_forever()