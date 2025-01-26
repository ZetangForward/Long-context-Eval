import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return [past_key_values,generated_text[pos:]]


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )[0]
    return past_key_values[1]

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 

# from snapkv.monkeypatch.monkeypatch import replace_llama
# replace_llama() # Use monkey patches enable SnapKV
time_start = time.time()
torch_dtype=torch.float16
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="/opt/data/private/sora/models/Llama-3-8B-Instruct-Gradient-1048k",help="Model name on hugginface")
parser.add_argument("--port", type=int, default=5003, help="the port")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...")
start_size = 4
recent_size = 2000

# Load the model and tokenizer

model, tokenizer = load(args.model_name)


kv_cache = enable_streaming_llm(
    model, start_size=start_size, recent_size=recent_size
)



params_dict = {  
    "do_sample": True,
    "temperature": 0.7,
    }

@app.route('/infer', methods=['POST'])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompt = datas["instances"]
    for key, value in params.items():
        params_dict[key]=value

    generated_text=streaming_inference(
        model,
        tokenizer,
        prompt,
        kv_cache,
    )


    print(generated_text)
    torch.cuda.empty_cache()
    return jsonify(generated_text)

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', args.port), app)
    http_server.serve_forever()