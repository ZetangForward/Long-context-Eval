
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
time_start = time.time()
torch_dtype=torch.float16
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="/opt/data/private/sora/models/Llama-3-8B-Instruct-Gradient-1048k",help="Model name on hugginface")
parser.add_argument("--port", type=int, default=5005, help="the port")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...")


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="cuda:6",torch_dtype=torch.bfloat16,  low_cpu_mem_usage=True,attn_implementation="eager")
# Load the attention pattern
attn_heads, sink_size, recent_size = load_attn_pattern(
    "/opt/data/private/sora/lab/gits/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
)

# Sparsify attention heads
attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=0.5)

# Enable DuoAttention
enable_duo_attention_eval(
    model,
    attn_heads,
    sink_size=64,
    recent_size=256,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Check and add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer initialized.")
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost))
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

    if prompt == "":
        return jsonify({'error': 'No prompt provided'}), 400
    max_length = 32000
    tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:6")
    with torch.no_grad():
        output = model(
            input_ids=input.input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]
        for _ in range(params_dict["max_new_tokens"] - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content += [pred_token_idx.item()]

    pred = tokenizer.decode(generated_content, skip_special_tokens=True)

    return jsonify(pred )


if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', args.port), app)
    http_server.serve_forever()