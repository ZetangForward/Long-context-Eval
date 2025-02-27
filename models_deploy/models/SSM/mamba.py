from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gevent.pywsgi import WSGIServer
import argparse

import time 

from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

time_start = time.time()
torch_dtype=torch.float16
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="/opt/data/private/sora/models/mamba-2.8b-hf",help="Model name on hugginface")
parser.add_argument("--port", type=int, default=5003, help="the port")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...")
device = "cuda:5"

# Load the model and tokenizer
model =  MambaForCausalLM.from_pretrained(args.model_name,device_map=device,torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)



print("Model and tokenizer initialized.")
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost))
params_dict = {  
    "return_dict_in_generate":True,
    "temperature": 0.7,
    }

@app.route('/infer', methods=['POST'])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompt = datas["instances"]
    print(type(prompt))
    for key, value in params.items():
        params_dict[key]=value

    if prompt == "":
        return jsonify({'error': 'No prompt provided'}), 400

    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    if "stop_token_ids" in params_dict:
        params_dict["eos_token_id"]=[tokenizer.eos_token_id, tokenizer.encode("{}".format(params_dict["stop_token_ids"]), add_special_tokens=False)[-1]]
        params_dict.pop("stop_token_ids")
    if "max_tokens" in params_dict:
        params_dict["max_length"]=params_dict["max_tokens"]+input_ids.shape[1]
        params_dict.pop("max_tokens")
    print(params_dict)
    generate_ids = model.generate(input_ids,  **params_dict)

    # Decoding the generated ids to text
    generated_text = tokenizer.batch_decode(generate_ids.sequences.tolist())
    torch.cuda.empty_cache()
    return jsonify(generated_text[0][input_ids.shape[1]:])

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', args.port), app)
    http_server.serve_forever()