from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 

##init
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--params", type=str,help="Model params", required=True)
args = parser.parse_args()
params = eval(args.params)
model_path = params["model_path"]
device = params["device"]
torch_dtype = eval(params["torch_dtype"])
port = int(params["port"])
use_flash = True
os.environ["CUDA_VISIBLE_DEVICES"] = device
params_dict = {  
    "do_sample": True,
    "temperature": 0.7,
    }

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",torch_dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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



# Check and add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer initialized.",flush=True )
time_cost = time.time()-time_start
print("Model_deploy time :{}".format(time_cost),flush=True )


app = Flask(__name__)
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
    print(params_dict)

    inputs = tokenizer(prompt, truncation=False,padding=True, return_tensors="pt").to(model.device)  # Prepare the input tensor
    if "stop_token_ids" in params_dict:
        params_dict["eos_token_id"]=[tokenizer.eos_token_id, tokenizer.encode("{}".format(params_dict["stop_token_ids"]), add_special_tokens=False)[-1]]
        params_dict.pop("stop_token_ids")
    if "max_tokens" in params_dict:
        params_dict["max_new_tokens"]=params_dict["max_tokens"]
        params_dict.pop("max_tokens")
    generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **params_dict)

    # Decoding the generated ids to text
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    assert len(prompt) == len(generated_text)
    for j in range(len(prompt)):
        generated_text[j] = generated_text[j][len(prompt[j]):]
    print(generated_text)
    torch.cuda.empty_cache()
    return jsonify(generated_text)

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', port), app)
    http_server.serve_forever()