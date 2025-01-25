from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gevent.pywsgi import WSGIServer
import argparse
import os
import time 

from snapkv.monkeypatch.monkeypatch import replace_llama
replace_llama() # Use monkey patches enable SnapKV
time_start = time.time()
torch_dtype=torch.float16
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,help="Model name on hugginface")
parser.add_argument("--port", type=int, help="the port")
args = parser.parse_args()

app = Flask(__name__)

print("Initializing model and tokenizer...")


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto",torch_dtype=torch.float16)
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
    "temperature": 0.1,
    }

@app.route('/infer', methods=['POST'])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompt = datas["instances"]
    for key, value in params.items():
        if key in params_dict:
            if key=="temperature" and value == 0:
                params_dict["do_sample"]=False
            else:
                params_dict[key]=value
        if key == "max_tokens":
            params_dict["max_new_tokens"] = value

    if prompt == "":
        return jsonify({'error': 'No prompt provided'}), 400
    print(params_dict)

    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)  # Prepare the input tensor

    generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **params_dict)

    # Decoding the generated ids to text
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(prompt)
    print('*********')
    print(generated_text)
    print('*********')
    assert len(prompt) == len(generated_text)
    for j in range(len(prompt)):
        generated_text[j] = generated_text[j][len(prompt[j]):]
    print(generated_text)
    torch.cuda.empty_cache()
    return jsonify(generated_text)

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', args.port), app)
    http_server.serve_forever()
