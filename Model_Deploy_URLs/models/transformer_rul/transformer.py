from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from peft import PeftModelForCausalLM
import torch
import os
import time 
from copy import deepcopy
import logging
logger = logging.getLogger('my_logger')

class Transformer():
    def __init__(self,args,devices):
        self.args = args
        self.devices = devices
        self.params_dict = {}

        
    def deploy(self,):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
        time_start = time.time()
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path,device_map="auto",torch_dtype=eval(self.args.torch_dtype))
        if self.args.adapter_path:
            self.model=PeftModelForCausalLM.from_pretrained(self.model ,self.args.adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,mean_resizing=False)
        
        # Check and add pad token if necessary
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info("Model and tokenizer initialized.",flush=True )
        time_cost = time.time()-time_start
        logger.info("Model_deploy time :{}".format(time_cost),flush=True)

    def generate(self,params_dict,prompt):
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)
        if "stop_token_ids" in params_:
            params_["eos_token_id"]=[self.tokenizer.eos_token_id, self.tokenizer.encode("{}".format(params_["stop_token_ids"]), add_special_tokens=False)[-1]]
            params_.pop("stop_token_ids")
        if "max_tokens" in params_:
            params_["max_new_tokens"]=params_["max_tokens"]
            params_.pop("max_tokens")

        # zecheng_note

        input = self.tokenizer(text=prompt, truncation=False, return_tensors="pt").to(self.model.device)
        context_length = input.input_ids.shape[-1]
        output = self.model.generate(
                **input,
                **params_
            )[0]
        
        del input
        torch.cuda.empty_cache()
        pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        del output
        return pred


