from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModelForCausalLM
import torch
import time 
import transformers
from copy import deepcopy
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")

class Transformer():
    def __init__(self,args,devices):
        self.args = args
        self.devices = devices
        self.params_dict = {}
        
    def deploy(self):
        time_start = time.time()

        # Load the model and tokenizer
        model_config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_path, 
            config = model_config,
            attn_implementation = "flash_attention_2", 
            device_map="auto", 
            torch_dtype=eval(self.args.torch_dtype),
            trust_remote_code=True,
        ).eval()
        
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

    def generate(self, params_dict, prompt):
        
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)

        if "stop" in params_:
            params_["eos_token_id"]=[self.tokenizer.eos_token_id, self.tokenizer.encode("{}".format(params_["stop"]), add_special_tokens=False)[-1]]
            params_.pop("stop")
        if "max_tokens" in params_:
            params_["max_new_tokens"]=params_["max_tokens"]
            params_.pop("max_tokens")
        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.model.device)

        context_length = input.input_ids.shape[-1]

        if "glm" in self.args.model_path:
            params_ = {"max_new_tokens":params_["max_new_tokens"],"do_sample": False}

        output = self.model.generate(
                **input,
                **params_
            )[0]

        pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        torch.cuda.empty_cache()
        del input
        del output
        return pred


