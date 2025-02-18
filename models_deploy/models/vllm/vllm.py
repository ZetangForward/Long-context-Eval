import time
import os
import torch
from copy import deepcopy
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams,TokensPrompt
import time

class Vllm():
    def __init__(self,args, devices):
        self.args = args
        self.devices = devices
        self.params_dict = {
            "max_tokens": 16,
            "temperature": 1,
            "repetition_penalty": 1.0,
            "n": 1,
            "best_of": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "seed": None,
            "stop": None,
            "stop_token_ids": None,
            "bad_words": None,
            "ignore_eos": False,
            "min_tokens": 0,
            "logprobs": None,
            "prompt_logprobs": None,
            "detokenize": True,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": True,
            "logits_processors": None,
            "include_stop_str_in_output": False,
            "truncate_prompt_tokens": None,
            "output_kind": "CUMULATIVE"
        }
   
    def deploy(self):
        time_start = time.time()
        self.tokenizer =  AutoTokenizer.from_pretrained(self.args.model_path,mean_resizing=False)
        # Load the model and tokenizer
        if self.args.max_model_len:
            self.model = LLM(
                model=self.args.model_path,
                trust_remote_code = True,
                tensor_parallel_size=len(self.devices.split(",")),
                max_model_len = self.args.max_model_len)
        else:
            self.model = LLM(
                model=self.args.model_path,
                trust_remote_code = True,
                tensor_parallel_size=len(self.devices.split(",")),)
        # Check and add pad token if necessary
        logger.info("Model and tokenizer initialized.",flush=True )
        time_cost = time.time()-time_start
        logger.info("Model_deploy time :{}".format(time_cost),flush=True )

    def generate(self,params_dict,prompt):
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)
        del_list = ["do_sample","num_beams"]
        for i in del_list:
            if i in params_dict:
                params_dict.pop(i)
        outputs = self.model.generate(prompt,SamplingParams(**params_dict))


        generated_text = outputs[0].outputs[0].text
        res = generated_text

        torch.cuda.empty_cache()
        return res











  