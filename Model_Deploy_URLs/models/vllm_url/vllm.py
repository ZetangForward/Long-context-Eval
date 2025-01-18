import time
import os
import torch
from copy import deepcopy
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams,TokensPrompt
import time
logger = logging.getLogger('my_logger')

class Vllm():
    def __init__(self,args, devices):
        self.args = args
        self.devices = devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device       
        self.params_dict = {  
        }
    def deploy(self):
        time_start = time.time()
        self.tokenizer =  AutoTokenizer.from_pretrained(self.args.model_path,mean_resizing=False)
        # Load the model and tokenizer
        self.model = LLM(
            model=self.args.model_path,
            trust_remote_code = True,
            tensor_parallel_size=len(self.args.device.split(",")),
        )
        # Check and add pad token if necessary
        logger.info("Model and tokenizer initialized.",flush=True )
        time_cost = time.time()-time_start
        logger.info("Model_deploy time :{}".format(time_cost),flush=True )

    def generate(self,params_dict,prompt):
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)
        if "num_beams" in params_dict:
            params_dict.pop("num_beams")
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        max_model_len = len(tokenized_prompt)+500
        outputs = self.model.generate(prompt,SamplingParams(**params_dict) ,max_model_len=max_model_len)


        generated_text = outputs[0].outputs[0].text
        res = generated_text

        torch.cuda.empty_cache()
        return res











  