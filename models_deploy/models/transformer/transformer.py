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
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
        time_start = time.time()
        # Load the model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_path, attn_implementation="flash_attention_2", device_map="auto", torch_dtype=eval(self.args.torch_dtype)).eval()
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

        if self.args.template:
            prompt = self.args.template.format(user_input=prompt, assistant_response='')
        # elif hasattr(self.tokenizer, 'apply_chat_template'):
        #     prompt = self.tokenizer.apply_chat_template(
        #         [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}],
        #         tokenize=False, add_generation_prompt=True
        #     )

        if "stop" in params_:
            params_["eos_token_id"]=[self.tokenizer.eos_token_id, self.tokenizer.encode("{}".format(params_["stop"]), add_special_tokens=False)[-1]]
            params_.pop("stop")
        if "max_tokens" in params_:
            params_["max_new_tokens"]=params_["max_tokens"]
            params_.pop("max_tokens")

        # zecheng_note

        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.model.device)
        context_length = input.input_ids.shape[-1]
        output = self.model.generate(
                **input,
                **params_
            )[0]
        
        pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        torch.cuda.empty_cache()
        del input
        del output
        return pred


