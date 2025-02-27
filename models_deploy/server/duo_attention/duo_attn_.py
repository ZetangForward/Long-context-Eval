from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from peft import PeftModelForCausalLM
import torch
import time 
import transformers
from copy import deepcopy
from loguru import logger
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
class duo_attn():
    def __init__(self,args,devices):
        self.args = args
        self.devices = devices
        self.params_dict = {}
        
    def deploy(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
        time_start = time.time()
        # Load the model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_path, device_map="auto", torch_dtype=eval(self.args.torch_dtype),low_cpu_mem_usage=True,attn_implementation="eager").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,mean_resizing=False)
        attn_heads, sink_size, recent_size = load_attn_pattern(
    f"duo-attention/attn_patterns/{self.args.model_path.spilt('/')[0]}/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"
)# Sparsify attention heads
        attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=0.5)

        # Enable DuoAttention
        enable_duo_attention_eval(
            self.model,
            attn_heads,
            sink_size=64,
            recent_size=256,
        )

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

        if "stop" in params_:
            params_["eos_token_id"]=[self.tokenizer.eos_token_id, self.tokenizer.encode("{}".format(params_["stop"]), add_special_tokens=False)[-1]]
            params_.pop("stop")
        if "max_tokens" in params_:
            params_["max_new_tokens"]=params_["max_tokens"]
            params_.pop("max_tokens")

        # zecheng_note
        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
            )
        
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(params_dict["max_new_tokens"] - 1):
                outputs = self.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
        pred = self.tokenizer.decode(generated_content, skip_special_tokens=True)
        torch.cuda.empty_cache()
        del input
        del output
        return pred


