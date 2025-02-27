
import torch
import time 
from copy import deepcopy
from loguru import logger
import sys,os
import warnings
warnings.filterwarnings("ignore")
from models_deploy.server.streaming_llm.streaming_llm_utils.utils import load, download_url, load_jsonl
from models_deploy.server.streaming_llm.streaming_llm_utils.enable_streaming_llm import enable_streaming_llm

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return [past_key_values,generated_text[pos:]]


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )[0]
    return past_key_values[1]
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")

class Streaming_LLM():
    def __init__(self,args,devices):
        self.args = args
        self.devices = devices
        self.params_dict = {}
        self.start_size = 4
        self.recent_size = 2000  
        
    def deploy(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.devices
        time_start = time.time()
        # Load the model and tokenizer
        self.model, self.tokenizer = load(self.args.model_path)
        self.kv_cache = enable_streaming_llm(
        self.model, start_size=self.start_size, recent_size=self.recent_size
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

        generated_text=streaming_inference(
        self.model,
        self.tokenizer,
        [prompt],
        self.kv_cache,
    )
        torch.cuda.empty_cache()
        return generated_text


