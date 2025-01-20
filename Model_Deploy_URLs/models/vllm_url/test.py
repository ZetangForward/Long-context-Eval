from transformers import AutoTokenizer
import torch
import os
import time
from copy import deepcopy
import logging
from vllm import LLM  # 导入vllm的LLM类用于加载模型

logger = logging.getLogger('my_logger')


class Transformer():
    def __init__(self, args, devices):
        self.args = args
        self.devices = devices
        self.params_dict = {  
            "do_sample": True,
            "temperature": 0.7,
        }
        
    def deploy(self):
        """
        使用vllm的LLM来部署模型
        """
        time_start = time.time()
        # 使用vllm的LLM加载模型，配置相关参数
        self.llm = LLM(
            model=self.args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(self.args.device.split(",")),
        )
        # 加载对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        logger.info("Model and tokenizer initialized.", flush=True)
        time_cost = time.time() - time_start
        logger.info("Model_deploy time :{}".format(time_cost), flush=True)

    def generate(self, params_dict, prompt):
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)
        # 根据分词器情况处理输入提示
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True
            )
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')  # 将输入移动到GPU设备上（假设使用GPU，可根据实际调整）

        # 构建vllm生成的参数，需要根据vllm要求进行调整，这里简单示例，可按需完善
        generation_params = {
            "max_tokens": params_.get("max_new_tokens", 50),  # 假设默认生成50个新token，可根据传入参数调整
            "temperature": params_.get("temperature", 0.7),
            "do_sample": params_.get("do_sample", True)
        }

        # 使用vllm的LLM生成文本
        outputs = self.llm.generate(input_ids, **generation_params)
        output_texts = [output.text for output in outputs]

        # 这里暂只取第一个结果（如果有多个生成结果，可根据需求处理）
        pred = output_texts[0] if output_texts else ""
        return pred