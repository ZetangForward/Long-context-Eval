import os
import warnings

# use this for now to filter out torch warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle
import textwrap
import sys
from squeezedattention.utils import build_chat, truncate_fn

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
                                                                    "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k",
                                                                    "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "LLaMA-2-7B-32K",
                                                                    "LWM-Text-Chat-1M"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--path_to_clusters", type=str, default="/tmp")
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--percentile_lower", type=float, default=0.7)
    parser.add_argument("--obs_window", type=int, default=100)
    parser.add_argument("--task", type=str, default=None)
    return parser.parse_args(args)

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, config_params)

    # iterate over longbench dataset
    for json_obj in tqdm(data):
        different_prefix_index = json_obj.pop('different_prefix_index')
        prompt = prompt_format.format(**json_obj)
        prompt_noquery = prompt_only_format.format(**json_obj)

        # perform truncation
        prompt, truncated_shared_prefix_length = truncate_fn(prompt, prompt_noquery, tokenizer, max_length, dataset, device)
        model.model.shared_prefix_length = truncated_shared_prefix_length
        model.model.different_prefix_index = different_prefix_index

        # encode input
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                use_cache=True
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, config_params):
    if "LLaMA-2-7B-32K" in model_name or "LWM" in model_name or "longchat" in model_name:
        config = LlamaConfig.from_pretrained(path)

        # set attn implementation
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
        dtype = torch.bfloat16

        # clustering config parameters
        config.path_to_clusters_cosine = config_params['path_to_clusters_cosine']
        config.use_centroids = config_params['use_centroids']
        config.hierarchical_lookup = config_params['hierarchical_lookup']
        config.percent_clusters = config_params['percent_clusters']
        config.percent_clusters_l2 = config_params['percent_clusters_l2']
        config.percentile = config_params['percentile']
        config.percentile_lower = config_params['percentile_lower']
        config.obs_window = config_params['obs_window']

        # load model
        model = LlamaForCausalLM.from_pretrained(path, config=config, torch_dtype=dtype)
        model = model.to(device)
        tokenizer = LlamaTokenizer.from_pretrained(path)

    else:
        assert (False) # not implemented yet for other models

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    max_length = model2maxlen[model_name]

    if args.task is not None:
        datasets = [args.task] # only run single task
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "lcc", "repobench-p"]

    # config params
    config_params = {}
    config_params['use_centroids'] = args.use_centroids
    config_params['hierarchical_lookup'] = args.hierarchical_lookup
    config_params['percent_clusters'] = args.percent_clusters
    config_params['percent_clusters_l2'] = args.percent_clusters_l2
    config_params['percentile'] = args.percentile
    config_params['percentile_lower'] = args.percentile_lower
    config_params['obs_window'] = args.obs_window

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print('dataset: ', dataset)

        # update path to clusters here for each dataset
        config_params['path_to_clusters_cosine'] = args.path_to_clusters + f'{dataset}/'
        data = load_dataset('THUDM/LongBench', dataset, split='test')

        # construct savepath here
        if not args.use_centroids:
            savepath = f"pred/{model_name}_baseline"
        else:
            if args.hierarchical_lookup:
                savepath = f"pred/{model_name}_PC1_{args.percent_clusters}_PERC1_{args.percentile}_PC2_{args.percent_clusters_l2}_PERC2_{args.percentile_lower}_lookup"
            else:
                savepath = f"pred/{model_name}_PC{args.percent_clusters}_PERC{args.percentile}"

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        out_path = savepath + f"/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        prompt_only_format = dataset2prompt[dataset + '_prompt']
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        for i in range(len(data_all)):
            data_all[i]['different_prefix_index'] = i

        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
