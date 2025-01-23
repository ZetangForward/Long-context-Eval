import argparse

def handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,help = "Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used.")
    parser.add_argument("--rag", default="", choices=["",'openai', 'BM25', 'raptor', 'llamaindex', 'contriever'],help="Selects the model type or provider for information retrieval. You must choose from the following options: 'openai', 'BM25', 'raptor', 'llamaindex', 'contriever'. Each option corresponds to a different approach for retrieving relevant information.")
    parser.add_argument("--server", type=str, default="transformers",help="Selects which reasoning frame , Must be a string `[transformers,vllm]")
    parser.add_argument("--acceleration", type=str, default="",help="Selects which acceleration frame , Must be a string:`[duo_attn,Snapkv,streaming_llm,ms-poe,selfextend,lm-inifinite(llama,gpt-j,mpt)，activation_beacon,tova],现在别用")
    parser.add_argument("--port", type=int, default="5002",help=" Accepts an integer and sets a port for deploying the model.")
    parser.add_argument("--device", type=str, default="0",help=" It is used to set the device on which the model will be placed. The input must be a string, such as `0,1,2,3`.")
    parser.add_argument("--torch_dtype", type=str, default="torch.bfloat16",help="Specifies the PyTorch data type that you want to use. ")
    parser.add_argument("--benchmark_names", type=str,required=True,help="you can choose from the following options: 1. Benchmark, which may involve tasks such as L-cite-eval; 2. Task, like L-CiteEval-Data_narrativeqa; 3. Ability, for example General Capability. Multiple tasks can be specified, with each subject separated by a comma.")
    parser.add_argument("--batch_size", type=int, default=1,help="Sets the batch size used for evaluation.")
    parser.add_argument("--limit", type=str, default="auto",help="ccepts an integer,or just `auto` .it will limit the number of documents to evaluate to the first X documents (if an integer) per task of all tasks with `auto`")
    parser.add_argument("--save_path",  type=str, default="save/output",help="The final results will be saved in the save_path.")
    parser.add_argument("--eval", action="store_true", default=False,help="If the --eval option is not provided, the process will terminate after the model generation and before the evaluation.")
    parser.add_argument("--write_out", action="store_true", default=False,help="If the --write_out option is provided, we will store the model genration details in output_path")
    parser.add_argument("--model_deploy_datails_path", type=str, default="./model_out_details",help="Sets a path for saving the details regarding model deployment.")
    parser.add_argument("--selfextend_window_size",type=int,default=1024,help="the params of selfextend")
    parser.add_argument("--selfextend_group_size",type=int,default=32,help="the params of selfextend")
    parser.add_argument("--tova_multi_state_size",type=int,default=4096,help="the params of tova")
    parser.add_argument("--device_split_num",type=int,default=2,help="each task needs the number of gpu")
    parser.add_argument("--generation_path",type=str,default="save/genration",help="e")
    parser.add_argument("--adapter_path",type=str,default="",help="The path to the adapter file")
    # parser.add_argument("--lb_v2_cot", "-cot2", action='store_true') # set to True if using COT,when you test longbench-v2
    # parser.add_argument("--lb_v2_no_context", "-nc2", action='store_true') # set to True if using no context (directly measuring memorization),when you test longbench-v2
    # parser.add_argument("--lb_v2_rag", "-rag2", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context,when you test longbench-v2
    args = parser.parse_args()
    return args