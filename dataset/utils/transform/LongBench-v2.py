import random
import json
# template_rag = open('./dataset/utils/demo_prompt/LongBench-v2/0shot_rag.txt', encoding='utf-8').read()
# template_no_context = open('./dataset/utils/demo_prompt/LongBench-v2/0shot_no_context.txt', encoding='utf-8').read()
# template_0shot = open('./dataset/utils/demo_prompt/LongBench-v2/0shot.txt', encoding='utf-8').read()
# template_0shot_cot = open('./dataset/utils/demo_prompt/LongBench-v2/0shot_cot.txt', encoding='utf-8').read()
# template_0shot_cot_ans = open('./dataset/utils/demo_prompt/LongBench-v2/0shot_cot_ans.txt', encoding='utf-8').read()
# def transform(data,task_name,**kwargs):
#     context = data['passage']
#     if args.lb_v2_rag > 0:
#         template = template_rag
#         retrieved = data["retrieved_context"][:args.lb_v2_rag]
#         retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
#         context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['passage']}" for idx, x in enumerate(retrieved)])
#     elif args.lb_v2_no_context:
#         template = template_no_context
#     elif args.lb_v2_cot:
#         template = template_0shot_cot
#     else:
#         template = template_0shot

   
    return {

        "input": prompt_list[task_name].format(context=data["passage"],input=data["question"],choices = data["choices"]),
        "output": data["answer"],
        "processed_output": data["answer"],
    }
