from transformers import AutoConfig
from lte.utils.main_args import handle_cli_args

class Base:
    def __init__(self,**kwargs):
        self.benchmark_name = None
        self.task_names = None
        self.ability = None
        self.hf_repo = None
        self.data_path = None
        self.download_all = False
        self.download_data =False
        self.llm_params = {}
        self.metric = {}
    
    def download_and_transform_data(self,**kwargs):
        """Placeholder for the data-making logic."""
        raise NotImplementedError("Subclasses must implement this method.")
    def make_data(self, dataset, ability, task_name,**kwargs):
        """Placeholder for the data-making logic."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def transform(self, data, task_name,**kwargs):

        return {
            "input": data["question"],
            "output": data["answer"],
            "processed_output": data["answer"],
        }
    
    def modify(self, prompt, model, model_path,**kwargs):
        """Adjust input prompt to fit within the model's token limit."""
        args = handle_cli_args()
        if args.template:
            prompt = self.args.template.format(user_input=prompt, assistant_response='')
        elif hasattr(model.tokenizer, 'apply_chat_template') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template:
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
        # else:
        #     tokenized_prompt = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if args.max_lenth:
        #     tokenized_prompt = tokenized_prompt[:args.max_lenth]
        return prompt