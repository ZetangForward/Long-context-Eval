from transformers import AutoConfig

class Base:
    def __init__(self):
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
        if hasattr(model.tokenizer, 'apply_chat_template') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template:
            tokenized_prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True
            )
        else:
            tokenized_prompt = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        config = AutoConfig.from_pretrained(model_path)
        max_length = config.max_position_embeddings - 500
        if len(tokenized_prompt) > max_length:
            half = max_length // 2
            prompt = (
                model.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                model.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )
        return prompt