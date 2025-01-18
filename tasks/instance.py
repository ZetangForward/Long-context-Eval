import json
from collections import defaultdict

class Instance:

    def __init__(self, data):
        self.data = data
        self.ground_truth = ""
        self.prompt_inputs = []
        self.raw_outputs = []
        self.processed_outputs = []
        self.eval_results = defaultdict(list)
        self.metrics = defaultdict(None)

    def dump(self):
        instance_data = {
            "data": self.data,
            "ground_truth": self.ground_truth,
            "prompt_inputs": self.prompt_inputs,
            "raw_outputs": self.raw_outputs,
            "processed_outputs": self.processed_outputs,
            "eval_results": self.eval_results,
            "metrics": self.metrics,
        }
        
        return json.dumps(instance_data, ensure_ascii=False)