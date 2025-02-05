
from rouge_score import rouge_scorer
class niah:
    def __init__(self,**kwargs):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    def __call__(self, passage, ground_truth, results):
        return self.scorer.score(ground_truth, results)['rouge1'].fmeasure*10