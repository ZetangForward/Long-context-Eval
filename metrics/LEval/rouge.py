import nltk
from rouge_score import rouge_scorer
from multiprocessing import Pool
import pdb

class le_rouge:
    def __init__(self,**kwargs):
        pass
    def __call__(self, choices, ground_truths, results):
        prediction,ground_truth= self.postprocess_text(results),self.postprocess_text(ground_truths)
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types,use_stemmer=False)
        scores = scorer.score(ground_truth, prediction)
        return (scores["rouge1"].fmeasure * scores["rouge2"].fmeasure *scores["rougeL"].fmeasure) ** (1.0 / 3.0)
    def postprocess_text(self,text):
    # rougeLSum expects newline after each sentence
        return "\n".join(nltk.sent_tokenize(text))