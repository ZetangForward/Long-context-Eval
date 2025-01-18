
from rouge import Rouge

class ROUGE:
    def __init__(self, **kwargs):
        pass

    def __call__(self, doc, ground_truth, results):
        score = {"rouge-1":0,"rouge-2":0,"rouge-l":0}
        rouge = Rouge()
        rouge_ = rouge.get_scores(hyps=[results], refs=[ground_truth])[0]
        score["rouge-1"]=rouge_["rouge-1"]["r"]
        score["rouge-2"]=rouge_["rouge-2"]["r"]
        score["rouge-l"]=rouge_["rouge-l"]["r"]
        return score


