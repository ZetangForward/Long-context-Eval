from bert_score import score
class Bert_Score:
    def __init__(self, **kwargs):
        pass
    def __call__(self, doc, ground_truth, results):
        bertscore = score([ground_truth], [results], lang="EN")
        return float(bertscore[1])
