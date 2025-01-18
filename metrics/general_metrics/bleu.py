import pdb
from typing import Any
from nltk.translate.bleu_score import sentence_bleu

class BLEU1:
    def __init__(self, **kwargs):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        ground_truth, results = (
            ground_truth.replace("\n", " ").split(),
            results.replace("\n", " ").split(),
        )
        bleu1 = sentence_bleu([ground_truth], results, weights=(1, 0, 0, 0))
        return bleu1
class BLEU4:
    def __init__(self, **kwargs):
        pass
    def __call__(self, doc, ground_truth, results) -> Any:
        ground_truth, results = (
            ground_truth.replace("\n", " ").split(),
            results.replace("\n", " ").split(),
        )
        bleu4 = sentence_bleu([ground_truth], results, weights=(0, 0, 0, 1))
        return bleu4
    