from typing import Any
from nltk.translate.meteor_score import single_meteor_score

class METEOR:
    def __init__(self, **kwargs):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:

        ground_truth, results = (
            ground_truth.replace("\n", " ").split(),
            results.replace("\n", " ").split(),
        )
        
        meteor = single_meteor_score(set(ground_truth), set(results))
        return float(meteor)
