
from typing import Any



class String_Match_Part:
    def __init__(self, **kwargs):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        for i in ground_truth:
            i = i.lower()
            if i in results.lower():
                return 1
        return 0

class String_Match_All:
    def __init__(self, **kwargs):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        sun = 0
        for i in ground_truth:
            i = i.lower()
            if i in results.lower():
                sun+=1
        return sun/len(ground_truth)


