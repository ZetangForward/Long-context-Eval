from typing import Any
import json
import re
class Counting_Stars_Searching:
    def __init__(self,**kwargs):
        self.a_stars = [15, 117, 42, 69, 58, 107, 9, 49, 113, 66, 26, 102, 81, 94, 77, 61, 5, 19, 109, 3, 35, 54, 86, 89, 127, 21, 46, 122, 38, 97, 74, 29]

    def __call__(self, passage, ground_truth, results) -> Any:
        try:
            model_ans = results.strip()
            re.sub(r"\\+", "",model_ans)
            ind1 = model_ans.index("{")
            ind2 = model_ans.index('}')
            model_ans = json.loads(model_ans[ind1:ind2+1])
        except:
            return 0
        if "little_penguin" in model_ans:
            results = model_ans["little_penguin"]
        elif "小企鹅" in model_ans:
            results = model_ans["小企鹅"]
        else:
            return 0
        results = results[:32]
        results = list(set(results))
        score = 0
        for i in self.a_stars:
            if i in results:
                score += 1
        return score/32
