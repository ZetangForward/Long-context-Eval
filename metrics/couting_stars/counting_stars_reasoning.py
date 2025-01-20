from typing import Any
import json
import re
class Counting_Stars_Reasoning:
    def __init__(self,**kwargs):
        self.r_stars = [16, 116, 43, 70, 59, 106, 8, 48, 112, 67, 25, 101, 82, 93, 76, 62, 6, 18, 108, 4, 34, 53, 85, 90, 126, 22, 45, 121, 39, 96, 75, 30]
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
        score = 0
        for i in range(len(self.a_stars)):
            if self.a_stars[i] in results and self.r_stars[i] in results:
                score += 0.5
            elif self.a_stars[i] in results and self.r_stars[i] not in results:
                score += 1
            elif self.a_stars[i] not in results and self.r_stars[i] in results:
               score += 0.25
            else:
               score += 0
        return score/32

