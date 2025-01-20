from typing import Any
import re
from ..longbench_metrics import F1_Score,Recall,Precision

class Counting_Stars_Citation_F1:
    def __init__(self,**kwargs):
        pass

    def __call__(self, passage, ground_truth, results) -> Any:
        gold_ind_lst = []
        gold_ans_lst = []
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))
        if 'passage_id' not in results:
            return 0
        else:
            results['passage_id'] = list(set(results['passage_id']))
            f1_calculator = F1_Score()
            return f1_calculator("",results['passage_id'] ,gold_ind_lst)
class Counting_Stars_Citation_Recall:
    def __init__(self,**kwargs):
        pass
    def __call__(self, passage, ground_truth, results) -> Any:
        gold_ind_lst = []
        gold_ans_lst = []
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))
        if 'passage_id' not in results:
            return 0
        else:
            results['passage_id'] = list(set(results['passage_id']))
            recall_calculator = Recall()
        
            return recall_calculator("",results['passage_id'] ,gold_ind_lst)
class Counting_Stars_Citation_Precision:
    def __init__(self,**kwargs):
        pass
    def __call__(self, passage, ground_truth, results) -> Any:
        gold_ind_lst = []
        gold_ans_lst = []
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))

        if 'passage_id' not in results:
            return 0
        else:
            results['passage_id'] = list(set(results['passage_id']))
            precision = Precision()
            return precision("",results['passage_id'] ,gold_ind_lst)
        
class Counting_Stars_Citation_ACC:
    def __init__(self,**kwargs):
        pass
    def __call__(self, passage, ground_truth, results) -> Any:
        gold_ind_lst = []
        gold_ans_lst = []
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))
        if 'little_penguin' not in results:
            return 0
        else:
            total_correct  = 0
            results['little_penguin'] = results['little_penguin'][:len(gold_ans_lst)]
            for idx, ans in enumerate(results['little_penguin']):
                if ans in gold_ans_lst:
                    total_correct += 1
            return total_correct/len(gold_ans_lst)