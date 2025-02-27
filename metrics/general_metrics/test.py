import json
import argparse
import glob
from transformers import AutoTokenizer
from collections import Counter
from rouge import Rouge
from tqdm import tqdm
import numpy as np
import re
import string



def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def rouge_score(prediction, ground_truth, **kwargs):
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class L_cite_eavl_Qa_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self, c, results):
        if '\n' in results:
            ind = results.index('\n')
            results = results[:ind]
        results = remove_citations(results)
        temp_f1_score, temp_precsion_score, temp_recall_score = qa_f1_score(results, results)
        return [temp_f1_score, temp_precsion_score, temp_recall_score]
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = {"f1":0,"recall":0,"precision":0}
            for ground_truth in ground_truths:
                score["f1"] = max(score["f1"],self.get_score(ground_truth,results)[0])
                score["precision"] = max(score["precision"],self.get_score(ground_truth,results)[0])
                score["recall"] = max(score["f1"],self.get_score(ground_truth,results)[0])
            return score
        return self.get_score(ground_truths,results)


class L_cite_eavl_Counting_Stars:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        model_ans = results.strip()
        if 'summary' in model_ans[:100].lower():
            try:
                ind = model_ans.index(':')
            except:
                return 0
            model_ans = model_ans[ind+1:].strip()

        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]

        model_ans = remove_citations(model_ans)
        if model_ans == "":
            return 0
        return rouge_score(model_ans, ground_truth)['rouge-l']['f']
    def __call__(self, passage, ground_truth, results) -> Any:
            gold_ind_lst = []
            gold_ans_lst = []
            for j in range(len(passage)):
                if "The little penguin counted" in passage[j]:
                    gold_ind_lst.append(j+1)
                    pattern = r'The little penguin counted (\d+) ★'
                    match = re.search(pattern, passage[j])
                    gold_ans_lst.append(int(match.group(1)))
            model_ans = results.strip()
            try:
                model_ans = json.loads('{' + model_ans + '}')
            except:
                return 0
            if 'passage_id' not in model_ans:
                return 0
            model_ans['passage_id'] = list(set(model_ans['passage_id']))

            for idx, psg_id in enumerate(model_ans['passage_id']):

                if psg_id in gold_ind_lst:
                    cite_correct += 1
            
            precision = cite_correct / len(model_ans['passage_id'])
            recall = cite_correct / len(gold_ind_lst)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
    