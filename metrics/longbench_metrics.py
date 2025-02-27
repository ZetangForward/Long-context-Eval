import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib

from typing import List
from collections import Counter
from rouge import Rouge

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


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


class Count_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self,  ground_truth, results):
        numbers = re.findall(r"\d+", results)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
    

class Retrieval_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self,  ground_truth, results):
        pattern = r'Paragraph (\d+)'
        matches = re.findall(pattern, ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", results)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth_id):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
class Retrieval_ZH_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self, ground_truth, results):
        pattern = r'段落(\d+)'
        matches = re.findall(pattern, ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", results)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth_id):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)
    def __call__(self, choices, ground_truths, results):

        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)

class Code_Sim_Score:
    def __init__(self,**kwargs):
        pass

    def get_score(self, ground_truth, results):

        all_lines = results.lstrip('\n').split('\n')
        results = ""
        for line in all_lines:
            if ('`' not in line) and ('#' not in line) and ('//' not in line):
                results = line
                break
        return (fuzz.ratio(results, ground_truth) / 100)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)

class Classification_Score:
    def __init__(self,**kwargs):
        pass

    def get_score(self, choices, ground_truth, results):
        em_match_list = []
        all_classes = choices
        for class_name in all_classes:
            if class_name in results:
                em_match_list.append(class_name)
        for match_term in em_match_list:
            if match_term in ground_truth and match_term != ground_truth:
                em_match_list.remove(match_term)
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
        return score
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(choices,ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
    
class Rouge_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        rouge = Rouge()
        try:
            scores = rouge.get_scores([results], [ground_truth], avg=True)
        except:
            return 0.0
        return scores["rouge-l"]["f"]

    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)

class Rouge_Zh_Score:
    def __init__(self,**kwargs):
        pass

    def get_score(self, ground_truth, results):

        results = " ".join(list(jieba.cut(results, cut_all=False)))
        ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
        rough_score_calculator = Rouge_Score()
        return rough_score_calculator("", ground_truth,results)

    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
class F1_Score:
    def __init__(self,**kwargs):
        pass

    def __call__(self, doc, ground_truth, results):

        common = Counter(results) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(results)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

class Precision:
    def __init__(self,**kwargs):
        pass

    def __call__(self, doc, ground_truth, results):

        common = Counter(results) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(results)
        return precision
class Recall:
    def __init__(self,**kwargs):
        pass

    def __call__(self, doc, ground_truth, results):

        common = Counter(results) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(results)
        recall = 1.0 * num_same / len(ground_truth)
        return recall 
    
class Qa_F1_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):

        normalized_prediction = normalize_answer(results)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        f1_calculator = F1_Score()
        
        return f1_calculator("", ground_truth_tokens,prediction_tokens)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
    
class Qa_F1_Zh_Score:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        prediction_tokens = list(jieba.cut(results, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
        ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
        f1_calculator = F1_Score()
        return f1_calculator("", ground_truth_tokens,prediction_tokens)
    
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
    
class Qa_Recall:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        normalized_prediction = normalize_answer(results)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        recall_calculator = Recall()
        return recall_calculator("", ground_truth_tokens,prediction_tokens)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
class Qa_Precision:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        normalized_prediction = normalize_answer(results)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        recall_calculator = Precision()
        return recall_calculator("", ground_truth_tokens,prediction_tokens)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
class Qa_F1_ZH_Score:
    def __init__(self,**kwargs):
        pass

    def get_score(self, ground_truth, results):

        prediction_tokens = list(jieba.cut(results, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
        ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
        f1_calculator = F1_Score()
        return f1_calculator("", ground_truth_tokens,prediction_tokens)
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)


