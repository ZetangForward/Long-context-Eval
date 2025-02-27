import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def exact_match_score(prediction, ground_truth):
    flag = False  # whether with options
    Choice = ['A', 'B', 'C', 'D']
    for char in normalize_answer(ground_truth):
        if char not in Choice:
            flag = True
            break
    res = 0
    if not flag:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            res = 1
        elif set(normalize_answer(prediction)).issubset(set(normalize_answer(ground_truth))):
            res = 0.25  # has many correct options
    else:
        try:
            pre = float(prediction)
            gt = float(ground_truth)
            res = int(pre == gt)
        except ValueError:
            if ground_truth.lower().replace(" ", "") in prediction.lower().replace(" ", ""):
                res = 1

    print(prediction, ground_truth, f"| score={res}")
    print("=" * 20)
    return res
class le_em:
    def __init__(self,**kwargs):
        pass
    def get_score(self,choices, ground_truths, results):
        prediction,ground_truth= results,ground_truths
        flag = False  # whether with options
        Choice = ['A', 'B', 'C', 'D']
        for char in normalize_answer(ground_truth):
            if char not in Choice:
                flag = True
                break
        res = 0
        if not flag:
            if normalize_answer(prediction) == normalize_answer(ground_truth):
                res = 1
            elif set(normalize_answer(prediction)).issubset(set(normalize_answer(ground_truth))):
                res = 0.25  # has many correct options
        else:
            try:
                pre = float(prediction)
                gt = float(ground_truth)
                res = int(pre == gt)
            except ValueError:
                if ground_truth.lower().replace(" ", "") in prediction.lower().replace(" ", ""):
                    res = 1
        return res
    def __call__(self, choices, ground_truths, results):
        prediction,ground_truth= results,ground_truths
        if isinstance(prediction,list):
            re1 = self.get_score(choices, ground_truths[0], results[0])
            re2 = self.get_score(choices, ground_truths[1], results[1])
            return (re1+re2)/2
        return self.get_score(choices, ground_truths, results)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_exact_match(predictions, references):
    exact_match = 0
    correct = 0
    half_correct = 0
    for prediction, ground_truths in zip(predictions, references):
        res = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        exact_match += res
        if res == 1:
            correct += 1
        if res == 0.25:
            half_correct += 1
    print(
        f"There are {correct} correct answers \n [for coursera:] {half_correct} can not select all correct options\n Total: {len(predictions)} questions.")
    return 100.0 * exact_match / len(predictions)