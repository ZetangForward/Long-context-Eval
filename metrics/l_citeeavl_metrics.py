
import json
from collections import Counter
import re
import string
from rouge import Rouge
from nltk import sent_tokenize
import copy



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
    rouge = Rouge()
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
    def get_score(self, ground_truth, results):
        if '\n' in results:
            ind = results.index('\n')
            results = results[:ind]
        results = remove_citations(results)
        temp_f1_score, temp_precsion_score, temp_recall_score = qa_f1_score(results, ground_truth)
        return [temp_f1_score, temp_precsion_score, temp_recall_score]
    def __call__(self, choices, ground_truths, results):
        score = {"f1_generation":0,"recall_generation":0,"precision_generation":0}
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            for ground_truth in ground_truths:
                all_score = self.get_score(ground_truth,results)
                score["f1_generation"] = max(score["f1_generation"],all_score[0])
                score["precision_generation"] = max(score["precision_generation"],all_score[2])
                score["recall_generation"] = max(score["recall_generation"],all_score[1])
            return score
        all_score = self.get_score(ground_truths,results)
        score["f1_generation"] = max(score["f1_generation"],all_score[0])
        score["precision_generation"] = max(score["precision_generation"],all_score[2])
        score["recall_generation"] = max(score["recall_generation"],all_score[1])
        return score
    
class L_cite_eavl_Rouge_Score:
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
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)

class L_cite_eavl_Counting_Stars:
    def __init__(self,**kwargs):
        pass

    def __call__(self, passage, ground_truth, results):
        score = {"f1_cite":0,"recall_cite":0,"precision_cite":0,"acc_generation":0}
        gold_ind_lst = []
        gold_ans_lst = []
        total_correct=0
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))
        model_ans = results.strip()
        try:
            ind1 = model_ans.index("{")
            ind2 = model_ans.index('}')
            model_ans = json.loads(model_ans[ind1:ind2+1])
        except:
            try:
                model_ans = json.loads('{' + model_ans + '}')
            except:
                return score 
        cite_correct= 0
        if 'passage_id' not in model_ans:
            return score
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
        score["f1_cite"] =f1;score["precision_cite"]=precision;score["recall_cite"]=recall
        model_ans['little_penguin'] = model_ans['little_penguin'][:len(gold_ans_lst)]
        for idx, ans in enumerate(model_ans['little_penguin']):
            if ans in gold_ans_lst:
                total_correct += 1
        score["acc_generation"]=total_correct/len(gold_ans_lst)
        return score
        

class L_cite_eavl_Niah:
    def __init__(self,**kwargs):
        pass
    def get_score(self,ground_truth, results):
        model_ans = results.strip()
        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]

        model_ans = remove_citations(model_ans)
        if model_ans == "":
            return 0
        return rouge_score(model_ans, ground_truth)['rouge-1']['r']
    def __call__(self, choices, ground_truths, results):
        if isinstance(ground_truths,int):
            ground_truths = str(ground_truths)
        elif isinstance(ground_truths,list):
            score = 0
            for ground_truth in ground_truths:
                score = max(score,self.get_score(ground_truth,results))
            return score
        return self.get_score(ground_truths,results)
    



def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")



def format_document(doc):
    if type(doc) == str:
        return "Passage: %s" % (doc)
    if type(doc) == dict:
        return "Title: %s\nPassage: %s" % (doc['title'], doc['text'])


class L_cite_eavl_cite:
    def __init__(self,pipe,**kwargs):
        self.pipe = pipe

    def run_nli_autoais(self,passage, claim):
        result = self.pipe([dict(text=passage, text_pair=claim)])[0]['label']
        inference = 1 if result == "entailment" else 0
        return inference
    def __call__(self, passage, ground_truth, results):
        score = {"f1_cite":0,"recall_cite":0,"precision_cite":0,"cite_num_cite":0}
        sents = sent_tokenize(results)
        if len(sents) == 0:
            return score

        target_sents = [remove_citations(sent).strip() for sent in sents]
        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided
            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            if len(ref) == 0:
                # No citations
                joint_entail = 0

            elif any([ref_id >= len(passage) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0

            else:
                ref = ref[:3]
                total_citations += len(ref)
                joint_passage = '\n'.join([format_document(passage[psgs_id]) for psgs_id in ref]) # title+text

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = self.run_nli_autoais(joint_passage, target_sent)
            entail += joint_entail

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage_ = format_document(passage[psgs_id]) 
                    nli_result = self.run_nli_autoais(passage_, target_sent)
                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage_ = '\n'.join([format_document(passage[pid]) for pid in subset_exclude])
                        nli_result = self.run_nli_autoais(passage_, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 


        citation_recall = entail / len(sents)
        citation_prec = entail_prec / total_citations if total_citations > 0 else 0
        if citation_recall + citation_prec == 0:
            citation_f1 = 0
        else:
            citation_f1 = 2 * citation_recall * citation_prec / (citation_recall + citation_prec)

        score["f1_cite"]=citation_f1
        score['precision_cite']=entail_prec / total_citations if total_citations > 0 else 0
        score["recall_cite"]=entail / len(sents)
        score['cite_num_cite']=total_citations
        return score
    
class L_cite_eavl_niah_cite:
    def __init__(self,pipe,**kwargs):
        self.pipe = pipe
    def run_nli_autoais(self,passage, claim):
        result = self.pipe([dict(text=passage, text_pair=claim)])[0]['label']
        inference = 1 if result == "entailment" else 0
        return inference
    
    def __call__(self, passage, ground_truth, results):
        score = {"f1_cite":0,"recall_cite":0,"precision_cite":0,"cite_num_cite":0}
        for j in range(len(passage)):
            if ground_truth in passage[j]:
                gold_ind = j
                break
        else:
            gold_ind=0
        try:
            model_ans = results.strip()
        except:
            model_ans = results
        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]
        ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", model_ans)][:3]
        if gold_ind in ref:
            score["recall_cite"]=1
            score['precision_cite']=1/len(ref)
            f1 = (2 * 1 * 1/len(ref)) / (1 + 1/len(ref))
            score["f1_cite"]=f1
        score['cite_num_cite']=len(ref)
        return  score

class L_cite_eavl_counting_stars_cite:
    def __init__(self,**kwargs):
        pass
    def __call__(self, passage, ground_truth, results):
        score = {"f1_cite":0,"recall_cite":0,"precision_cite":0,"cite_num_cite":0}
        gold_ind_lst = []
        gold_ans_lst = []
        for j in range(len(passage)):
            if "The little penguin counted" in passage[j]:
                gold_ind_lst.append(j+1)
                pattern = r'The little penguin counted (\d+) ★'
                match = re.search(pattern, passage[j])
                gold_ans_lst.append(int(match.group(1)))

        try:
            model_ans = results.strip()
        except:
            model_ans = results
        try:
            ind1 = model_ans.index("{")
            ind2 = model_ans.index('}')
            model_ans = json.loads(model_ans[ind1:ind2+1])
        except:
            try:
                model_ans = json.loads('{' + model_ans + '}')
            except:
                return score 
        total_correct = cite_correct = 0
        if 'passage_id' not in model_ans:
            return score
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
        score["f1_cite"]=f1
        score['precision_cite']=precision
        score["recall_cite"]=recall
        score['cite_num_cite']=len(model_ans['passage_id'])
        return  score



