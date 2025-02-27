# zecheng_note: metrics 分类 + 归并

from .general_metrics.bleu import BLEU1,BLEU4
from .general_metrics.rouge import ROUGE
from .general_metrics.bert_score import Bert_Score
from .couting_stars.counting_stars_searching import Counting_Stars_Searching
from .couting_stars.counting_stars_reasoning import Counting_Stars_Reasoning
from .general_metrics.meteor_score import METEOR
from .gpt import GPT_Loogle_Su,GPT_Loogle_Qa
from .longbench_metrics import Count_Score,Retrieval_Score,Retrieval_ZH_Score,Code_Sim_Score,Classification_Score,Rouge_Score,F1_Score,Qa_F1_Score,Qa_F1_ZH_Score,Precision,Recall,Qa_Recall,Qa_Precision,Qa_F1_Zh_Score,Rouge_Zh_Score
from .couting_stars.counting_stars_citation import  Counting_Stars_Citation_ACC,Counting_Stars_Citation_Precision,Counting_Stars_Citation_Recall,Counting_Stars_Citation_F1
from .general_metrics.string_math import  String_Match_Part, String_Match_All
from .l_citeeavl_metrics import L_cite_eavl_Counting_Stars,L_cite_eavl_Niah,L_cite_eavl_Qa_Score,L_cite_eavl_Rouge_Score,L_cite_eavl_cite,L_cite_eavl_niah_cite,L_cite_eavl_counting_stars_cite
from .niah import niah
from .LEval.f1 import le_f1
from .LEval.rouge import le_rouge
from .LEval.em import le_em
from .LongBench_v2 import LongBench_v2
METRICS_REGISTRY = {
    "bert_score":Bert_Score,
    "bleu1": BLEU1,
    "bleu4": BLEU4,
    "rouge": ROUGE,
    "searching_acc": Counting_Stars_Searching,
    "reasoning_acc": Counting_Stars_Reasoning,
    "meteor_score": METEOR,
    "gpt_loogle_su":GPT_Loogle_Su,
    "gpt_loogle_qa":GPT_Loogle_Qa,
    "qa_f1_score":Qa_F1_Score,
    "rouge_score":Rouge_Score,
    "classification_score":Classification_Score,
    "count_score":Count_Score,
    "retrieval_score":Retrieval_Score,
    "code_sim_score":Code_Sim_Score,
    "f1":F1_Score,
    "precision":Precision,
    "recall":Recall,
    "qa_precision":Qa_Precision,
    "qa_recall":Qa_Recall,
    "counting_stars_citation_recall":Counting_Stars_Citation_Recall,
    "counting_stars_citation_acc":Counting_Stars_Citation_ACC,
    "counting_stars_citation_f1":Counting_Stars_Citation_F1,
    "counting_stars_citation_precision":Counting_Stars_Citation_Precision,
    "string_match_all":String_Match_All,
    "string_match_part":String_Match_Part,
    "qa_f1_zh_score":Qa_F1_Zh_Score,
    "rouge_zh_score":Rouge_Zh_Score,
    "retrieval_zh_score":Retrieval_ZH_Score,
    "l_cite_eavl_counting_stars":L_cite_eavl_Counting_Stars,
    "l_cite_eavl_niah":L_cite_eavl_Niah,
    "l_cite_eavl_qa_score":L_cite_eavl_Qa_Score,
    "l_cite_eavl_rouge_score":L_cite_eavl_Rouge_Score,
    
    "l_cite_eavl_cite":L_cite_eavl_cite,
    "l_cite_eavl_niah_cite":L_cite_eavl_niah_cite,
    "l_cite_eavl_counting_stars_cite":L_cite_eavl_counting_stars_cite,
    "niah":niah,
    "le_f1":le_f1,
    "le_em":le_em,
    "le_rouge":le_rouge,
    "longbench_v2":LongBench_v2,
}


def get_metric(metric_name):
    return METRICS_REGISTRY[metric_name]
