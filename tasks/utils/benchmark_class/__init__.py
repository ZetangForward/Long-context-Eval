import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from tasks.utils.benchmark_class.LooGLE import LooGLE
from tasks.utils.benchmark_class.RULER import RULER
from tasks.utils.benchmark_class.LongBench import LongBench
from tasks.utils.benchmark_class.Counting_Stars import Counting_Stars
from tasks.utils.benchmark_class.L_CiteEval import L_CiteEval
from tasks.utils.benchmark_class.NIAH import NIAH
Class_REGISTRY = {
    "LooGLE":LooGLE,
    "RULER":RULER,
    "LongBench":LongBench,
    "Counting_Stars":Counting_Stars,
    "L_CiteEval":L_CiteEval,
    "NIAH":NIAH
}

def get_benchmark_class(benchmark_name):
    return Class_REGISTRY[benchmark_name]
