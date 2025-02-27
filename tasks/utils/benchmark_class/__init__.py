import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from tasks.utils.benchmark_class.LooGLE import LooGLE
from tasks.utils.benchmark_class.RULER import RULER
from tasks.utils.benchmark_class.LongBench import LongBench
from tasks.utils.benchmark_class.Counting_Stars import Counting_Stars
from tasks.utils.benchmark_class.L_CiteEval import L_CiteEval
from tasks.utils.benchmark_class.NIAH import NIAH
from tasks.utils.benchmark_class.LEval import LEval
from tasks.utils.benchmark_class.EN_CiteEval import EN_CiteEval
from tasks.utils.benchmark_class.LongBench_v2 import LongBench_v2
Class_REGISTRY = {
    "LooGLE":LooGLE,
    "RULER":RULER,
    "LongBench":LongBench,
    "Counting_Stars":Counting_Stars,
    "L_CiteEval":L_CiteEval,
    "NIAH":NIAH,
    "LEval":LEval,
    "EN_CiteEval":EN_CiteEval,
    "LongBench_v2":LongBench_v2
}

def get_benchmark_class(benchmark_name):
    return Class_REGISTRY[benchmark_name]
