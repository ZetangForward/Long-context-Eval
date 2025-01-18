import sys,os
sys.path.append(os.path.join(os.path.abspath(__file__)))
from dataset.utils.benchmark_class.LooGLE import LooGLE
from dataset.utils.benchmark_class.RULER import RULER
from dataset.utils.benchmark_class.LongBench import LongBench
from dataset.utils.benchmark_class.Counting_Stars import Counting_Stars
Class_REGISTRY = {
    "LooGLE":LooGLE,
    "RULER":RULER,
    "LongBench":LongBench,
    "Counting_Stars":Counting_Stars
}

def get_benchmark_class(benchmark_name):
    return Class_REGISTRY[benchmark_name]
