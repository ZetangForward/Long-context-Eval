
class LongBench_v2:
    def __init__(self,**kwargs):
        pass
    def __call__(self, choices, ground_truths, results):
        acc = int(ground_truths == results)
        return acc

    