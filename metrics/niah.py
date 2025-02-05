
class niah:
    def __init__(self,**kwargs):
        pass
    def __call__(self, passage, ground_truth, results):
        ground_truth = ground_truth.lower().split()
        results = results.lower().split()
        return len(set(ground_truth).intersection(set(results))) / len(results)