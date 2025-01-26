REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 1,
    "generate": 1,
}


class Request:
    def __init__(self, instances, params, raw_example):
        self.instances = instances
        self.params = params
        self.raw_example = raw_example
