REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 1,
    "generate": 1,
}


class Request:
    def __init__(self, prompt_input, params, raw_example):
        self.prompt_input = prompt_input
        self.params = params
        self.raw_example = raw_example
