
import json
import requests


#发生到服务器的数据类型
data_prompt = {
    "params": {},
    "instances": [],
}

def _post_request(url, data):
    data_prompt["instances"] = data
    s = json.dumps(data_prompt)
    headers = {"Content-Type": "application/json"}

    return requests.post(url, data=s, headers=headers).json()

class GeneralModel:
    def __init__(self,url):
        self.url = url

    def make_request_instance(self, input, output):
        return input + output



    def generate(self, request):
        data_prompt["params"].update(request.params)

        data = [
            self.make_request_instance(req["input"], "")
            for req in request.instances
        ]
        result = _post_request(self.url, data)


        return result

    