import re
import json
class GeneralTorch:
    def __init__(self):
        pass
    def __call__(self, result):
        raw_outputs = []
        process_outputs = []
        if isinstance(result, list):  # 返回是列表，加了batch。已做截断
            raw_outputs = result
        elif isinstance(result, str):  # 返回是字符串，没有加batch。已做截断
            raw_outputs = [result]
        process_outputs = raw_outputs[::]
        return raw_outputs, process_outputs

class CountingStarsEN:
    def __init__(self):
        pass
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(raw_outputs)):
            try:
                processed_outputs[i] = eval(processed_outputs[i].strip())["little_penguin"]
            except:
                processed_outputs[i] = []  
        return raw_outputs, processed_outputs
    
class CountingStarsZH:
    def __init__(self):
        pass
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            if "```" in processed_outputs[i]:
                processed_outputs[i] = processed_outputs[i].replace('```','').replace("json",'')

            try:
                processed_outputs[i] = eval(processed_outputs[i].strip())['小企鹅']
            except:
                processed_outputs[i] = []  
        return raw_outputs, processed_outputs
class CountingStarsCitaion:
    def __init__(self):
        pass

    def porcess(self,model_ans):
        model_ans = model_ans.strip()
        try:
            ind1 = model_ans.index("{")
            ind2 = model_ans.index('}')
            model_ans = json.loads(model_ans[ind1:ind2+1])
            return model_ans
        except:
            return []
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs
    
class NIAH:
    def __init__(self):
        pass
    def porcess(self,data):
        data = data.strip()
        if '\n' in data:
            ind = data.index('\n')
            data = data[:ind]
        ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", data)]
        return ref
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs
    
class CitationGeneral:
    def __init__(self):
        pass
    def porcess(self,data):
        data = data.strip()
        if '\n' in data:
            ind = data.index('\n')
            data = data[:ind]
        ref = re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", data)).replace(" |", "").replace("]", "")
        return ref
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs
    
class Citation_Sum:
    def __init__(self):
        pass
    def porcess(self,model_ans):
        if 'summary' in model_ans[:100].lower():
            try:
                ind = model_ans.index(':')
            except:
                pass
        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]
            model_ans = re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", model_ans)).replace(" |", "").replace("]", "")
        return model_ans
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs
class CitationNiah:
    def __init__(self):
        pass
    def porcess(self,model_ans):
        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]
            model_ans = re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", model_ans)).replace(" |", "").replace("]", "")
        return model_ans
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs
    
class RULER:
    def __init__(self):
        pass
    def porcess(self,predict_str):

        predict_str = predict_str.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        predict_str = np_pattern.sub('\n', predict_str).strip()
        return predict_str
    
    def __call__(self, raw_outputs, processed_outputs):
        for i in range(len(processed_outputs)):
            processed_outputs[i] = self.porcess(processed_outputs[i])
        return raw_outputs, processed_outputs

postprocess_dict = {
    "general_torch": GeneralTorch,
    "counting_stars_en":CountingStarsEN,
    "counting_stars_zh":CountingStarsZH,
    "citation_general":CitationGeneral,
    "citation_sum":Citation_Sum,
    "counting_stars_citaion":CountingStarsCitaion,
    "citation_niah":CitationNiah,
    "ruler":RULER,
    
}


def get_postprocess(postprocess_name):
    return postprocess_dict[postprocess_name]