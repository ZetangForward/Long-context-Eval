这是一个向框架添加任务的简单教程。
## 第一步
在 ./tasks/utils/benchmark_class 中创建你的 benchmark.py 并从 base_class 中的 BASE 继承。以下是部分代码要求的示例。
### 类
```python
llm_param = {
}
metric1 = {"metric_name1":None} #None 位置可以是一个字典，用于存储评估所需的参数
Class benchmark(Base):
    def __init__(self,limit):
        super().__init__()
        benchmark_name = "benchmark_name" 
        task_names = ["task_name1"]
        ability = "Reasoning"
        data_path = f"tasks/{ability}/{benchmark_name}/data"
        limit = int(limit) if limit!="auto" else 10000
        hf = ""  #用于下载数据
        llm_params = {"task_name1":llm_paramm} #llm_params 属性是一个字典，其中键是 task_names 中指定的任务名称，值是对应每个任务的模型推理参数
        metric = {"task_name1":metric1} #该参数是一个字符串。请将其添加到 lab/lte/metrics/__init__.py 的 METRICS_REGISTRY 函数中，以导入评估指标。
``` 
### 评估指标
通过METRICS_REGISTRY获取的评估函数如下：

``` python
class metric_name1:
    def __init__(self,**kwargs):
        pass
    def __call__(self, choices, ground_truths, results):
        return score
``` 
- choices: 多项选择题中提供的选项。
- ground_truths: 问题的正确答案，无论是多项选择题还是开放式问题。
- results: 模型针对问题生成的输出。
### 功能
1. 数据下载与转换（简化代码）
将数据下载到路径“./tasks/{}/{}/tmp_Rawdata”.format(ability, benchmark_name)后，转换成框架所需的数据格式，保存到“./tasks/{}/{}/data/{}.json”.format(ability, self.benchmark_name, task_name)中。

``` python
def download_and_transform_data(self,**kwargs):
     load_dataset()
    self.make_data()
    # 将原始数据下载到指定路径后，将其转换为所需的格式。
def make_data(self,**kwargs):
    for data in "input_path": #遍历下载的数据并将其转换为所需的格式。
        transform_data(data)
        save in output_path
def transform_data(self,raw_data):
    return {
        "passage": raw_data["passage"],
        "question": raw_data["question"],
        "choices": "",
        "answer": raw_data["reference_counting_results"],
    }
``` 

2. 数据处理
``` python
# 处理输入到模型的数据，修改格式并添加提示词。
def transform(self,data,task_name,**kwargs):
    prompt_list = {
    "task_name": prompt:。\n\n{context}\n\n****\n\n{input}\n\n******\n\n******：",
}
    return {
        "input": prompt_list[task_name].format(),
        "output": data["answer"],
        "processed_output": data["answer"],
    }
##对于不同的基准测试，输入到模型之前的数据处理方式各不相同。以下是设置修改函数以处理数据的示例：
def modify(self, prompt, model, model_path,**kwargs):
    """Adjust input prompt to fit within the model's token limit."""
    if hasattr(model.tokenizer, 'apply_chat_template'):
        prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
    return prompt
```

## 第二步
完成代码后，在 ./tasks/{ability}/{benchmark_name}/benchmark_name.yaml 设置基准配置文件。
## 第三步 
运行代码

``` bash
lte.run --model_path your model_path \
    --eval \
    --benchmark_configs your config path:./tasks/{ability}/{benchmark_name}/
``` 