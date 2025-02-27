Our framework now incorporates information retrieval capabilities, offering four different methods for users to choose from: BM25, contriever, llamaindex, and openai. 
# Rag
If you want to use the RAG (Retrieval Augmented Generation) function, you just need to add the parameter "--rag BM25/contriever/llamaindex/openai" during the evaluation. Here is an example of its usage.
## Example Code
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 0,1 --save_tag "tag" --rag openai
```
or
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 0,1 --save_tag "tag" --rag openai
```




