Our framework now incorporates information retrieval capabilities, offering four different methods for users to choose from: BM25, contriever, llamaindex, and openai. 
# Rag
If you want to use the RAG (Retrieval Augmented Generation) function, you just need to add the parameter "--rag BM25/contriever/llamaindex/openai" during the evaluation. Here is an example of its usage.
## Example Code
```python
lte.run --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 0,1 --save_tag "tag" --rag contriever
```
or
```bash
python lte/main.py --model_path caskcsg/Llama-3-8B-NExtLong-512K-Instruct --eval --benchmark_configs tasks/Faithfulness/L_CiteEval/L_CiteEval_Data.yaml --device 0,1 --save_tag "tag" --rag contriever
```
## congfigs
In RAG, chunk_size refers to the size of each chunk of the retrieved information, and chunk_overlap refers to the number of elements (such as tokens, words or characters) that overlap between adjacent chunks when splitting the text or retrieved documents. You can modify these values by adjusting the parameters in the 'models_deploy/rag/rag_configs.yaml' file.



