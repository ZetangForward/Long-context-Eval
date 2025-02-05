

# lte.run --model_path /mnt/petrelfs/tangzecheng/local_ckpt/pg19/Llama-3-8B-Scaling-CE/lora/global_step100_hf --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "1_CE_32k" > 1_CE_32k.log 2>&1 &

# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "2_LONGPPL" > 2_LONGPPL.log 2>&1 &


# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-ce/global_step150_hf --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "2_CE" > 2_CE.log 2>&1 &


# lte.run --model_path /mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/long-context-training-V1/Llama-3.1-8B-Instruct/full --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "3_CD" > 3_CD.log 2>&1 &


# lte.run --model_path /mnt/petrelfs/tangzecheng/local_ckpt/Llama-3.1-8B-Instruct-SEALONG --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "3_Self-improve" > 3_Self_improve.log 2>&1 &


# lte.run --model_path /mnt/petrelfs/tangzecheng/local_ckpt/Llama-3-8B-ProLong-64k-Instruct --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "prolong64k" > prolong64k.log 2>&1 &

# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300 --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "2_CD" > 2_CD.log 2>&1 &


# RULER==========================




# lte.run --model_path /mnt/petrelfs/tangzecheng/local_ckpt/Llama-3-8B-ProLong-64k-Base --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "prolong64kbase" > prolong64kbase.log 2>&1 &





# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3-8B-pg19-longce/200ep  --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "1_LongPPL-32K" > 1_LongPPL_32K.log 2>&1 &



# lte.run --model_path /mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/pg19/Llama-3-8B-Scaling-Noise/full --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "1_CD" > 1_CD.log 2>&1 &


# lte.run --model_path meta-llama/Meta-Llama-3-8B --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "1_base" > 1_base.log 2>&1 &


#>>>>


# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3.1-8B-Instruct-longalpaca-adapter-global_step250/ --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "3_LongLoRA" > 3_LongLoRA.log 2>&1 &




# lte.run --model_path /mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3-8B-Scaling-CE-lora-global_step450_hf/ --eval --benchmark_configs /mnt/petrelfs/tangzecheng/lte/tasks/General/RULER/RULER.yaml --save_tag "1_CE_32k" > 1_CE_32k.log 2>&1 &
