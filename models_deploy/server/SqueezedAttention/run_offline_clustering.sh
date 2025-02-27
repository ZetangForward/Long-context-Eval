#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL="Llama-2-7B-32K-Instruct"
DATASETS=(2wikimqa)
PERC_CLUSTERS="5" # Percent Clusters

# loop over datasets
for DATASET in "${DATASETS[@]}"; do
  PATH_TO_SAVE_CLUSTERS="./Model_Deploy_URLs/SqueezedAttention/fixed-prompt-clusters/${DATASET}/"
  python ./Model_Deploy_URLs/SqueezedAttention/offline_clustering.py $MODEL \
                                 --dataset $DATASET \
                                 --output_path $PATH_TO_SAVE_CLUSTERS \
                                 --percent_clusters $PERC_CLUSTERS \
                                 --observation_window 100 \
                                 --device 0
done
