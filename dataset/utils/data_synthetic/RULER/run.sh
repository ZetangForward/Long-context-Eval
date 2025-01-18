#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi



ROOT_DIR="RULER" # the path that stores generated task samples and model predictions.
MODEL_DIR="/opt/data/private/sora/models" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization


# Model and Tokenizer
source config_models.sh
MODEL_NAME=$1
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} )
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"



# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK



# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}"
    mkdir -p ${DATA_DIR}

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    done
done

echo "Total time spent on call_api: $total_time seconds"