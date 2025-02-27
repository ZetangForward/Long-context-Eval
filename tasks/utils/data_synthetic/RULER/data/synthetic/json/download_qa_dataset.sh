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

if [ -f ./tasks/General/RULER/tmp_Rawdata/squad.json ]; then
    echo "squad.json already exists"
else
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ./tasks/General/RULER/tmp_Rawdata/squad.json
fi

if [ -f ./tasks/General/RULER/tmp_Rawdata/hotpotqa.json ]; then
    echo "hotpotqa.json already exists"
else
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O ./tasks/General/RULER/tmp_Rawdata/hotpotqa.json
fi