#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import csv
import json
import numpy as np
from collections import namedtuple

import tokenization
from batching import pad_batch_data
from reader.pretraining import ErnieDataReader
from model.ernie import ErnieModel, ErnieConfig

if __name__ == '__main__':
    with open("./config/task.json") as f:
        task_group = json.load(f)
    ernie_config = ErnieConfig("./config/ernie_config.json")

    data_reader = ErnieDataReader(
        task_group,
        True,
        batch_size=128,
        vocab_path="./config/vocab.txt",
        voc_size=ernie_config['vocab_size'],
        epoch=1,
        max_seq_len=64,
        generate_neg_sample=False,
        hack_old_trainset=False,
        is_test=True)
    train_data_generator = data_reader.data_generator()
    cnt = 0
    while True:
    #for batch_data in train_data_generator():
        batch_data = next(train_data_generator(), None)
        src_id, pos_id, sent_id, task_id, self_input_mask, mask_label, mask_pos, lm_w, batch_mask, loss_mask, gather_idx = batch_data[:11]
        
        print(mask_pos)
        #for data in batch_data:
        #    print(data.shape)
        #print(batch_data[1])
        #print(mask_pos.tolist())
        #print(src_id.tolist())
        #print(self_input_mask.tolist())
        #print(task_labels.tolist())
        #print(src_id.shape[0] * src_id.shape[1], len(mask_pos.tolist()), len(mask_pos.tolist()) * 1.0 / (src_id.shape[0] * src_id.shape[1]))
        #for src in src_id.tolist():
        #    print(src)
        print("########3")
        #for pos in pos_id.tolist():
        #    print(pos)
        #print(len(self_input_mask), self_input_mask[0].shape)
        print("=======")
        
