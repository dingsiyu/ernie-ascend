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
"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import random

from six.moves import xrange


def shuffle_entity(batch_tokens, seg_labels, total_token_num, max_seq_len=None):
    if max_seq_len:
        max_len = max_seq_len
    else:
        max_len = max([len(sent) for sent in batch_tokens])
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        prob_index += pre_sent_len
        beg = 0
        for token_index, token in enumerate(sent):
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                continue
            if beg == 0:
                if seg_label != -1:
                    beg = token_index
                continue

            prob = prob_mask[prob_index + beg]
            if prob > 0.10:
                pass
            else:
                tmp = sent[beg: token_index]
                np.random.shuffle(tmp)
                sent[beg: token_index] = tmp
                    
            if seg_label == -1:
                beg = 0
            else:
                beg = token_index
        pre_sent_len = len(sent)

    return batch_tokens

def augment_data(batch_tokens, seg_labels, batch_sent_ids, batch_pos_ids, batch_task_ids, total_token_num, vocab_size, max_seq_len=None):
    if max_seq_len:
        max_len = max_seq_len
    else:
        max_len = max([len(sent) for sent in batch_tokens])
    #print([len(sent) for sent in batch_tokens])
    #print(sum(len(sent) for sent in batch_tokens), len(batch_tokens))
    prob_mask = np.random.rand(total_token_num)
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    # Note: the first token is [CLS], so [low=1]
    pre_sent_len = 0
    prob_index = 0
    
    for sent_index, sent in enumerate(batch_tokens):
        len_sent = len(sent)
        #print("current_sent_len: ", len_sent)
        prob_aug = np.random.random()
        prob_index += pre_sent_len
        #print("current_prob_index: ", prob_index)
        beg = 0
            
        if 0 < prob_aug <= 1/3.0:
            ## keep the original text
            pass
        elif 1/3.0 < prob_aug <= 2/3.0:
            ## random delete
            sent_new = []
            sent_ids_new, pos_ids_new, task_ids_new = [], [], []
            for token_index, token in enumerate(sent):
                sent_new.append(token)
                pos_ids_new.append(batch_pos_ids[sent_index][token_index])
                sent_ids_new.append(batch_sent_ids[sent_index][token_index])
                task_ids_new.append(batch_task_ids[sent_index][token_index])
                seg_label = seg_labels[sent_index][token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.10:
                    pass
                else:
                    del sent_new[beg: token_index] 
                    del pos_ids_new[beg: token_index]
                    del sent_ids_new[beg: token_index]
                    del task_ids_new[beg: token_index]
                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
            batch_tokens[sent_index] = sent_new
            batch_pos_ids[sent_index] = pos_ids_new
            batch_sent_ids[sent_index] = sent_ids_new
            batch_task_ids[sent_index] = task_ids_new
        else:
            ## random replace
            for token_index, token in enumerate(sent):
                seg_label = seg_labels[sent_index][token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.10:
                    pass
                else:
                    for index in xrange(beg, token_index):
                        prob = prob_mask[prob_index + index]
                        base_prob = 1.0
                        if index == beg:
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            sent[index] = replace_ids[prob_index + index]
                        else:
                            continue
                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
            
        pre_sent_len = len_sent

    return batch_tokens, batch_pos_ids, batch_sent_ids, batch_task_ids


def mask(batch_tokens,
         seg_labels,
         mask_word_tags,
         total_token_num,
         vocab_size,
         max_seq_len=None,
         CLS=1,
         SEP=2,
         MASK=3):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    if max_seq_len:
        max_len = max_seq_len
    else:
        max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    mask_pos_ids = []
    sbo_pos_left = []
    sbo_pos_right = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        mask_word = mask_word_tags[sent_index]
        prob_index += pre_sent_len
        if mask_word:
            beg = 0
            for token_index, token in enumerate(sent):
                seg_label = seg_labels[sent_index][token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.15:
                    pass
                else:
                    for index in xrange(beg, token_index):
                        prob = prob_mask[prob_index + index]
                        base_prob = 1.0
                        if index == beg:
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            mask_label.append(sent[index])
                            sent[index] = MASK
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)
                        elif base_prob * 0.1 < prob <= base_prob * 0.2:
                            mask_label.append(sent[index])
                            sent[index] = replace_ids[prob_index + index]
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)
                        else:
                            mask_label.append(sent[index])
                            mask_pos.append(sent_index * max_len + index)
                            mask_pos_ids.append(index)
                            sbo_pos_left.append(sent_index * max_len + beg)
                            sbo_pos_right.append(sent_index * max_len + token_index - 1)

                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            for token_index, token in enumerate(sent):
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = MASK
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

                elif 0.015 < prob <= 0.03:
                    # random replace
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = replace_ids[prob_index +
                                                        token_index]
                        mask_flag = True
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

                else:
                    # keep the original token
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        mask_pos.append(sent_index * max_len + token_index)
                        mask_pos_ids.append(token_index)
                        sbo_pos_left.append(sent_index * max_len + token_index - 1)
                        sbo_pos_right.append(sent_index * max_len + token_index + 1)

        pre_sent_len = len(sent)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int32").reshape([-1])
    mask_pos_ids = np.array(mask_pos_ids).astype("int64").reshape([-1, 1])
    sbo_pos_left = np.array(sbo_pos_left).astype("int64").reshape([-1, 1])
    sbo_pos_right = np.array(sbo_pos_right).astype("int64").reshape([-1, 1])

    return batch_tokens, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right

def get_random_pos_id(batch_pos_ids):
    max_pos_len = 2048
    batch_size = len(batch_pos_ids)
    random_batch_pos_ids = []
    for pos_ids in batch_pos_ids:
        len_pos = len(pos_ids)
        random_pos_start = random.randint(1, max_pos_len-1)
        random_pos_ids = [0, random_pos_start]
        last_pos_id = random_pos_start
        for _ in range(len_pos - 2):
            random_gap = random.sample(range(1,4), 1)[0]
            pos_id = (last_pos_id + random_gap) % max_pos_len
            if pos_id == 0:
                pos_id += 1
            random_pos_ids.append(pos_id)
            last_pos_id = pos_id
        assert len(random_pos_ids) == len_pos
        random_batch_pos_ids.append(random_pos_ids)
    return random_batch_pos_ids

def prepare_batch_data(insts,
                       total_token_num,
                       task_index,
                       lm_weight,
                       max_seq_len,
                       task_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    batch_pos_ids_org = get_random_pos_id(batch_pos_ids)
    batch_task_ids = [inst[3] for inst in insts]
    labels = [inst[4] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1, 1])
    seg_labels = [inst[5] for inst in insts]
    mask_word_tags = [inst[6] for inst in insts]

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"

    not_mask = False
    if lm_weight < 0.01:
        not_mask = True

    if not_mask:
        out = copy.deepcopy(batch_src_ids)
        out_aug = copy.deepcopy(batch_src_ids)
        sent_ids_aug = copy.deepcopy(batch_sent_ids)
        pos_ids_aug = copy.deepcopy(batch_pos_ids)
        task_ids_aug = copy.deepcopy(batch_task_ids)
        out = shuffle_entity(batch_src_ids, seg_labels, total_token_num)
        _, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right = mask(
            batch_src_ids,
            seg_labels,
            mask_word_tags,
            total_token_num,
            vocab_size=voc_size,
            max_seq_len=max_seq_len,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id)
        #print(total_token_num) 
        out_aug, pos_ids_aug, sent_ids_aug, task_ids_aug = augment_data(out_aug, seg_labels, sent_ids_aug, pos_ids_aug, task_ids_aug, total_token_num, voc_size, max_seq_len=max_seq_len)
        gather_idx = np.array(list(range(len(out))), "int32").reshape([-1])
        batch_pos_ids_aug = get_random_pos_id(pos_ids_aug)
    
        out = out + out_aug
        batch_pos_ids = batch_pos_ids_org + batch_pos_ids_aug
        batch_sent_ids = batch_sent_ids + sent_ids_aug
        batch_task_ids = batch_task_ids + task_ids_aug
    else:
        out, mask_label, mask_pos, mask_pos_ids, sbo_pos_left, sbo_pos_right = mask(
            batch_src_ids,
            seg_labels,
            mask_word_tags,
            total_token_num,
            vocab_size=voc_size,
            max_seq_len=max_seq_len,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id)
        gather_idx = np.array(list(range(len(out))), "int32").reshape([-1])
        batch_pos_ids = batch_pos_ids_org
    

    # Second step: padding
    src_id, self_input_mask = pad_batch_data(
        out, pad_idx=pad_id, return_input_mask=True, max_seq_len=max_seq_len)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id, max_seq_len=max_seq_len)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id, max_seq_len=max_seq_len)
    task_id = pad_batch_data(batch_task_ids, pad_idx=pad_id, max_seq_len=max_seq_len)
    lm_w = np.array([lm_weight]).astype("float32")

    ## add constract 
    batch_mask = np.ones([len(src_id), len(src_id)], dtype="float32") - np.eye(len(src_id), dtype="float32")
    loss_mask = np.zeros([len(src_id), len(src_id)], dtype="float32") + np.eye(len(src_id), k=len(src_id)//2, dtype="float32") + \
                                                                  np.eye(len(src_id), k=-len(src_id)//2, dtype="float32")


    return_list = [
        src_id, pos_id, sent_id, task_id, self_input_mask, mask_label, mask_pos, lm_w, batch_mask, loss_mask, gather_idx
    ]

    for i in xrange(task_num):
        if i == task_index:
            return_list.append(labels)
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.zeros_like(labels))
            return_list.append(np.array([0.0]).astype("float32"))
    
    #return_list = [np.squeeze(src_id), lm_weight]
    #return_list = [src_id, lm_weight]

    return return_list


def pad_batch_data(insts,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   max_seq_len=None):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if max_seq_len:
        max_len = max_seq_len
    else:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":

    pass
