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
import numpy as np

def extract_single(token_ids, seg_labels, cls_id, sep_id, must=None):
    sep_index = token_ids.index(sep_id)

    if must == "first":
        return (token_ids[1:sep_index + 1], seg_labels[1:sep_index + 1])
    elif must == "second":
        return (token_ids[sep_index + 1:], seg_labels[sep_index + 1:])

    if np.random.random() < 0.5:
        return (token_ids[sep_index + 1:], seg_labels[sep_index + 1:])
    else:
        return (token_ids[1:sep_index + 1], seg_labels[1:sep_index + 1])


def merge_pair(sent1, sent2, cls_id, sep_id):
    token_ids = [cls_id] + sent1[0] + sent2[0]
    seg_labels = [-1] + sent1[1] + sent2[1]
    sent_ids = [0] * (len(sent1[0]) + 1) + [1] * len(sent2[0])
    pos_ids = range(len(token_ids))
    return token_ids, sent_ids, pos_ids, seg_labels


def sent_rel_16cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(sent_rel_16cls, 'last_data'):
        sent_rel_16cls.last_data = None

    prob = np.random.random()
    if label != 0 and prob < 0.4:
        sep_index = token_ids.index(sep_id)
        token_ids = [cls_id] + token_ids[sep_index + 1:] + token_ids[1:sep_index + 1]
        sent_ids = [0] * (len(token_ids) - sep_index) + [1] * sep_index
        seg_labels = [-1] + seg_labels[sep_index + 1:] + seg_labels[1:sep_index + 1]
        if label % 2 == 0:
            label -= 1
        else:
            label += 1

    elif sent_rel_16cls.last_data and prob > 0.8:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = \
                sent_rel_16cls.last_data
        token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                extract_single(last_token_ids, last_seg_labels, cls_id, sep_id),
                extract_single(token_ids, seg_labels, cls_id, sep_id), cls_id, sep_id)
        label = 15

    sent_rel_16cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def qa_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(qa_3cls, 'last_data'):
        qa_3cls.last_data = None

    prob = np.random.random()
    if qa_3cls.last_data and prob < 0.5:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = \
                qa_3cls.last_data
        token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                extract_single(last_token_ids, last_seg_labels, cls_id, sep_id),
                extract_single(token_ids, seg_labels, cls_id, sep_id), cls_id, sep_id)
        label = 0

    qa_3cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def qt_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if not hasattr(qt_3cls, 'last_data'):
        qt_3cls.last_data = None

    prob = np.random.random()
    if qt_3cls.last_data and prob < 0.5:
        last_token_ids, last_sent_ids, last_pos_ids, last_label, last_seg_labels = \
                qt_3cls.last_data
        if np.random.random() < 0.5:
            token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                    extract_single(last_token_ids, last_seg_labels, cls_id, sep_id, "first"),
                    extract_single(token_ids, seg_labels, cls_id, sep_id, "second"), cls_id, sep_id)
        else:
            token_ids, sent_ids, pos_ids, seg_labels = merge_pair(
                     extract_single(token_ids, seg_labels, cls_id, sep_id, "first"),
                     extract_single(last_token_ids, last_seg_labels, cls_id, sep_id, "second"), cls_id, sep_id) 
        label = 0

    qt_3cls.last_data = (token_ids, sent_ids, pos_ids, label, seg_labels)
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def next_sent_3cls(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    if label == 1 and np.random.random() < 0.5:
        sep_index = token_ids.index(sep_id)
        token_ids = [cls_id] + token_ids[sep_index + 1:] + token_ids[1:sep_index + 1]
        sent_ids = [0] * (len(token_ids) - sep_index) + [1] * sep_index
        label = 2
        seg_labels = [-1] + seg_labels[sep_index + 1:] + seg_labels[1:sep_index + 1] 

    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def sent_distance(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    return (token_ids, sent_ids, pos_ids, label, seg_labels)


def multi_sent_sorted(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
    start = 0
    token_ids_list = []
    seg_labels_list = []
    while True:
        try:
            sep_index = token_ids[start + 1:].index(sep_id)
            end = start + 1 + sep_index
            token_ids_list.append(token_ids[start + 1: end])
            seg_labels_list.append(seg_labels[start + 1: end])
            start = end
        except Exception as e:
            break
    premutation_2_sent = [[0, 1], [1, 0]]
    premutation_3_sent = [[0, 1, 2], [0, 2, 1],
                          [1, 0, 2], [1, 2, 0],
                          [2, 0, 1], [2, 1, 0]]
    premutation_4_sent = [[0, 1, 2, 3], [0, 1, 3, 2],
                          [0, 2, 1, 3], [0, 2, 3, 1],
                          [0, 3, 1, 2], [0, 3, 2, 1],
                          [1, 0, 2, 3], [1, 0, 3, 2],
                          [1, 2, 0, 3], [1, 2, 3, 0],
                          [1, 3, 0, 2], [1, 3, 2, 0],
                          [2, 0, 1, 3], [2, 0, 3, 1],
                          [2, 1, 0, 3], [2, 1, 3, 0],
                          [2, 3, 0, 1], [2, 3, 1, 0],
                          [3, 0, 1, 2], [3, 0, 2, 1],
                          [3, 1, 0, 2], [3, 1, 2, 0],
                          [3, 2, 0, 1], [3, 2, 1, 0]]

    shuffle_token_ids = [cls_id]
    shuffle_sent_ids = [0]
    shuffle_seg_labels = [-1]
    if label == 0 and len(token_ids_list) == 2:
        choice_index = np.random.choice(2)
        for index, order in enumerate(premutation_2_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 2 and len(token_ids_list) == 3:
        choice_index = np.random.choice(6)
        for index, order in enumerate(premutation_3_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    elif label == 8 and len(token_ids_list) == 4:
        choice_index = np.random.choice(24)
        for index, order in enumerate(premutation_4_sent[choice_index]):
            shuffle_token_ids += token_ids_list[order] + [sep_id]
            shuffle_sent_ids += [index] * len(token_ids_list[order]) + [index]
            shuffle_seg_labels += seg_labels_list[order] + [-1]
        shuffle_label = label + choice_index
    else:
        print("format error")

    return (shuffle_token_ids, shuffle_sent_ids, pos_ids, shuffle_label, shuffle_seg_labels)


if __name__ == "__main__":
    cls_id = 1
    sep_id = 2
    token_ids = [1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 2, 12, 13, 2, 14, 2]
    sent_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2]
    pos_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
    label = 8
    seg_labels = [-1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1]
    for tmp in multi_sent_sorted(token_ids, sent_ids, pos_ids, label, seg_labels, cls_id, sep_id):
        print(tmp)

    
            
    
