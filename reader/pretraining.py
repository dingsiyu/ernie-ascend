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

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import types
import gzip
import logging
import re
import six
import collections
import tokenization

import paddle
import paddle.fluid as fluid
import reader.data_utils as data_utils

from batching import prepare_batch_data


class ErnieDataReader(object):
    def __init__(self,
                 task_group,
                 is_valid,
                 vocab_path,
                 batch_size=4096,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 is_test=False,
                 generate_neg_sample=False,
                 hack_old_trainset=False):

        self.vocab = self.load_vocab(vocab_path)
        self.task_group = task_group
        self.is_valid = is_valid
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.input_slots = 5
        self.is_test = is_test
        self.generate_neg_sample = generate_neg_sample
        
        assert self.batch_size > 100, "Current batch size means total token's number, \
                                       it should not be set to too small number."

        if hack_old_trainset:
            self.input_slots = 4
        
        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file, self.mask_type

    def parse_line(self, line, max_seq_len=512, task_index=None):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().split(";")
        assert len(line) == self.input_slots, \
                "One sample must have %d fields!" % self.input_slots

        if self.input_slots == 4:
            (token_ids, sent_ids, pos_ids, label) = line
            token_ids = [int(token) for token in token_ids.split(" ")]
            sent_ids = [int(token) for token in sent_ids.split(" ")]
            pos_ids = [int(token) for token in pos_ids.split(" ")]
            #fake seg_labels
            seg_labels = [0, ] * len(token_ids)
            seg_labels[0] = -1
            start = 0
            while True:
                try:
                    sep_index = token_ids[start + 1:].index(self.sep_id)
                    end = start + 1 + sep_index
                    seg_labels[end] = -1
                    start = end
                except Exception as e:
                    break

        if self.input_slots == 5:
            (token_ids, sent_ids, pos_ids, seg_labels, label) = line
            token_ids = [int(token) for token in token_ids.split(" ")]
            sent_ids = [int(token) for token in sent_ids.split(" ")]
            pos_ids = [int(token) for token in pos_ids.split(" ")]
            seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]

        label = int(label)
        
        task = self.task_group[task_index]
        data_func = task.get("data_func", None)
        if data_func and data_func != "":
            token_ids, sent_ids, pos_ids, label, seg_labels = \
                eval("data_utils." + data_func)(token_ids, sent_ids,
                    pos_ids, label, seg_labels, self.cls_id, self.sep_id)

        if isinstance(task_index, int):
            task_ids = [task_index] * len(token_ids)
        else:
            task_ids = [0] * len(token_ids)

        assert len(token_ids) == len(sent_ids) == len(pos_ids) == len(
            seg_labels) == len(task_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(task_ids)"

        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, task_ids, label, seg_labels]

    def read_file(self, file, task_index):
        # assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        with gzip.open(file, "rb") as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            for line in lines:
                line = line.decode("utf8")
                parsed_line = self.parse_line(
                    line, max_seq_len=self.max_seq_len, task_index=task_index)
                if parsed_line is None:
                    continue
                yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples

            Args:
                pos_samples: list of positive samples
            
            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            token_seq, type_seq, pos_seq, task_seq, label, seg_labels = sample
            sep_index = token_seq.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_seq[1:sep_index], seg_labels[1:sep_index])
            else:
                return [
                    token_seq[sep_index + 1:-1], seg_labels[sep_index + 1:-1]
                ]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id)
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens), self.sep_id)

            token_seq = [self.cls_id] + left_tokens + [self.sep_id] + \
                    right_tokens + [self.sep_id]
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) + 1)
            pos_seq = range(len(token_seq))
            task_seq = [task_seq[0]] * len(token_seq)
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [-1]

            assert len(token_seq) == len(type_seq) == len(pos_seq) == len(seg_label_seq) == len(task_seq), \
                    "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append([token_seq, type_seq, pos_seq, task_seq, 0, seg_label_seq])

        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples
            
            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def data_generator(self):
        """
        data_generator
        """
        filelist_key = "train_filelist"
        if self.is_valid:
            filelist_key = "valid_filelist"

        all_files = []
        task_probs = []
        sum = 0.0
        for task in self.task_group:
            all_files.append(open(task[filelist_key]).readlines())
            task_probs.append(task["prob"])
            sum += task["prob"]
        for i in range(len(task_probs)):
            task_probs[i] = task_probs[i] / sum
        task_probs = np.array(task_probs).ravel()

        def wrapper():
            def reader(task_index):
                files = all_files[task_index]
                for epoch in range(self.epoch):
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        file, mask_word_prob = file.strip().split("\t")
                        mask_word = (np.random.random() < float(mask_word_prob))

                        if mask_word:
                            self.mask_type = "mask_word"
                        else:
                            self.mask_type = "mask_char"

                        sample_generator = self.read_file(file, task_index)
                        if not self.is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(
                                sample_generator)

                        for sample in sample_generator:
                            self.current_epoch = epoch + 1
                            self.current_file_index = index + 1
                            self.current_file = file
                            self.total_file = len(files)
                            self.current_epoch = epoch + 1

                            if sample is None:
                                continue
                            sample.append(mask_word)
                            yield sample

            def batch_reader(reader, batch_size):
                batch, total_token_num, max_len = [], 0, 0
                dev_count = 1
                buff = []
                readers = []
                for i in range(len(task_probs)):
                    buff.append(None)
                    readers.append(reader(i))
                task_indices = range(len(task_probs))

                end_times = 0
                while end_times < 50:
                    task_index = np.random.choice(task_indices, p=task_probs)
                    #print("before batch_size: ", batch_size)    
                    if self.task_group[task_index]["constart"]:
                        batch_size_fact = batch_size // 2
                    else:
                        batch_size_fact = batch_size
                    #print("after batch_size: ", batch_size_fact)
                    
                    dev_num = 0
                    cur_reader = readers[task_index]

                    while dev_num < dev_count:
                        if buff[task_index] is not None:
                            cur_len = len(buff[task_index][0])
                            max_len = max(max_len, cur_len)
                            batch.append(buff[task_index])
                            total_token_num += cur_len
                            buff[task_index] = None

                        parsed_line = next(cur_reader, None)
                        if parsed_line is None:
                            end_times += 1
                            dev_num += 1
                            if len(batch) > 0:
                                yield batch, total_token_num, task_index, self.task_group[task_index]["lm_weight"]
                                batch, total_token_num, max_len = [], 0, 0
                            continue

                        end_times = 0
                        cur_len = len(parsed_line[0])
                        max_len = self.max_seq_len #max(max_len, cur_len)
                        if (len(batch) + 1) * max_len > batch_size_fact:
                             yield batch, total_token_num, task_index, self.task_group[task_index]["lm_weight"]
                             batch, total_token_num, max_len = [], 0, 0
                             dev_num += 1
                             buff[task_index] = parsed_line
                        else:
                            batch.append(parsed_line)
                            total_token_num += cur_len

            for batch_data, total_token_num, task_index, lm_weight in batch_reader(reader,
                                                            self.batch_size):
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    task_index,
                    lm_weight,
                    self.max_seq_len,
                    len(self.task_group),
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
