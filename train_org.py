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
"""ERNIE pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import paddle.fluid as fluid

from reader.pretraining import ErnieDataReader
from model.ernie import ErnieModel, ErnieConfig
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params

from pretrain_args import parser
import json

args = parser.parse_args()

# yapf: enable.


bsz = int(args.batch_size / args.max_seq_len) # -1
def create_model(pyreader_name, ernie_config, task_group):
    """create_model"""
    ## get input
    shapes = [[bsz, args.max_seq_len, 1], [bsz, args.max_seq_len, 1], [bsz, args.max_seq_len, 1], 
             [bsz, args.max_seq_len, 1], [bsz, args.max_seq_len, 1], [bsz, 1],
             [bsz, 1], [1], [bsz, 1], [bsz, 1], [bsz, 1]]
    names = ["src_ids", "pos_ids", "sent_ids", "task_ids", "input_mask", "mask_label", 
             "mask_pos", "lm_weight", "batch_mask", "loss_mask", "gather_idx"]
    dtypes = ["int64", "int64", "int64", "int64", "float32", "int64", "int64", 
             "float32", "float32", "float32", "int64"]
    cnt_general_input = len(shapes)

    for index, task in enumerate(task_group):
        shapes.extend([[bsz, 1], [1]])
        names.extend(['task_label_' + str(index), 'task_weight_' + str(index)])
        dtypes.extend(["int64", "float32"])

    assert len(shapes) == len(names) == len(dtypes), "The three fields must have same size"
    inputs = []
    for i in range(len(shapes)):
        inputs.append(fluid.layers.data(name=names[i], shape=shapes[i], 
                                        dtype=dtypes[i], append_batch_size=False))
    
    general_data, task_params = inputs[:cnt_general_input], inputs[cnt_general_input:]
    src_ids, pos_ids, sent_ids, task_ids, input_mask, \
                  mask_label, mask_pos, lm_weight, batch_mask, loss_mask, gather_idx = general_data
    

    ## build graph
    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_amp)

    mask_lm_loss = ernie.get_lm_output(mask_label, mask_pos)
    checkpoints = ernie.get_checkpoints()
    total_loss = mask_lm_loss * lm_weight
    graph_vars = [mask_lm_loss, lm_weight]

    index = 0
    total_constract_loss = 0
    for task in task_group:
        task_labels = task_params[index]
        task_weight = task_params[index + 1]
        task_loss, task_acc = ernie.get_task_output(task, task_labels, gather_idx)
        total_loss += task_loss * task_weight * task["loss_weight"]
        if task["constart"]:
            contract_loss = ernie.get_contrastive_loss(batch_mask, loss_mask)
            total_loss += contract_loss * task_weight
            total_constract_loss += contract_loss * task_weight
        graph_vars.extend([task_acc, task_weight])
        index += 2
   
    ## build output
    graph_vars.append(total_constract_loss)
    graph_vars.append(total_loss)
    #for var in graph_vars:
    #    var.persistable = True

    fetch_vars = {"graph_vars": graph_vars,
                  "checkpoints": checkpoints}

    return fetch_vars, names

def predict_wrapper(args,
                    exe,
                    ernie_config,
                    task_group,
                    test_prog=None,
                    pyreader=None,
                    data_names=None,
                    fetch_list=None):
    # Context to do validation.
    data_reader = ErnieDataReader(
        task_group,
        True,
        vocab_path=args.vocab_path,
        batch_size=2048,#args.batch_size,
        voc_size=ernie_config['vocab_size'],
        shuffle_files=False,
        epoch=1,
        max_seq_len=args.max_seq_len,
        hack_old_trainset=args.hack_old_data,
        is_test=True)

    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)

    def predict(exe=exe, data_names=data_names):

        #pyreader.set_batch_generator(data_reader.data_generator())
        #pyreader.start()
        predict_data_generator = data_reader.data_generator()
    
        cost = 0
        constract_loss = 0
        lm_cost = 0
        lm_steps = 0
        task_acc = {}
        task_steps = {}
        steps = 0
        feed_list = {}
        time_begin = time.time()
        while True:
            try:
                input_list = next(predict_data_generator, None)
                for index in range(len(input_list)):
                    feed_list[train_data_names[index]] = input_list[index] 
                outputs = exe.run(feed=feed_list, fetch_list=fetch_list, program=test_prog)
                each_mask_lm_cost, lm_w = outputs[:2]
                each_total_constract_loss = outputs[-2]
                each_total_cost =  outputs[-1]
                lm_cost += np.sum(each_mask_lm_cost * lm_w)
                lm_steps += np.sum(lm_w)
                cost += np.mean(each_total_cost)
                constract_loss += np.mean(each_total_constract_loss)
                steps += 1

                index = 2
                for task in task_group:
                    each_task_acc = outputs[index]
                    task_w = outputs[index + 1]
                    task_acc[task["task_name"]] = task_acc.get(task["task_name"], 0.0) \
                                                + np.sum(each_task_acc * task_w)
                    task_steps[task["task_name"]] = task_steps.get(task["task_name"], 0.0) \
                                                  + np.sum(task_w)
                    index += 2

            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        ret = ["loss: %f" % (cost / steps),
               "constract_loss: %f" % (constract_loss / steps),
               "ppl: %f" % (np.exp(lm_cost / lm_steps))]
        for task in task_group:
            acc = task_acc[task["task_name"]] / task_steps[task["task_name"]]
            ret.append("%s acc: %f" % (task["task_name"], acc))

        ret.append("speed: " + str(args.skip_steps / used_time) + " steps/s")
        return ret

    return predict


def train(args):
    print("pretraining start")
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    with open(args.task_group_json) as f:
        task_group = json.load(f)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 4 if args.use_amp else 2
    exec_strategy.num_iteration_per_drop_scope = min(1, args.skip_steps)

    node_nums = 1 #int(os.getenv("PADDLE_NODES_NUM"))
    print("args.is_distributed:", args.is_distributed)
    num_trainers = 1
    trainer_id = 0
    
    dist_strategy=None

    gpu_id=0
    gpus = 1#fluid.core.get_cuda_device_count()
    print(gpus)
    if args.is_distributed:
        gpus = os.getenv("FLAGS_selected_gpus").split(",")
        gpu_id = int(gpus[0])

    place = fluid.CPUPlace()
    dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("Device count %d, gpu_id:%d" % (dev_count, gpu_id))

    train_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_program, startup_prog):
        with fluid.unique_name.guard():
            fetch_vars, train_data_names = create_model(
                pyreader_name='train_reader', ernie_config=ernie_config, task_group=task_group)
            graph_vars = fetch_vars["graph_vars"]
            checkpoints = fetch_vars["checkpoints"]
            total_loss = graph_vars[-1]
            if args.use_recompute:
                dist_strategy.recompute_checkpoints = checkpoints
            fetch_list_ascend = [var for var in graph_vars]   
            scheduled_lr, loss_scaling = optimization(
                loss=total_loss,
                warmup_steps=args.warmup_steps,
                num_train_steps=args.num_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_program,
                startup_prog=startup_prog,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                use_fp16=args.use_amp,
                use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                init_loss_scaling=args.init_loss_scaling,
                incr_every_n_steps=args.incr_every_n_steps,
                decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                incr_ratio=args.incr_ratio,
                decr_ratio=args.decr_ratio,
                fetch_list=fetch_list_ascend,
                dist_strategy=dist_strategy)    

    origin_train_program = train_program

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            fetch_vars, test_data_names = create_model(
                pyreader_name='test_reader', ernie_config=ernie_config, task_group=task_group)
            graph_vars = fetch_vars["graph_vars"]
            total_loss = graph_vars[-1]

    test_prog = test_prog.clone(for_test=True)
    
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    
    if args.init_checkpoint and args.init_checkpoint != "":
        #init_checkpoint(exe, args.init_checkpoint, origin_train_program, args.use_amp)
        init_pretraining_params(exe, args.init_checkpoint, origin_train_program, args.use_amp)

    data_reader = ErnieDataReader(
        task_group,
        False,
        batch_size=args.batch_size,
        vocab_path=args.vocab_path,
        voc_size=ernie_config['vocab_size'],
        epoch=args.epoch,
        max_seq_len=args.max_seq_len,
        generate_neg_sample=args.generate_neg_sample,
        hack_old_trainset=args.hack_old_data)
    
    #only fleet
    train_exe = exe

    predict = predict_wrapper(
        args,
        exe,
        ernie_config,
        task_group,
        test_prog=test_prog,
        data_names=test_data_names,
        fetch_list=[var for var in graph_vars])

    #train_pyreader.set_batch_generator(data_reader.data_generator())
    #train_pyreader.start()
    train_data_generator = data_reader.data_generator()
    steps = 0
    time_begin = time.time()
    feed_list = {}
    while True:#steps < args.num_train_steps:
        try:
            steps += 1#node_nums
            skip_steps = args.skip_steps# * node_nums
            
            input_list = next(train_data_generator(), None)
            for index in range(len(input_list)):
                feed_list[train_data_names[index]] = input_list[index]
                
            fetch_list = []
            if trainer_id == 0 and steps % skip_steps == 0:
                fetch_list = [var for var in graph_vars] + [scheduled_lr.name]
                if args.use_amp:
                    fetch_list.append(loss_scaling.name)

            outputs = train_exe.run(feed=feed_list, fetch_list=fetch_list, program=train_program)
            time_end = time.time()
            used_time = time_end - time_begin
            
            if outputs:
                each_mask_lm_cost, lm_w = outputs[:2]
                if args.use_amp:
                    each_total_constract_loss, each_total_cost, np_lr, l_scaling = outputs[-4:]
                else:
                    each_total_constract_loss, each_total_cost, np_lr = outputs[-3:]
                acc_list =[]
                index = 2
                for task in task_group:
                    each_task_acc = outputs[index]
                    task_w = outputs[index + 1]
                    acc = np.sum(each_task_acc * task_w) / np.sum(task_w)
                    acc_list.append("%s acc: %f" % (task["task_name"], acc))
                    index += 2

                epoch, current_file_index, total_file, current_file, mask_type = data_reader.get_progress()
                if args.use_amp:
                    print("current learning_rate:%f, loss scaling:%f" % (np_lr[0], l_scaling[0]))
                else:
                    print("current learning_rate:%f" % np_lr[0])
                print(
                    "epoch: %d, progress: %d/%d, step: %d, constract_loss: %f, loss: %f, "
                    "ppl: %f, %s, speed: %f steps/s, file: %s, mask_type: %s"
                    % (epoch, current_file_index, total_file, steps,
                       np.mean(each_total_constract_loss), np.mean(each_total_cost),
                       np.exp(np.sum(each_mask_lm_cost * lm_w) / np.sum(lm_w)),
                       ", ".join(acc_list), skip_steps / used_time,
                       current_file, mask_type))
                time_begin = time.time()
            elif steps % skip_steps == 0:
                epoch, current_file_index, total_file, current_file, mask_type = data_reader.get_progress(
                )
                print("epoch: %d, progress: %d/%d, step: %d, "
                        "speed: %f steps/s, file: %s, mask_type: %s"
                        % (epoch, current_file_index, total_file, steps,
                            skip_steps / used_time, current_file, mask_type))
                time_begin = time.time()

            if not trainer_id == 0:
                continue

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, origin_train_program)

            if steps % args.validation_steps == 0:
                valid_list = predict()
                print("[validation_set] epoch: %d, step: %d, %s" % \
                      (epoch, steps, ", ".join(valid_list)))

        except fluid.core.EOFException:
            train_pyreader.reset()
            break


if __name__ == '__main__':
    print_arguments(args)
    train(args)
