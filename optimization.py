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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import paddle
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_optimizers.ascend import ascend_optimizer
from utils.fp16 import create_master_params_grads, master_param_to_train_param, apply_dynamic_loss_scaling

def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 use_dynamic_loss_scaling=False,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8,
                 ascend=False,
                 fetch_list=None,
                 dist_strategy=None):
    
    scheduled_lr = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("learning_rate"),
        shape=[1],
        value=learning_rate,
        dtype='float32',
        persistable=True)
    #optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    optimizer = fluid.optimizer.SGD(learning_rate=scheduled_lr)
    optimizer._learning_rate_map[fluid.default_main_program()] = scheduled_lr
    if ascend:
        optimizer = ascend_optimizer.AscendOptimizer(optimizer, fetch_list=fetch_list, auto_dp=True,
                                                     rank_table_file=os.environ["RANK_TABLE_FILE"])

    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    def exclude_from_weight_decay(name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict() 
    for param in train_program.global_block().all_parameters():
        param_list[param.name] = param * 1.0
        param_list[param.name].stop_gradient = True

    loss = fluid.layers.mean(loss)
    if ascend:
        _, param_grads = optimizer.minimize(loss, startup_prog, 
                                            auto_dp=True,
                                            rank_table_file=os.getenv("RANK_TABLE_FILE", None))
    else:
        _, param_grads = optimizer.minimize(loss, startup_prog)

    # 打印网络
    with open('mnist.start.program', 'w') as fout:
        #print(str(fluid.default_startup_program()), file=fout)
        print(str(startup_prog), file=fout)
    with open('mnist.main.program', 'w') as fout:
        #print(str(fluid.default_main_program()), file=fout)
        print(str(train_program), file=fout)

    if weight_decay > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param.name):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)

    return optimizer, scheduled_lr, 0
