import os
import argparse
import paddle.fluid as fluid
import numpy as np

#emb_dim = 768
#src_name = "mask_lm_out_fc.b_0"
#src_num = 18000
#tgt_name = "mask_lm_out_fc.b_0_2"
#tgt_num = 19000
#cut_num = 12051

from_dir = "./params"
to_dir = "./params_tgt"


def get_params(filepath):
    for root, dirs, files in os.walk(filepath):
        for f in files:
            filename = os.path.join(root, f)
            print(filename)
            src_name = f
            tgt_name = f 
            src_num = 3072
            tgt_num = 2048
            convert()


def convert():
    program = fluid.Program()
    global_block = program.global_block()
    global_block.create_parameter(name=src_name,
            shape=[1,1],
            dtype='float32',
            initializer=fluid.initializer.TruncatedNormal())
    global_block.create_parameter(name=tgt_name,
            shape=[1,1],
            dtype='float32',
            initializer=fluid.initializer.TruncatedNormal())

    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)
    
    print type(fluid.global_scope().find_var(src_name))
    fluid.io.load_params(exe, 
            from_dir,
            main_program=program)

    np_src = np.array(fluid.global_scope().find_var(src_name).get_tensor())
    np_tgt = np.array(fluid.global_scope().find_var(tgt_name).get_tensor())
    np_tgt[:cut_num] = np_src[:cut_num]
    np_tgt[18017] = np_src[17963]

    fluid.global_scope().find_var(tgt_name).get_tensor().set(np_tgt, place)
    fluid.io.save_params(exe,
             to_dir,
             main_program=program)

if __name__ == "__main__":
    convert()
   
