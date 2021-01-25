# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle
from optimization import optimization
import paddle.fluid.layers as layers
from ascend import ascend_optimizer

ASCEND = 1

batch_size = 3
cls_num = 2
image_size = 2

image = fluid.layers.data(name='image', shape=[image_size], dtype='float32', stop_gradient=False)
index = fluid.layers.data(name="index", shape=[], dtype="int32", stop_gradient=False)
label = fluid.layers.data(name='label', shape=[1], dtype='int64', stop_gradient=False)

#image = fluid.layers.elementwise_mul(image, image)
#image = fluid.layers.elementwise_add(image, image)
#image = fluid.layers.elementwise_div(image, image)
#image = fluid.layers.log(image)
#image = fluid.layers.tanh(image)
#image = fluid.layers.pow(image, 2)
image = fluid.layers.sqrt(image)
if ASCEND == False:
    image = fluid.layers.Print(image, message="image_sqrt")
#image = fluid.layers.reshape(image, [3, 2])
#image = fluid.layers.scatter(image, index, image)
#image = fluid.layers.gather(image, index)
#label = fluid.layers.gather(label, index)
#image = fluid.layers.transpose(image, perm=[0,1])
image = fluid.layers.layer_norm(image, begin_norm_axis=1, param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.02, seed=0), bias_attr=fluid.initializer.Constant(value=2.0))
#image.stop_gradient=True
image2 = fluid.layers.stack([image, image], 0)
#image2, index = fluid.layers.topk(image, 1)
#image2 = fluid.layers.accuracy(index, label)
if ASCEND == False:
    image = fluid.layers.Print(image, message="image_layer_norm")
    image2 = fluid.layers.Print(image2, message="image_stack")



fc0 = fluid.layers.fc(image, size=3, act=None, bias_attr=False, param_attr=fluid.initializer.Constant(value=2.0))
fc1 = fluid.layers.fc(fc0, size=3, act=None, bias_attr=False, param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.02, seed=0))
#fc1 = fluid.layers.fc(fc0, size=cls_num, act='relu', bias_attr=False, param_attr=fluid.initializer.Constant(value=2.0))

if ASCEND == False:
    fc0 = layers.Print(fc0, message="fc0")
    fc1 = layers.Print(fc1, message="fc1")

# CLASS_NUM = 10
# fc1 = fluid.layers.fc(fc0, size=CLASS_NUM, bias_attr=False,param_attr=fluid.initializer.Constant(value=2.0))
# layers.Print(fc1)
cross_entropy = fluid.layers.softmax_with_cross_entropy(fc1, label)
if ASCEND == False:
    cross_entropy = layers.Print(cross_entropy, message="cross_entropy")
cost = fluid.layers.reduce_sum(cross_entropy)
#cost = fluid.layers.log(cost)
#cost = fluid.layers.tanh(cost)
#cost = fluid.layers.pow(cost, 2)
#cost = fluid.layers.sqrt(cost)
#cost = fluid.layers.mean(cost)
if ASCEND == False:
    cost = layers.Print(cost, message="cost")

###############################
fetch_list = [image, fc1, label, image, index]
def optimize(fetch_list):
    optimizer = paddle.optimizer.SGD(learning_rate=0.01)
    if ASCEND:
        optimizer = ascend_optimizer.AscendOptimizer(optimizer, fetch_list=fetch_list)
    optimizer.minimize(cost, fluid.default_startup_program())
    return optimizer

optimizer = optimize(fetch_list)
#optimizer.minimize(cost, fluid.default_startup_program())
#optimization(
#                loss=cost,
#                warmup_steps=args.warmup_steps,
#                num_train_steps=args.num_train_steps,
#                learning_rate=args.learning_rate,
#                train_program=fluid.default_main_program(),
#                startup_prog=fluid.default_startup_program(),
#                weight_decay=args.weight_decay,
#                scheduler=args.lr_scheduler,
#                use_fp16=args.use_amp,
#                use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
#                init_loss_scaling=args.init_loss_scaling,
#                incr_every_n_steps=args.incr_every_n_steps,
#                decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
#                incr_ratio=args.incr_ratio,
#                decr_ratio=args.decr_ratio,
#                fetch_list=fetch_list,
#                ascend=ASCEND,
#                dist_strategy=None)

# 打印网络
with open('mnist.start.program', 'w') as fout:
    print(str(fluid.default_startup_program()), file=fout)
with open('mnist.main.program', 'w') as fout:
    print(str(fluid.default_main_program()), file=fout)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

def print_out(out_var, dtype=np.float32):
    data = np.array(out_var, dtype=np.uint8)
    b_arr = data.tobytes()
    arr_2 = np.frombuffer(b_arr, dtype=dtype)
    print(arr_2.shape)
    return arr_2

np.random.seed(1)
for i in range(1):
    out = exe.run(fluid.default_main_program(), 
        #feed={"image": (np.random.rand(batch_size, image_size)-0.5).astype("float32").reshape(batch_size, image_size), 
                #"label": np.array([np.random.randint(0,2) for i in range(batch_size)]).astype("int64").reshape(batch_size, 1)},
        feed={"image": np.array([0.2, 0.3, 0.7, 0.1, 0.7, 0.9]).astype("float32").reshape(batch_size, image_size),
        "index": np.array([0,1,2]).astype("int32").reshape(3),
        "label": np.array([0,1,0]).astype("int64").reshape(3,1)},
        fetch_list = fetch_list)

    if not ASCEND:
        for i in out:
            print(i)

    
    # if ASCEND == False and i == 0:
    #     hehe = ["fc_0.w_0"]
    #     for var in fluid.default_main_program().list_vars():
    #         print("var:", var.name)
    #         if var.name not in hehe:
    #             continue
    #         tmp = fluid.global_scope().find_var(var.name).get_tensor()
    #         print("var[%s]: " % (var.name), tmp)

    if ASCEND:
        print("len(out): ", len(out))
        print("type(out[0]): ", type(out[0]))
        for i, out_var in enumerate(out[:-1]):
            print("%d th output is %s" % (i, str(print_out(out_var))))
        print("%d th output is %s" % (i, str(print_out(out_var, np.int32))))



print("script done")

# 0 th output is [1.239831  1.0189247]
# 1 th output is [2.2587557]
