import os
import numpy as np
import argparse
import paddle.fluid as fluid
emb_size=768


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--from_dir", type=str, required=True,
    help="The dir includes the param.")
  parser.add_argument(
    "--param_name", type=str, required=True,
    help="The param name.")

  parser.add_argument(
    "--to_dir", type=str, required=True,
    help="The dir to store the new parameter.")
  parser.add_argument('--two_lines', nargs='+', type=int, help="The two lines to be exchanged")
  args=parser.parse_args()
  return args

def exchange(args):
  program = fluid.Program()
  global_block = program.global_block()
  
  global_block.create_parameter(name=args.param_name, 
                               shape=[3072, 768], 
                               dtype='float32',
                               initializer=fluid.initializer.Constant(value=0.00)) 

  place = fluid.core.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(program)

  param_path = os.path.join(args.from_dir, args.param_name)
  if os.path.exists(param_path):
    print('Load param from %s' % param_path)
    fluid.io.load_params(exe, args.from_dir, main_program=program, filename=args.param_name)
  else:
    raise IOError("%s doesn't exist" % param_path)

  np_value = np.array(fluid.global_scope().find_var(args.param_name).get_tensor())
  #np.savetxt("./emb.txt", np_value)

  print("before trans shape:{}".format(np_value.shape))
  print("before trans value:{}".format(np_value))

  new_np_value = np_value[:2048, :]
  #print("before trans shape:{}".format(new_np_value.shape))
  #print("before trans value:{}".format(new_np_value))

  #print("\nfirst 10 elems before exchange:")
  #print("line %d" % args.two_lines[0])
  #print(np_value[args.two_lines[0]][0:10])
  #print("line %d" % args.two_lines[1])
  #print(np_value[args.two_lines[1]][0:10])

  # exchange two lines
  #np_value[args.two_lines, :] = np_value[[args.two_lines[1], args.two_lines[0]], :]
  
  #print("\nfirst 10 elems after exchange:")
  #print("line %d" % args.two_lines[0]) 
  #print(np_value[args.two_lines[0]][0:10])
  #print("line %d" % args.two_lines[1]) 
  #print(np_value[args.two_lines[1]][0:10])
  
  fluid.global_scope().find_var(args.param_name).get_tensor().set(new_np_value, place)

  fluid.io.save_params(exe, args.to_dir, main_program=program, filename=args.param_name)
  print("\nWrite param to %s" % os.path.join(args.to_dir, args.param_name))  
  
if __name__ == '__main__':
  args = parse_args()
  #if len(args.two_lines) != 2:
  #  raise ValueError("length of `arg.two_lines` should be 2", args.two_lines)
  exchange(args)

