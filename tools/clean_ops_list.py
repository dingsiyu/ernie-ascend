cut_ops_list = []
with open("./ops_list.log", "r+") as fr, open("./clean_ops_list.log", "w+") as fw_1:
    lines = fr.readlines()
    begin_print, end_print = 0, 0
    for line in lines:
        if "ops" in line:
            begin_print = 1

        if begin_print:
            fw_1.write(line)

        if "type: " in line and begin_print == 1:
            fw_1.write("  }"+"\n")
            fw_1.write("==================================================" + "\n")
            cut_ops_list.append(line.strip())
            begin_print = 0

ops_list = set(cut_ops_list)
forwards_op, backwards_op = 0, 0
with open("./cut_ops_list.log", "w+") as fw:
    for ops in ops_list:
        if "grad" not in ops:
            forwards_op +=1
            fw.write(ops + "\n")
    fw.write("number of forwards_op: " + str(forwards_op) + "\n")
    
    for ops in ops_list:
        if "grad" in ops:
            backwards_op +=1
            fw.write(ops + "\n")
    fw.write("number of backwards_op: " + str(backwards_op) + "\n")
