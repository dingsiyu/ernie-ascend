root="./data//sent_pair_ids_15cls_mix/"
with open("1", "w+") as fw:
    for i in range(1000):
        if i < 10:
            str_name = root + "part-0000" + str(i) + ".gz"
        elif i >= 10 and i < 100:
            str_name = root + "part-000" + str(i) + ".gz"
        else:
            str_name = root + "part-00" + str(i) + ".gz"

        fw.write(str_name + "\t" + "-1.0" + "\n")


