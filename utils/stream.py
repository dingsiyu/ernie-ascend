#!/usr/bin/env python
# -*- coding:utf8 -*-

############################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
############################################################


import re
import commands
import urllib2

from HTMLParser import HTMLParser


def scan_link(url):
    f = urllib2.urlopen(url)
    data = f.read()
    
    class MyHTMLParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.models = []
        
    	def handle_data(self, data):
            if re.match('step', data.strip()):
                self.models.append(data.strip())
    
    parser = MyHTMLParser()
    models = parser.feed(data)
    return parser.models


def stream_by_running(stream_job):
        #test
        cluster, job_id = stream_job.split(":")
        get_host_name = 'scontrol -M ' + cluster \
                + ' show job ' + job_id \
                + ' | grep NodeList | tail -1 | cut -f2 -d"="' \
                + ' | cut -f1 -d","'
        
        #only for local test
        #get_host_name = 'showjob -j ' + job_id \
        #        + ' -p ' + queue \
        #        + ' | grep NodeList | tail -1 | cut -f2 -d"="' \
        #        + ' | cut -f1 -d","'

        retcode, host_name = commands.getstatusoutput(get_host_name)
        if retcode != 0:
            print(host_name)
        print(host_name)
        url_link = 'http://' + host_name + \
                ':8882/look/overview/dir?id=' + job_id + \
                '&file=/job-' + job_id + '/output'
        print(url_link)
        model_list = scan_link(url_link)
        model_pairs = []
        for i in model_list:
            i_s = i.split("_")
            if len(i_s) != 3:
                continue
            model_pairs.append((i, int(i_s[1])))
        
        model_pairs = sorted(model_pairs, key=lambda pair: pair[1])

        download_pairs = []
        for tar, steps in model_pairs:
            download = 'wget http://' + host_name + ':8882/downloadfile/job-' \
                    + job_id + '/output/' + tar
            tar_model = 'tar -xvf ' + tar
            model = 'step_' + str(steps)
            download_pairs.append((download, tar_model, model))
        return download_pairs
            

def stream_by_stop(hdfs_output):
        hadoop_bin = 'hadoop fs -D hadoop.job.ugi=${ugi} -D fs.default.name=${hdfs_path}'
        
        retcode, ugi = commands.getstatusoutput('source ./model_conf; echo $ugi')
        retcode, hdfs_path = commands.getstatusoutput('source ./model_conf; echo $hdfs_path')
        
        hadoop_bin = 'hadoop fs -D hadoop.job.ugi=' + ugi \
                + ' -D fs.default.name=' + hdfs_path
        ls_hadoop = hadoop_bin + " -ls " + hdfs_output \
                + " | awk '{if(NF == 8) print $NF}'"
        print(ls_hadoop)
        retcode, ret = commands.getstatusoutput(ls_hadoop)
        print(ret)
        model_pairs = []
        for i in ret.split("\n"):
            tar = i.split("/")[-1]
            tar_s = tar.split("_")
            if len(tar_s) != 3:
                continue
            model_pairs.append((i, tar, int(tar_s[1])))
        
        model_pairs = sorted(model_pairs, key=lambda pair: pair[2])
        print(model_pairs)

        download_pairs = []
        for path, tar, steps in model_pairs:
            download = hadoop_bin + " -get " + path + " ./"
            tar_model = 'tar -xvf ' + tar
            model = 'step_' + str(steps)
            download_pairs.append((download, tar_model, model))
        return download_pairs
